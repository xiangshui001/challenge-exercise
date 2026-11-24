"""Recompute the frozen confirmation outcomes with Python 3 and NumPy.

Run beside the artifacts, or pass --root DIRECTORY. --self-test checks synthetic
pairing, exact probabilities, score directions, and source validation.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4

import numpy as np


ARMS = ("original", "wrong1", "wrong2", "wrong3", "near", "far", "text")
ALT = ARMS[1:6]
MODELS = ("qwen", "siglip", "clip")
CATS = ("replace_att", "replace_obj", "replace_rel", "swap_att", "swap_obj")
REPEATS = 10000
SEED = 20261122


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def lines(data):
    return [json.loads(x) for x in data.splitlines()]


def json_bytes(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def write_if_match(path, data):
    if path.exists():
        require(path.read_bytes() == data, f"Existing derived file differs: {path.name}")
    else:
        path.write_bytes(data)


def load_inputs(root):
    freeze = load(root / "freeze.json")
    sims_bytes = (root / "similarities.json").read_bytes()
    require(sha(sims_bytes) == freeze["similarities_sha256"], "Similarity SHA mismatch")
    sims = json.loads(sims_bytes)
    canonical = root / "manifest.audit.jsonl"
    binding = root / "manifest.audit.binding.json"
    private = root / "manifest.private.jsonl"
    if private.is_file():
        source = private.read_bytes()
        require(sha(source) == freeze["manifest_sha256"], "Private manifest SHA mismatch")
        audit = []
        for row in lines(source):
            entry = {k: row[k] for k in ("id", "cat", "img", "sha256", "wrong", "near", "far")}
            entry["text_sha256"] = [sha(t.encode()) for t in row["texts"]]
            audit.append(entry)
        data = b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in audit)
        write_if_match(canonical, data)
        write_if_match(binding, json_bytes({
            "manifest_sha256": freeze["manifest_sha256"], "audit_sha256": sha(data)}))
    data = canonical.read_bytes()
    ref = load(binding)
    require(ref["manifest_sha256"] == freeze["manifest_sha256"] and
            ref["audit_sha256"] == sha(data), "Public/private manifest commitment mismatch")
    manifest = lines(data)
    n = len(manifest)
    ids = [r["id"] for r in manifest]
    pics = {r["img"]: r for r in manifest}
    require(n == len(set(ids)) == len(pics) == len({r["sha256"] for r in manifest}) == 150,
            "Unexpected cohort count or duplicate identities")
    require(set(ids) == set(sims), "Incomplete similarity coverage")
    cat = np.array([r["cat"] for r in manifest])
    require(set(cat) == set(CATS), "Unexpected edit categories")
    groups = {s: np.flatnonzero(cat == s) for s in CATS}
    require(all(len(g) == 30 for g in groups.values()), "Unexpected stratum count")
    for r in manifest:
        partners = [*r["wrong"], r["near"], r["far"]]
        require(len(partners) == 5 and len({r["img"], *partners}) == 6 and
                all(pics[p]["cat"] == r["cat"] for p in partners), "Image partner mismatch")
        require(len(r["text_sha256"]) == len(set(r["text_sha256"])) == 3,
                "Text identity mismatch")
    for g in groups.values():
        des = [manifest[i] for i in g]
        src = {r["img"] for r in des}
        for j in range(5):
            require({[*r["wrong"], r["near"], r["far"]][j] for r in des} == src,
                    "Image assignment is not one-to-one within stratum")
    require(all(set(sims[k]) == set(ALT) for k in ids), "Similarity arm mismatch")
    similarities = np.array([[sims[k][arm] for arm in ALT] for k in ids], dtype=float)
    require(np.isfinite(similarities).all() and (np.abs(similarities) <= 1.00001).all(),
            "Invalid selector cosine")
    all_scores = {}
    input_hashes = {}
    for model in MODELS:
        run = load(root / f"full_{model}.run.json")
        raw = (root / f"full_{model}.jsonl").read_bytes()
        require(run["model"] == model and run["run"] == f"full_{model}" and
                run["n"] == n and sha(raw) == run["predictions_sha256"],
                f"Unsealed or incompatible predictions for {model}")
        for key in ("infer_sha256", "manifest_sha256", "protocol_sha256"):
            require(run[key] == freeze[key], f"Incompatible {key} for {model}")
        rows = lines(raw)
        require([r["id"] for r in rows] == ids, f"Missing/reordered units: {model}")
        for m, r in zip(manifest, rows):
            require(all(r[k] == m[k] for k in ("cat","sha256","wrong","near","far","text_sha256")),
                    f"Mismatched {model} input for {m['id']}")
            require(set(r["scores"]) == set(ARMS), f"Incomplete {model} arm")
        scores = np.array([[r["scores"][arm] for arm in ARMS] for r in rows],dtype=float)
        require(scores.shape == (n,7,3) and np.isfinite(scores).all(), f"Bad scores: {model}")
        all_scores[model] = scores
        input_hashes[model] = sha(raw)
    for name, key in (("protocol.txt","protocol_sha256"),("infer.py","infer_sha256"),
                      ("prepare_confirm.py","prepare_sha256"),("assign.py","assign_sha256")):
        path = root/name
        if path.is_file():
            require(sha(path.read_bytes()) == freeze[key], f"Frozen code/protocol changed: {name}")
    return manifest, groups, similarities, all_scores, input_hashes, freeze


def mcnemar(fixed, lost):
    n = fixed + lost
    if n == 0:
        return 1.0
    return min(1., 2. * sum(math.comb(n,j) for j in range(min(fixed,lost)+1)) / (2**n))


def draws_for(groups):
    rng = np.random.default_rng(SEED)
    return np.column_stack([rng.choice(g,size=(REPEATS,len(g)),replace=True) for g in groups.values()])


def diff_summary(values, draws):
    delta = np.asarray(values,dtype=float)
    return {"mean":float(delta.mean()),"stratified_image_bootstrap_ci95":
        np.quantile(delta[draws].mean(axis=1),[.025,.975]).tolist()}


def counts(m):
    n = len(m)
    won = m > 0
    wins = won.sum(axis=1)
    return dict(images=n,contests=2*n,both_won=int((wins==2).sum()),one_won=int((wins==1).sum()),
                neither_won=int((wins==0).sum()),pair_wins=int(won.sum()),pair_ties=int((m==0).sum()))


def compare(base, other, draws):
    b, a = (base>0).all(axis=1), (other>0).all(axis=1)
    fixed, lost = int((~b&a).sum()), int((b&~a).sum())
    return {"base":counts(base),"comparison":counts(other),
            "comparison_only_images":fixed,"base_only_images":lost,
            "image_delta_proportion":diff_summary(a.astype(float)-b,draws),
            "mcnemar_exact_two_sided":mcnemar(fixed,lost),
            "comparison_only_pairs":int(((base<=0)&(other>0)).sum()),
            "base_only_pairs":int(((base>0)&(other<=0)).sum()),
            "pair_delta_proportion":diff_summary(((other>0).mean(1)-(base>0).mean(1)),draws)}


def retains(original, wrong):
    o, w = original<=0, wrong<=0
    return dict(original_failed=int(o.sum()), still_failed=int((o&w).sum()),
                switched_from_failed_to_win=int((o&~w).sum()),
                switched_from_win_to_failed=int((~o&w).sum()),both_win=int((~o&~w).sum()))


def direct(m):
    delta = m[:,0,None,:]-m[:,1:4,:]
    majority=(delta>0).sum(axis=1)>=2
    return dict(observations=int(delta.size),larger=int((delta>0).sum()),ties=int((delta==0).sum()),
                median=float(np.median(delta)),mean=float(delta.mean()),
                pair_majority=int(majority.sum()),n_pairs=int(majority.size),
                image_both_majority=int(majority.all(axis=1).sum()),n_images=len(delta))


def yes_counts(s):
    out={}
    for label,x in (("all_candidates",s),("foil",s[:,2]),("references",s[:,:2])):
        out[label]={"yes":int((x>0).sum()),"no":int((x<0).sum()),"tie":int((x==0).sum()),
                    "n":int(x.size)}
    return out


def similarity_summary(s,draws):
    return {"arms":{arm:{"n":len(s),"mean":float(s[:,i].mean()),"median":float(np.median(s[:,i]))}
                    for i,arm in enumerate(ALT)},
            "near_minus_far":diff_summary(s[:,3]-s[:,4],draws),
            "near_above_far":int((s[:,3]>s[:,4]).sum()),"near_equal_far":int((s[:,3]==s[:,4]).sum()),
            "n":len(s)}


def describe(scores, groups, draws, model):
    m = scores[:,:,:2]-scores[:,:,2,None]
    original = m[:,0]
    near,far=m[:,4],m[:,5]
    result={"arms":{a:counts(m[:,i]) for i,a in enumerate(ARMS)},
            "direct_original_minus_random":direct(m),
            "near_minus_far_rank":compare(far,near,draws),
            "wrong_and_text_vs_original_rank":{a:retains(original,m[:,i])
                                             for i,a in enumerate(ARMS) if i},
            "categories":{}}
    o=original<=0
    often=(m[:,1:4]<=0).sum(axis=1)>=2
    result["random_majority"]={"original_failed":int(o.sum()),"same_fail_random_majority":int((o & often).sum()),
        "same_fail_majority_and_text":int((o & often & (m[:,6]<=0)).sum())}
    for key,g in groups.items():
        result["categories"][key]={"arms":{arm:counts(m[g,i]) for i,arm in enumerate(ARMS)},
            "direct_original_minus_random":direct(m[g])}
    if model=='qwen':
        result["absolute_yes"]= {a:yes_counts(scores[:,i]) for i,a in enumerate(ARMS)}
        yes_near=scores[:,4,:]>0
        yes_far=scores[:,5,:]>0
        result["near_minus_far_yes"]={}
        for label,selected in (("all_candidates",[0,1,2]),("foil",[2]),("references",[0,1])):
            yn, yf=yes_near[:,selected],yes_far[:,selected]
            result["near_minus_far_yes"][label]={
                "n":int(yn.size),"near_yes":int(yn.sum()),"far_yes":int(yf.sum()),
                "near_only":int((yn&~yf).sum()),"far_only":int((yf&~yn).sum()),
                "delta_proportion":diff_summary(yn.mean(1)-yf.mean(1),draws)}
        for cat,g in groups.items():
            result["categories"][cat]["near_yes"]=yes_counts(scores[g,4])
            result["categories"][cat]["far_yes"]=yes_counts(scores[g,5])
    return result


def corrected_scores(scores):
    return {"minus_no_image":scores[:,0]-scores[:,6],
            "minus_three_random_mean":scores[:,0]-scores[:,1:4].mean(axis=1)}


def compute(root):
    manifest,groups,sims,data,hashes,freeze=load_inputs(root)
    draws=draws_for(groups)
    Q=data["qwen"]
    raw=Q[:,0,:2]-Q[:,0,2,None]
    methods={}
    for name,score in corrected_scores(Q).items():
        margins=score[:,:2]-score[:,2,None]
        methods[name]=compare(raw,margins,draws)
        methods[name]["per_category"]={}
        for c,g in groups.items():
            a,b=(margins[g]>0).all(1),(raw[g]>0).all(1)
            methods[name]["per_category"][c]={"base":counts(raw[g]),"corrected":counts(margins[g]),
               "repaired_images":int((a & ~b).sum()),"regressed_images":int((b & ~a).sum())}
    p=methods['minus_no_image']
    result={"case_count":len(manifest),"bootstrap_draws":REPEATS,"bootstrap_seed":SEED,
            "prediction_sha256":hashes,"manifest_sha256":freeze["manifest_sha256"],
            "protocol_sha256":freeze["protocol_sha256"],"similarities_sha256":freeze["similarities_sha256"],
            "similarities":similarity_summary(sims,draws),"qwen_comparisons":methods,
            "primary_superiority_criterion_met":bool(p['image_delta_proportion']['mean']>0 and p['mcnemar_exact_two_sided']<.05),
            "models":{k:describe(v,groups,draws,k) for k,v in data.items()}}
    (root/"confirm_summary.json").write_bytes(json_bytes(result))
    rows=[]
    for model,s in data.items():
        corr=corrected_scores(s) if model=='qwen' else {}
        for i,row in enumerate(manifest):
            audit={"id":row['id'],"model":model,"category":row['cat'],"image_sha256":row['sha256']}
            for j,arm in enumerate(ALT):audit[arm+"_selector_cosine"]=sims[i,j]
            for j,arm in enumerate(ARMS):
                for k,tag in enumerate(('ref1','ref2','foil')):audit[arm+'_'+tag]=s[i,j,k]
                margin=s[i,j,:2]-s[i,j,2]
                audit[arm+'_both_win']=int((margin>0).all())
            for name in ('minus_no_image','minus_three_random_mean'):
                audit[name+'_both_win']=int((corr[name][i,:2]>corr[name][i,2]).all()) if name in corr else ''
            rows.append(audit)
    with (root/'confirm_audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return result


def fixture(root):
    rng=np.random.default_rng(37);units=[];sim={}
    for c in CATS:
        for i in range(30):
            key=f"{c}:{i}"
            pic=lambda j:f"{c}/{j%30}.jpg"
            row={"cat":c,"id":key,"img":pic(i),"sha256":sha(key.encode()),"texts":[f"{key}/{j}" for j in range(3)],
                 "wrong":[pic(i+j) for j in (1,2,3)],"near":pic(i+4),"far":pic(i+5)}
            units.append(row);sim[key]=dict(zip(ALT,[.4,.3,.5,.8,.1]))
    b=b"".join((json.dumps(r)+"\n").encode() for r in units)
    (root/'manifest.private.jsonl').write_bytes(b)
    sb=json_bytes(sim);(root/'similarities.json').write_bytes(sb)
    h=dict(manifest_sha256=sha(b),infer_sha256='synthetic',protocol_sha256='synthetic',similarities_sha256=sha(sb))
    (root/'freeze.json').write_bytes(json_bytes(h))
    for model in MODELS:
        data=[]
        for i,r in enumerate(units):
            s={a:rng.normal(size=3).tolist() for a in ARMS}
            s['original']=[1.,1.,0.] if i%2 else [-1.,1.,0.]
            s['text']=[2.,0.,0.] if i%2 else [-2.,0.,0.]
            s['near']=[3.,3.,1.];s['far']=[-1.,-1.,-2.]
            d={k:r[k] for k in ('id','cat','sha256','wrong','near','far')}
            d.update(text_sha256=[sha(t.encode()) for t in r['texts']],scores=s);data.append(d)
        b=b"".join((json.dumps(r)+"\n").encode() for r in data)
        (root/f"full_{model}.jsonl").write_bytes(b)
        (root/f"full_{model}.run.json").write_bytes(json_bytes({**h,"model":model,"run":f"full_{model}",
            "n":150,"predictions_sha256":sha(b)}))


def self_test():
    root=Path(__file__).resolve().parent/("audit-fixture-"+uuid4().hex)
    root.mkdir()
    try:
        fixture(root)
        s=compute(root);p=s['qwen_comparisons']['minus_no_image']
        require(p['base']['both_won']==75==p['comparison']['both_won'], "Fixture accuracy")
        require(p['comparison_only_images']==75==p['base_only_images'] and p['mcnemar_exact_two_sided']==1, "Fixture pairs")
        q=s['models']['qwen']['near_minus_far_yes']['foil']
        require(q['near_only']==150 and q['far_only']==0 and q['delta_proportion']['stratified_image_bootstrap_ci95']==[1.,1.], "Fixture foil")
        require(mcnemar(0,5)==0.0625 and not s['primary_superiority_criterion_met'], "Exact p")
        (root/'manifest.private.jsonl').unlink()
        require(compute(root)==s, "Public-only reproduction")
        f=root/'full_qwen.jsonl';src=f.read_bytes();qrows=lines(src);qrows[0]['near']=qrows[0]['far']
        bad=b"".join((json.dumps(r)+"\n").encode() for r in qrows);f.write_bytes(bad)
        try:load_inputs(root)
        except ValueError as exc:require('Unsealed' in str(exc),'Expected seal failure')
        else:raise AssertionError("Missing seal failure")
        mpath=root/'full_qwen.run.json';meta=load(mpath);meta['predictions_sha256']=sha(bad);mpath.write_bytes(json_bytes(meta))
        try:load_inputs(root)
        except ValueError as exc:require('Mismatched' in str(exc),'Expected identity failure')
        else:raise AssertionError("Missing row identity failure")
    finally:
        for path in root.iterdir():
            if path.is_file():path.unlink()
        root.rmdir()
    print("Self-test passed (private/public parity, source seal, row identities, paired statistics).")


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test()
    else:
        stats=compute(args.root)
        primary=stats['qwen_comparisons']['minus_no_image']
        print('primary:',primary['base']['both_won'],primary['comparison']['both_won'],
              'paired:',primary['comparison_only_images'],primary['base_only_images'],
              'p:',primary['mcnemar_exact_two_sided'],'met:',stats['primary_superiority_criterion_met'])
