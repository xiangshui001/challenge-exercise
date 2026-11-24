"""Secondary replication of the November 22 lexical and dual-encoder diagnostics.

Run beside freeze.json, manifest.private.jsonl, and the sealed full dual-encoder predictions.
Requires NumPy. Publishes anonymous counts only. The three random wrong images,
not the extra near/far interventions, define persistence in both cohorts.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np


SEED, DRAWS = 2026112204, 30000
ARMS = ["original","wrong1","wrong2","wrong3","text"]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def permutation(first, second, strata, seed):
    groups = defaultdict(list)
    for i,key in enumerate(strata):
        groups[key].append(i)
    rng = np.random.default_rng(seed)
    null = np.zeros(DRAWS,dtype=int)
    for start in range(0,DRAWS,500):
        part = null[start:start+500]
        for g in groups.values():
            perm = np.argsort(rng.random((len(part),len(g))),axis=1)
            part += (first[g][None] & second[g][perm]).sum(axis=(1,2))
    obs = int((first & second).sum())
    return {"observed":obs, "expected_exact":sum(float(first[g].sum(0) @ second[g].sum(0))/len(g) for g in groups.values()),
            "central_95":np.quantile(null,[.025,.975],method="inverted_cdf").tolist(),
            "p_upper_plus_one":float((1+np.count_nonzero(null>=obs))/(DRAWS+1)), "strata":len(groups)}


def summarize(manifest, rows_by_model):
    cat = [r["cat"] for r in manifest]
    jac = []
    for r in manifest:
        t = [set(re.findall(r"\w+",s.casefold())) for s in r["texts"]]
        jac.append([len(v&t[2])/len(v|t[2]) for v in t[:2]])
    jac=np.asarray(jac)
    result={"n":len(cat), "pairs":2*len(cat), "seed":SEED, "permutations":DRAWS,
            "lexical":{"ref1_closer":int((jac[:,0]>jac[:,1]).sum()),"ref2_closer":int((jac[:,0]<jac[:,1]).sum()),
                       "tie":int((jac[:,0]==jac[:,1]).sum()),"mean_jaccard_by_reference":jac.mean(0).tolist()},
            "model_errors_by_reference":{},"cross_model":{}}
    errors={}
    for model,rows in rows_by_model.items():
        scores=np.asarray([[r["scores"][a] for a in ARMS] for r in rows],dtype=float)
        if scores.shape!=(len(cat),5,3) or not np.isfinite(scores).all():
            raise ValueError("Invalid score matrix")
        failed=scores[:,:,:2]<=scores[:,:,2,None]
        persist=failed[:,0] & (failed[:,1:4].sum(1)>=2)
        errors[model]={"original":failed[:,0],"text":failed[:,4],"persistent":persist,
                       "joint_text":persist&failed[:,4]}
        result["model_errors_by_reference"][model]={k:v.sum(0).tolist() for k,v in errors[model].items()}
    A,B=errors["siglip"],errors["clip"]
    for k,arm in enumerate(["original","persistent","joint_text"]):
        both=A[arm]&B[arm]
        obj={"common_pairs":int(both.sum()),"common_images":int(both.any(1).sum()),
             "union_pairs":int((A[arm]|B[arm]).sum()),"by_reference":both.sum(0).tolist(),
             "permute_clip_within_category":permutation(A[arm],B[arm],cat,SEED+k)}
        if arm!="original":
            for fixed,shuffled,name,salt in [(A,B,"clip",10),(B,A,"siglip",20)]:
                strata=list(zip(cat,shuffled["original"][:,0],shuffled["original"][:,1]))
                obj["permute_"+name+"_within_category_and_its_original_mask"]=permutation(
                    fixed[arm],shuffled[arm],strata,SEED+salt+k)
        result["cross_model"][arm]=obj
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    d=p.parse_args().root
    freeze=json.loads((d/"freeze.json").read_bytes())
    data=(d/"manifest.private.jsonl").read_bytes()
    if sha(data)!=freeze["manifest_sha256"]:
        raise ValueError("Manifest does not match its freeze")
    manifest=sorted(map(json.loads,data.splitlines()),key=lambda r:r["id"])
    cats=np.unique([r["cat"] for r in manifest],return_counts=True)[1]
    if len(manifest)!=150 or len({r["id"] for r in manifest})!=150 or sorted(cats)!=[30]*5:
        raise ValueError("Unexpected cohort")
    rows_by_model={}
    for model in ("siglip","clip"):
        raw=(d/f"full_{model}.jsonl").read_bytes()
        run=json.loads((d/f"full_{model}.run.json").read_bytes())
        if not (run["model"]==model and run["n"]==150 and run["predictions_sha256"]==sha(raw)
                and run["manifest_sha256"]==freeze["manifest_sha256"] and run["infer_sha256"]==freeze["infer_sha256"]):
            raise ValueError("Model run seal mismatch")
        rows=sorted(map(json.loads,raw.splitlines()),key=lambda r:r["id"])
        if len(rows)!=150:
            raise ValueError("Incomplete predictions")
        for m,r in zip(manifest,rows):
            if not all(m[k]==r[k] for k in ("id","cat","sha256","wrong","near","far")) or [sha(t.encode()) for t in m["texts"]]!=r["text_sha256"]:
                raise ValueError("Misaligned prediction input")
        rows_by_model[model]=rows
    result=summarize(manifest,rows_by_model)
    content=(json.dumps(result,indent=2,allow_nan=False)+"\n").encode()
    target=d/"replication_patterns.json"
    if target.exists() and target.read_bytes()!=content:
        raise ValueError("Existing results differ")
    target.write_bytes(content)
    print("Secondary replication complete.",json.dumps(result["lexical"]))


if __name__=="__main__":
    main()
