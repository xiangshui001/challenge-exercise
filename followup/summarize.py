"""Validate per-unit identities and coverage, then recompute paired statistics."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import numpy as np

from followup.run import NATURAL, SPATIAL, REFERENCE, LABELS
from m0.validate import WEIGHTS
from m0.metrics import sugar
from m0.prepare import save


def read(path):
    return [json.loads(l) for l in path.read_text(encoding="utf8").splitlines()]


def rate(values):
    n = len(values)
    k = sum(map(int, values))
    p = k/n
    z = 1.95996398454
    center = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return dict(count=k, n=n, percent=round(100*p, 2),
                wilson95=[round(100*max(0,center-radius),2),round(100*min(1,center+radius),2)])


def paired(left, right):
    diff = np.asarray(left, dtype=int)-np.asarray(right, dtype=int)
    draws = np.random.default_rng(20261119).choice(diff, (2000,len(diff))).mean(axis=1)
    return dict(delta_pp=round(100*float(diff.mean()),2),
                ci95_pp=np.round(100*np.quantile(draws,[.025,.975]),2).tolist(),
                increases=int((diff>0).sum()),decreases=int((diff<0).sum()))


def summarize(root):
    audit=json.loads((root/"audit.json").read_text())
    assert not audit["duplicates"]
    items={r["sample_id"]:r for r in audit["registered"]}
    assert len(items)==len(audit["registered"])==780
    original={r["sample_id"]:r for r in read(root.parent/"m0-20261102"/"manifest.jsonl")}
    old_images=json.loads((root.parent/"m0-20261102"/"image-audit.json").read_text())["files"]
    refs={r["sample_id"] for r in read(root/"ai-reference.jsonl") if r["sample_id"].startswith("esnli/")}
    assert len(refs)==13
    scores={}
    for model, bench, dataset, digest in (
            ("siglip","scpp","natural",NATURAL),("clip","scpp","natural",NATURAL),
            ("qwen","esnli","natural",NATURAL),("qwen","clevr","spatial",SPATIAL),
            ("qwen","reference","reference",REFERENCE)):
        directory=root/f"{model}-{dataset}"
        meta=json.loads((directory/"run.json").read_text())
        assert meta["manifest_sha256"]==digest
        assert all(meta["model_files"][f]==v for f,v in WEIGHTS[model].items())
        ids=refs if bench=="reference" else {i for i,r in items.items() if r["benchmark"]==bench}
        rows=read(directory/"predictions.jsonl")
        arms={"esnli":{"original","paraphrase","explicit"},"clevr":{"image/0","image/1","no_image/0","no_image/1"},
              "reference":{"K0","K1","K2","Kshuffle"}}.get(bench,{"rank"})
        seen=set()
        for row in rows:
            i=row["sample_id"]
            arm=row.get("arm","rank")
            assert i in ids and arm in arms and (i,arm) not in seen
            seen.add((i,arm))
            item=(original if bench=="reference" else items)[i]
            image_hash=old_images[item["images"][0]]["sha256"] if bench=="reference" else item["image_sha256"]
            if arm=="rank":
                assert row["status"]=="ok" and row["image_sha256"]==[image_hash]
                assert row["text_sha256"]==item["text_sha256"] and row["category"]==item["category"]
            else:
                assert row["image_sha256"]==(None if arm.startswith("no_image") else image_hash)
                assert row["gold"]==(item["gold"][int(arm[-1])] if bench=="clevr" else item["gold"])
        assert len(seen)==len(ids)*len(arms)
        if bench=="scpp":
            groups=defaultdict(list)
            for row in rows:
                v=sugar(row["scores"])
                groups["overall"].append(v)
                groups[row["category"]].append(v)
            scores[model]={k:{m:rate([v[m] for v in vs]) for m in ("itt","xor","both_wrong")} for k,vs in groups.items()}
            continue
        by={a:{r["sample_id"]:r for r in rows if r["arm"]==a} for a in sorted(arms)}
        ids=sorted(ids)
        if bench=="clevr":
            stats={}
            both={}
            for mode in ("image","no_image"):
                a,b=([by[f"{mode}/{j}"][i] for i in ids] for j in (0,1))
                both[mode]=[x["prediction"]==x["gold"] and y["prediction"]==y["gold"] for x,y in zip(a,b)]
                stats[mode]=dict(pair=rate(both[mode]),question=rate([x["prediction"]==x["gold"] for x in a+b]),
                                 same=rate([x["prediction"]==y["prediction"] for x,y in zip(a,b)]),
                                 yes=rate([x["prediction"]=="yes" for x in a+b]),
                                 invalid=sum(x["prediction"]=="invalid" for x in a+b))
            stats["image_effect"]=paired(both["image"],both["no_image"])
        else:
            stats={}
            for arm, per in by.items():
                stats[arm]=dict(accuracy=rate([v["prediction"]==v["gold"] for v in per.values()]),
                    confusion={label:dict(Counter(v["prediction"] for v in per.values() if v["gold"]==label)) for label in LABELS})
            if bench=="esnli":
                for arm in ("paraphrase","explicit"):
                    stats[arm]["paired_error_change"]={}
                    for g,p in (("contradiction","neutral"),("neutral","contradiction")):
                        subset=[i for i in ids if by[arm][i]["gold"]==g]
                        stats[arm]["paired_error_change"][g+"_to_"+p]=paired(
                            [by[arm][i]["prediction"]==p for i in subset],[by["original"][i]["prediction"]==p for i in subset])
            else:
                stats["direction_flips"]=sum(by["K2"][i]["prediction"]!=by["Kshuffle"][i]["prediction"] for i in ids)
        scores[bench]=stats
    return scores


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    a=parser.parse_args()
    save(a.root/"summary.json",summarize(a.root))

