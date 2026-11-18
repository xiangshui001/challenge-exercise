"""Compare locked AI references with official annotations in the original order."""
import argparse
from collections import Counter
import json
from pathlib import Path

def compare(root):
    data = root.parent/"m0-20261102"/"manifest.jsonl"
    manifest={r["sample_id"]:r for r in map(json.loads,data.read_text().splitlines())}
    counts=Counter()
    cases=[]
    for line in (root/"ai-reference.jsonl").read_text(encoding="utf8").splitlines():
        r=json.loads(line)
        item=manifest[r["sample_id"]]
        b=item["benchmark"]
        counts[b+"_items"]+=1
        if b=="scpp":
            sequence=r["answer"].split(",")
            original=[sequence[r["text_order"].index(i)] for i in (0,1,2)]
            match=original==["T","T","F"]
            counts["scpp_negative_"+original[2]]+=1
        elif b=="vwsd":
            original=r["image_order"][int(r["answer"])]
            match=original==item["gold"]
        else:
            original=r["answer"]
            match=original==item["gold"]
        counts[b+"_match"]+=int(match)
        cases.append(dict(id=r["id"],sample_id=r["sample_id"],benchmark=b,ai_original_order=original,
                          gold=item["gold"],match=match,human_review=None))
    return dict(counts=dict(counts),cases=cases)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    a=p.parse_args()
    (a.root/"review-comparison.json").write_text(json.dumps(compare(a.root),indent=2)+"\n",encoding="utf8",newline="\n")

