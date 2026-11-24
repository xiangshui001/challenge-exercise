"""Exploratory absolute Yes/No threshold diagnostic; the new cohort never selects a threshold."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(root, n):
    path = root / "full_qwen.jsonl"
    data = path.read_bytes()
    meta = json.loads((root / "full_qwen.run.json").read_bytes())
    digest = hashlib.sha256(data).hexdigest()
    if meta["predictions_sha256"] != digest or meta["model"] != "qwen" or meta["n"] != n:
        raise ValueError(f"{root}: Qwen source seal does not match")
    rows = [json.loads(x) for x in data.splitlines()]
    if len(rows) != n or len({r["id"] for r in rows}) != n or len({r["sha256"] for r in rows}) != n:
        raise ValueError(f"{root}: duplicate or missing images")
    scores = np.array([r["scores"]["original"] for r in rows], dtype=np.float64)
    if scores.shape != (n,3) or not np.isfinite(scores).all():
        raise ValueError(f"{root}: unexpected original scores")
    return scores, np.array([r["cat"] for r in rows]), {r["sha256"] for r in rows}, digest


def counts(scores, threshold):
    yes = scores > threshold
    return int(yes[:,:2].sum()), int(yes[:,2].sum()), (yes[:,:2].all(axis=1) & ~yes[:,2])


def select_threshold(scores):
    # The integer numerator avoids float tie errors when comparing balanced accuracies.
    values = np.unique(scores)
    midpoints = values[:-1] + (values[1:] - values[:-1]) / 2
    choices = sorted({0., float(np.nextafter(values[0], -np.inf)),
                      float(np.nextafter(values[-1], np.inf)), *midpoints.tolist()})
    def objective(t):
        ref_yes, foil_yes, _ = counts(scores,t)
        return ref_yes + 2*(len(scores)-foil_yes)
    best_value = max(map(objective,choices))
    tied = [t for t in choices if objective(t) == best_value]
    return min(tied,key=lambda t:(abs(t),t)),len(tied),len(choices)


def evaluate(scores, categories, threshold):
    n = len(scores)
    ranking_wins = (scores[:,:2] > scores[:,2,None]).all(axis=1)
    ref, foil, triple = counts(scores,threshold)
    old_yes = scores>0
    new_yes = scores>threshold
    gold = np.broadcast_to([True,True,False],scores.shape)
    old, new = old_yes==gold, new_yes==gold
    base_triple = old.all(axis=1)
    return {
        "threshold":float(threshold),"images":n,"reference_captions":2*n,"foils":n,
        "both_references_outrank_foil":int(ranking_wins.sum()),"both_win_but_foil_supported":int((ranking_wins & new_yes[:,2]).sum()),
        "references_supported":ref,"foils_supported":foil,"foils_rejected":n-foil,
        "reference_recall":ref/(2*n),"foil_rejection_rate":(n-foil)/n,
        "balanced_accuracy":(ref+2*(n-foil))/(4*n),"images_all_three_absolute_correct":int(triple.sum()),
        "exact_threshold_ties":int((scores==threshold).sum()),"image_repairs_vs_zero":int((~base_triple & triple).sum()),
        "image_regressions_vs_zero":int((base_triple & ~triple).sum()),"caption_repairs_vs_zero":int((~old & new).sum()),
        "caption_regressions_vs_zero":int((old & ~new).sum()),
        "foil_supported_by_type":{str(c):{"yes":int(new_yes[categories==c,2].sum()),"n":int((categories==c).sum())}
                                  for c in sorted(set(categories))}
    }


def main():
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("old_dir",type=Path,nargs="?",default=here.parent/"controls")
    p.add_argument("new_dir",type=Path,nargs="?",default=here)
    p.add_argument("--output",type=Path,default=None)
    a=p.parse_args()
    old,oc,oh,od=read(a.old_dir,200)
    new,nc,nh,nd=read(a.new_dir,150)
    if oh & nh:raise ValueError("Overlapping image hashes")
    threshold,ties,choices=select_threshold(old)
    result={"analysis":"exploratory; the benchmark foil is treated as unsupported for this diagnostic",
            "selection_rule":"Train balanced accuracy; ties nearest zero then smaller value. Yes iff score > threshold.",
            "old_scores_sha256":od,"new_scores_sha256":nd,
            "old_only_threshold":threshold,"candidate_cutoffs":choices,"maximizing_cutoffs":ties,
            "old_200":{"zero":evaluate(old,oc,0),"old_threshold":evaluate(old,oc,threshold)},
            "new_150":{"zero":evaluate(new,nc,0),"old_threshold":evaluate(new,nc,threshold)}}
    (a.output or a.new_dir/"absolute_summary.json").write_bytes((json.dumps(result,indent=2,allow_nan=False)+"\n").encode())
    for k in ("old_200","new_150"):
        for cond,x in result[k].items():
            print(k,cond,x['threshold'],x['both_references_outrank_foil'],x['both_win_but_foil_supported'],
                  x['references_supported'],x['foils_supported'],x['images_all_three_absolute_correct'],
                  x['balanced_accuracy'],x['image_repairs_vs_zero'],x['image_regressions_vs_zero'])


if __name__=="__main__":
    main()
