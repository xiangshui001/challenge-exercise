"""Audit post-hoc caption score corrections and genuine image-swap contrasts."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "controls"
ARMS = ("original", "wrong1", "wrong2", "wrong3", "text")


def exact_p(fixed, lost):
    discordant = fixed + lost
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(fixed, lost) + 1))
    return min(1.0, 2.0 * tail / 2.0**discordant)


def main():
    f = SOURCE / "full_qwen.jsonl"
    run = json.loads((SOURCE / "full_qwen.run.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(f.read_bytes()).hexdigest() == run["predictions_sha256"]
    rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines()]
    cats = np.array([r["cat"] for r in rows])
    groups = [np.flatnonzero(cats == cat) for cat in sorted(set(cats))]
    assert list(map(len, groups)) == [40] * 5
    a = np.asarray([[r["scores"][k] for k in ARMS] for r in rows])
    assert a.shape == (200, 5, 3)
    raw = a[:, 0]
    reference = np.column_stack([raw[:, 0] - raw[:, 2], raw[:, 1] - raw[:, 2]])
    raw_win = (reference > 0).all(1)
    rng = np.random.default_rng(65191)
    draws = np.column_stack([rng.choice(g, (12000, len(g)), replace=True) for g in groups])
    methods = {
        "raw":raw,
        "text_alone":a[:, 4],
        "raw_minus_text":raw-a[:, 4],
        "raw_minus_per_caption_three_wrong_median":raw-np.median(a[:, 1:4], axis=1),
        "raw_minus_per_caption_three_wrong_mean":raw-a[:, 1:4].mean(axis=1)
    }
    result = {"source_sha256":run["predictions_sha256"], "n_images":len(raw), "n_contests":reference.size}
    outcomes = {}
    for name, s in methods.items():
        m = s[:, :2] - s[:, 2, None]
        win = (m > 0).all(1)
        fix = int((~raw_win & win).sum())
        lose = int((raw_win & ~win).sum())
        d = win.astype(int) - raw_win.astype(int)
        outcomes[name] = dict(
            images_both=int(win.sum()),
            contests_won=int((m > 0).sum()),
            ties=int((m == 0).sum()),
            fix_lose_images=[fix,lose],
            fix_lose_contests=[int(((reference <= 0) & (m > 0)).sum()),int(((reference > 0) & (m <= 0)).sum())],
            delta_image_pct=100 * float(d.mean()),
            delta_image_pct_ci95=(100 * np.quantile(d[draws].mean(1),[0.025,0.975])).tolist(),
            image_mcnemar_p_2s_exact=exact_p(fix,lose))
    result["methods"] = outcomes
    margin = a[:, :, :2] - a[:, :, 2, None]
    wrong = margin[:, 1:4]
    original = margin[:, 0:1]
    direct = original - wrong
    corrected = a - a[:, 4:5, :]
    cm = corrected[:, :, :2] - corrected[:, :, 2, None]
    assert np.max(np.abs((cm[:, 0:1]-cm[:, 1:4])-direct)) < 1e-10

    def comparisons(mask):
        selected = direct[mask]
        # For subsets used below, retain all three different mismatches.
        return dict(n=int(selected.size), positive=int((selected>0).sum()),
                    negative=int((selected<0).sum()),tie=int((selected==0).sum()),
                    median=float(np.median(selected)),mean=float(selected.mean()))

    result["true_minus_wrong_contest_margins_original_larger"] = comparisons(np.ones_like(direct,dtype=bool))
    result["original_failed_only_true_minus_wrong_contest_margins"] = comparisons(np.broadcast_to(original<=0,direct.shape))
    original_pair = (margin[:, 0] > 0)
    # Compare two *raw* arms. Swapped-picture predicates refer to the original caption order.
    transition = []
    for i in range(3):
        matched = original_pair
        mismatched = wrong[:,i] > 0
        transition.append(dict(matched_only=int((matched & ~mismatched).sum()),
                               mismatched_only=int((~matched & mismatched).sum()),
                               passed_both=int((matched & mismatched).sum()),
                               passed_neither=int((~matched & ~mismatched).sum())))
    result["pair_outcome_by_swap"] = transition
    same_order = (margin[:,0]<=0)[:,None,:] & (wrong<=0)
    result["persistent_majority_original_minus_wrong"] = comparisons(same_order & np.broadcast_to((same_order.sum(axis=1)>=2)[:,None,:],direct.shape))
    result["direct_swap_by_model"] = {}
    for model in ("qwen","siglip","clip"):
        path = SOURCE / f"full_{model}.jsonl"
        meta = json.loads((SOURCE / f"full_{model}.run.json").read_text(encoding="utf-8"))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["predictions_sha256"]
        mrows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert [r["id"] for r in mrows] == [r["id"] for r in rows]
        scores = np.asarray([[r["scores"][a] for a in ARMS[:4]] for r in mrows])
        margins = scores[:,:,:2] - scores[:,:,2,None]
        contrast = margins[:,0:1]-margins[:,1:]
        majority = (contrast>0).sum(axis=1) >= 2
        result["direct_swap_by_model"][model] = dict(
            image_control_contests=int(contrast.size),
            actual_image_larger=int((contrast>0).sum()),
            actual_image_smaller=int((contrast<0).sum()),
            ties=int((contrast==0).sum()),
            contests_above_all_3=int((contrast>0).all(axis=1).sum()),
            contests_above_2of3=int(majority.sum()),
            images_both_above_2of3=int(majority.all(axis=1).sum()))
    out = ROOT/"audit_grounding.json"
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
