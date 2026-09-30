"""Grouped, paired bootstrap with a fixed gallery; repair AND regression lists."""
import numpy as np
from .metrics import METRICS, summarize


def aligned(baseline, other):
    by_id = {r["query_id"]: r for r in other}
    if len(by_id) != len(other) or len({r["query_id"] for r in baseline}) != len(baseline):
        raise ValueError("Duplicate query IDs in comparison")
    if {r["query_id"] for r in baseline} != set(by_id):
        raise ValueError("Paired comparison requires identical queries")
    ordered = [by_id[r["query_id"]] for r in baseline]
    for a, b in zip(baseline, ordered):
        if a["image_id"] != b["image_id"] or set(a["positive_ranks"]) != set(b["positive_ranks"]):
            raise ValueError("Pair group/positive IDs do not match")
    return ordered


def bootstrap(rows, reference=None, *, repeats=2000, seed=2026):
    if not rows or repeats < 1:
        raise ValueError("Bootstrap needs queries and positive repeats")
    if reference is not None:
        reference = aligned(rows, reference)
    groups = sorted({r["image_id"] for r in rows})
    lookup = {g: j for j, g in enumerate(groups)}
    sums = np.zeros((len(groups), len(METRICS)), dtype=np.float64)
    sizes = np.zeros(len(groups), dtype=np.int64)
    for i, r in enumerate(rows):
        g = lookup[r["image_id"]]
        values = np.array([r[m] for m in METRICS], dtype=np.float64)
        if reference is not None:
            values -= [reference[i][m] for m in METRICS]
        sums[g] += values
        sizes[g] += 1
    rng = np.random.default_rng(seed)
    samples = np.empty((repeats, len(METRICS)))
    for i in range(repeats):
        draw = rng.integers(0, len(groups), len(groups))
        samples[i] = sums[draw].sum(axis=0) / sizes[draw].sum()
    bounds = np.quantile(samples, [0.025, 0.975], axis=0)
    return {"group_by": "image_id", "fixed_gallery": True, "paired": reference is not None,
            "groups": len(groups), "repeats": repeats, "seed": seed,
            "metrics": {m: {"estimate": float(sums[:, j].sum() / sizes.sum()),
                            "low": float(bounds[0, j]), "high": float(bounds[1, j])}
                        for j, m in enumerate(METRICS)}}


def changes(baseline, other):
    other = aligned(baseline, other)
    cases = []
    for a, b in zip(baseline, other):
        old, new = a["hit_recall_at_10"], b["hit_recall_at_10"]
        if old != new:
            cases.append({"query_id": a["query_id"], "image_id": a["image_id"],
                          "change": "repair" if new > old else "regression",
                          "baseline_positive_ranks": a["positive_ranks"],
                          "other_positive_ranks": b["positive_ranks"],
                          "baseline_top_candidates": a["top_candidates"],
                          "other_top_candidates": b["top_candidates"]})
    repairs = sum(c["change"] == "repair" for c in cases)
    regressions = sum(c["change"] == "regression" for c in cases)
    return {"repairs": repairs, "regressions": regressions,
            "net_recall_at_10": (repairs - regressions) / len(baseline),
            "query_count": len(baseline)}, cases
