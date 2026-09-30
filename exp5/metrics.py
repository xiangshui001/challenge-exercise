"""Hit-style recall, binary NDCG and weighted RRF with deterministic ties."""
import math
import numpy as np

KS = (1, 5, 10, 20, 50)
METRICS = [f"hit_recall_at_{k}" for k in KS] + ["ndcg_at_10", "set_recall_at_10"]


def unique_ids(ids, name):
    if not ids or len(ids) != len(set(ids)) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError(f"Empty, duplicate or non-string {name} IDs")


def rank_scores(scores, candidate_ids):
    unique_ids(candidate_ids, "candidate")
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] != len(candidate_ids) or scores.shape[0] == 0:
        raise ValueError("Score matrix shape does not match IDs")
    if not np.issubdtype(scores.dtype, np.number) or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite numbers")
    lexical = np.array(sorted(range(len(candidate_ids)), key=candidate_ids.__getitem__))
    return lexical[np.argsort(-scores[:, lexical], axis=1, kind="stable")]


def evaluate(scores, query_ids, candidate_ids, positives, group_ids):
    scores = np.asarray(scores)
    unique_ids(query_ids, "query")
    if len(query_ids) != len(group_ids) or set(positives) != set(query_ids):
        raise ValueError("Query/group/positive mappings do not match")
    rankings = rank_scores(scores, candidate_ids)
    if rankings.shape[0] != len(query_ids):
        raise ValueError("Score matrix query shape mismatch")
    gallery = set(candidate_ids)
    rows = []
    for idx, qid in enumerate(query_ids):
        relevant = set(positives[qid])
        if not relevant or not relevant.issubset(gallery):
            raise ValueError(f"Empty or missing positive: {qid}")
        ranked_ids = [candidate_ids[j] for j in rankings[idx]]
        positive_ranks = {cid: rank + 1 for rank, cid in enumerate(ranked_ids) if cid in relevant}
        dcg = sum(1 / math.log2(r + 1) for r in positive_ranks.values() if r <= 10)
        idcg = sum(1 / math.log2(r + 1) for r in range(1, min(10, len(relevant)) + 1))
        row = {"query_id": qid, "image_id": group_ids[idx],
               "positive_ranks": positive_ranks, "top_candidates": ranked_ids[:50],
               "top_scores": [float(scores[idx, j]) for j in rankings[idx, :50]]}
        row.update({f"hit_recall_at_{k}": int(min(positive_ranks.values()) <= k) for k in KS})
        row["ndcg_at_10"] = dcg / idcg
        row["set_recall_at_10"] = sum(r <= 10 for r in positive_ranks.values()) / len(relevant)
        rows.append(row)
    return rows


def summarize(rows):
    if not rows:
        raise ValueError("Cannot summarize zero queries")
    return {"query_count": len(rows), **{m: float(np.mean([r[m] for r in rows])) for m in METRICS}}


def weighted_rrf(b0, b1, ids, weight):
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("B1 weight must be in [0,1]")
    if np.shape(b0) != np.shape(b1):
        raise ValueError("RRF input matrices have different shapes")
    ranks = []
    for scores in (b0, b1):
        order = rank_scores(scores, ids)
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(1, len(ids) + 1)[None, :], axis=1)
        ranks.append(rank)
    return (1 - weight) / (60 + ranks[0]) + weight / (60 + ranks[1])
