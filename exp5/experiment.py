"""Validation-only selection, frozen provenance, independent test and exports."""
from pathlib import Path
import numpy as np

from .cache import load_cache, model_identity
from .data import load_data
from .io import (digest, environment, file_hash, new_directory, read_json,
                 read_jsonl, write_csv, write_json, write_jsonl)
from .metrics import evaluate, summarize, weighted_rrf
from .stats import bootstrap, changes

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
PROTOCOL = {"schema_version": 1, "metric": "hit_style_recall", "ks": [1, 5, 10, 20, 50],
            "ndcg_k": 10, "tie_break": "candidate_id_lexical_stable",
            "bootstrap_repeats": 2000, "bootstrap_seed": 2026,
            "target_t2i_recall_at_10": 0.85, "rrf_offset": 60}


def load_frozen(path):
    wrapper = read_json(path)
    lock = wrapper["config"]
    if digest(lock) != wrapper["sha256"] or lock["protocol"] != PROTOCOL:
        raise ValueError("Frozen configuration checksum/protocol mismatch")
    if lock["selected_method"] not in {"B0", "B1", "F1"}:
        raise ValueError("Unknown frozen method")
    if lock["selected_method"] == "F1" and lock["selected_weight"] not in WEIGHTS:
        raise ValueError("Invalid frozen fusion weight")
    return lock


def predictions(cache, direction, scores=None, group_filter=None):
    image_features, text_features, _, images, captions = cache
    image_ids = [r["image_id"] for r in images]
    caption_ids = [r["caption_id"] for r in captions]
    if scores is None:
        scores = text_features @ image_features.T
        if direction == "i2t":
            scores = scores.T
    if direction == "t2i":
        queries, candidates = caption_ids, image_ids
        positives = {r["caption_id"]: {r["image_id"]} for r in captions}
        groups = [r["image_id"] for r in captions]
    elif direction == "i2t":
        queries, candidates, groups = image_ids, caption_ids, image_ids
        positives = {iid: set() for iid in image_ids}
        for row in captions:
            positives[row["image_id"]].add(row["caption_id"])
    else:
        raise ValueError("Unknown direction")
    if group_filter is not None:
        selection = [i for i, g in enumerate(groups) if g in set(group_filter)]
        queries = [queries[i] for i in selection]
        groups = [groups[i] for i in selection]
        positives = {q: positives[q] for q in queries}
        scores = scores[selection]
    return evaluate(scores, queries, candidates, positives, groups)


def load_pair(data, b0, b1, split, *, standard=True):
    caches = [load_cache(p, data) for p in (b0, b1)]
    for method, cache in zip(("B0", "B1"), caches):
        meta = cache[2]
        if meta["method"] != method or meta["split"] != split:
            raise ValueError("Wrong model or split for cache")
        if standard and not meta["standard_protocol"]:
            raise ValueError("Pilot/small caches cannot be used for selection or final test")
    if caches[0][2]["image_ids"] != caches[1][2]["image_ids"] or caches[0][2]["caption_ids"] != caches[1][2]["caption_ids"]:
        raise ValueError("Model caches have different query/gallery ID orders")
    if caches[0][2]["code_sha256"] != environment()["code_sha256"] or caches[1][2]["code_sha256"] != environment()["code_sha256"]:
        raise ValueError("Code changed since encoding; encode again or use the original code")
    return caches


def methods(caches, weight=None):
    b0 = caches[0][1] @ caches[0][0].T
    b1 = caches[1][1] @ caches[1][0].T
    scores = {"B0": {"t2i": b0, "i2t": b0.T}, "B1": {"t2i": b1, "i2t": b1.T}}
    if weight is not None:
        scores["F1"] = {"t2i": weighted_rrf(b0, b1, caches[0][2]["image_ids"], weight),
                        "i2t": weighted_rrf(b0.T, b1.T, caches[0][2]["caption_ids"], weight)}
    return scores


def export_results(out, caches, score_methods, *, subgroup=None):
    table, result = [], {}
    for method, directions in score_methods.items():
        result[method] = {}
        for direction, scores in directions.items():
            rows = predictions(caches[0], direction, scores, subgroup)
            result[method][direction] = rows
            name = f"{method}.{direction}"
            write_jsonl(out / f"{name}.queries.jsonl", rows)
            table.append({"method": method, "direction": direction, **summarize(rows)})
            write_json(out / f"{name}.bootstrap.json", bootstrap(rows))
    write_csv(out / "metrics.csv", table)
    comparison = {}
    for method in result:
        if method == "B0":
            continue
        for direction in ("t2i", "i2t"):
            name = f"{method}-B0.{direction}"
            counts, cases = changes(result["B0"][direction], result[method][direction])
            comparison[name] = {**counts, "paired_bootstrap":
                                bootstrap(result[method][direction], result["B0"][direction])}
            write_jsonl(out / f"{name}.changes.jsonl", cases)
    if "F1" in result:
        for direction in ("t2i", "i2t"):
            name = f"F1-B1.{direction}"
            counts, cases = changes(result["B1"][direction], result["F1"][direction])
            comparison[name] = {**counts, "paired_bootstrap":
                                bootstrap(result["F1"][direction], result["B1"][direction])}
            write_jsonl(out / f"{name}.changes.jsonl", cases)
    write_json(out / "comparisons.json", comparison)
    return table


def select(data, b0, b1, output, *, fusion=False):
    out = new_directory(output)
    caches = load_pair(data, b0, b1, "val")
    _, _, groups, audit = load_data(data)
    base = methods(caches)
    candidates = []
    for method in ("B0", "B1"):
        tune = summarize(predictions(caches[0], "t2i", base[method]["t2i"], groups["val_tune"]))
        candidates.append({"method": method, "weight": None, **tune})
    if fusion:
        for weight in WEIGHTS:
            fused = weighted_rrf(base["B0"]["t2i"], base["B1"]["t2i"], caches[0][2]["image_ids"], weight)
            tune = summarize(predictions(caches[0], "t2i", fused, groups["val_tune"]))
            candidates.append({"method": "F1", "weight": weight, **tune})
    # Same quality => prefer a single model; deterministic final tie chooses B1.
    winner = max(candidates, key=lambda r: (r["hit_recall_at_10"], r["ndcg_at_10"],
                                            r["method"] != "F1", r["method"] == "B1"))
    weight = winner["weight"] if winner["method"] == "F1" else None
    scores = methods(caches, weight)
    confirm = {direction: summarize(predictions(caches[0], direction, scores[winner["method"]][direction],
                                                groups["val_confirm"])) for direction in ("t2i", "i2t")}
    # Confirmation is reported once. No retuning/fallback based on its score.
    record = environment()
    lock = {"protocol": PROTOCOL, "data_sha256": audit["data_sha256"],
            "models": {m: model_identity(c[2]) for m, c in zip(("B0", "B1"), caches)},
            "validation_cache_sha256": {m: file_hash(Path(p) / "metadata.json")
                                        for m, p in zip(("B0", "B1"), (b0, b1))},
            "selected_method": winner["method"], "selected_weight": weight,
            "fusion_considered": fusion, "selection": candidates, "confirmation": confirm,
            "record": record}
    write_json(out / "frozen.json", {"config": lock, "sha256": digest(lock)})
    write_csv(out / "selection.csv", candidates)
    # Complete validation results are diagnostic; selection above used tune only.
    table = export_results(out, caches, scores)
    write_json(out / "run_record.json", {"stage": "validation_and_freeze", "record": record,
               "standard_protocol": True, "frozen_sha256": digest(lock), "metrics": table})
    return {"selected_method": winner["method"], "selected_weight": weight,
            "confirmation": confirm, "frozen": str(out / "frozen.json")}


def test(data, b0, b1, frozen, output):
    lock = load_frozen(frozen)
    caches = load_pair(data, b0, b1, "test")
    for method, cache in zip(("B0", "B1"), caches):
        meta = cache[2]
        if model_identity(meta) != lock["models"][method] or meta.get("frozen_sha256") != digest(lock):
            raise ValueError("Test cache was not encoded under this frozen configuration")
        if meta["data_sha256"] != lock["data_sha256"]:
            raise ValueError("Test dataset differs from freeze")
    out = new_directory(output)
    scores = methods(caches, lock["selected_weight"])
    table = export_results(out, caches, scores)
    primary = next(r for r in table if r["method"] == lock["selected_method"] and r["direction"] == "t2i")
    record = {"stage": "independent_test", "record": environment(), "standard_protocol": True,
              "frozen_sha256": digest(lock), "selected_method": lock["selected_method"],
              "selected_t2i_recall_at_10": primary["hit_recall_at_10"],
              "target_met": primary["hit_recall_at_10"] >= 0.85,
              "cache_metadata_sha256": {m: file_hash(Path(p) / "metadata.json")
                                        for m, p in zip(("B0", "B1"), (b0, b1))}, "metrics": table}
    write_json(out / "run_record.json", record)
    write_json(out / "frozen.json", {"config": lock, "sha256": digest(lock)})
    return {k: record[k] for k in ("selected_method", "selected_t2i_recall_at_10", "target_met")}


def pilot_evaluate(data, cache_path, output):
    cache = load_cache(cache_path, data)
    if cache[2]["split"] != "val":
        raise ValueError("Pilot evaluation only accepts validation")
    out = new_directory(output)
    table = []
    for direction in ("t2i", "i2t"):
        rows = predictions(cache, direction)
        write_jsonl(out / f"{cache[2]['method']}.{direction}.queries.jsonl", rows)
        table.append({"method": cache[2]["method"], "direction": direction, **summarize(rows)})
    write_csv(out / "metrics.csv", table)
    write_json(out / "run_record.json", {"stage": "pilot_only", "standard_protocol": False,
               "warning": "Reduced gallery: never report these values as standard Flickr30K results",
               "cache_metadata_sha256": file_hash(Path(cache_path) / "metadata.json"),
               "record": environment(), "metrics": table})
    return {"stage": "pilot_only", "metrics": table}


def recompute(prediction_file):
    """Check per-query sufficient statistics; no model reload required."""
    rows = read_jsonl(prediction_file)
    for row in rows:
        ranks = list(row["positive_ranks"].values())
        if not ranks or len(set(ranks)) != len(ranks) or any(type(r) is not int or r < 1 for r in ranks):
            raise ValueError("Invalid positive ranks")
        for k in PROTOCOL["ks"]:
            if row[f"hit_recall_at_{k}"] != int(min(ranks) <= k):
                raise ValueError("Stored hit disagrees with positive ranks")
        import math
        ndcg = sum(1 / math.log2(r + 1) for r in ranks if r <= 10) / sum(
            1 / math.log2(r + 1) for r in range(1, min(10, len(ranks)) + 1))
        recall = sum(r <= 10 for r in ranks) / len(ranks)
        if not np.isclose(row["ndcg_at_10"], ndcg) or not np.isclose(row["set_recall_at_10"], recall):
            raise ValueError("Stored NDCG/set recall disagrees with positive ranks")
    if len({r["query_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate prediction query IDs")
    return summarize(rows)
