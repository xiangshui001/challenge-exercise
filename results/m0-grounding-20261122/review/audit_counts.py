"""Reconcile the 42-image audit with the archived 150-image Qwen predictions."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONF = ROOT.parent / "replication" / "confirmation"


def main():
    raw = (CONF / "full_qwen.jsonl").read_bytes()
    record = json.loads((CONF / "full_qwen.run.json").read_bytes())
    if hashlib.sha256(raw).hexdigest() != record["predictions_sha256"]:
        raise ValueError("Qwen prediction seal mismatch")
    predictions = [json.loads(line) for line in raw.splitlines()]
    selected = {p["id"]: p for p in predictions if p["scores"]["original"][2] > 0}
    forms = json.loads((ROOT / "sources.json").read_bytes())["forms"]
    items = {r["id"]: r for r in forms["audit"]["rows"]}
    with (ROOT / "unblinding-42.csv").open(encoding="utf8", newline="") as f:
        rows = list(csv.DictReader(f))
    seen = set()
    counts = defaultdict(Counter)
    for row in rows:
        item = items[row["item"]]
        pred = selected[item["source"]]
        slot = "ABC".index(row["slot"])
        idx = item["permutation"][slot]
        identity = (item["source"], idx)
        scores = pred["scores"]["original"]
        rank_pass = min(scores[:2]) > scores[2]
        if (identity in seen or row["source_id"] != item["source"]
                or row["category"] != pred["cat"] or row["image_sha256"] != pred["sha256"]
                or row["caption_sha256"] != pred["text_sha256"][idx]
                or row["caption_sha256"] != item["display_text_sha256"][slot]
                or float(row["yes_minus_no"]) != scores[idx]
                or row["strict_ranking_pass"] != str(rank_pass)
                or row["benchmark_role"] != ("参考原句", "参考改写", "基准困难句")[idx]
                or row["assistant_visual_review"] not in ("冲突", "证据不足", "支持")):
            raise ValueError(f"Audit mismatch at {identity}")
        seen.add(identity)
        if idx == 2:
            counts[pred["cat"]][row["assistant_visual_review"]] += 1
    if len(predictions) != 150 or len(selected) != 42 or seen != {(k, i) for k in selected for i in range(3)}:
        raise ValueError("Incomplete audit cohort")
    print("category\tconflict\tinsufficient\tsupport")
    for category, values in [*sorted(counts.items()), ("ALL", sum(counts.values(), Counter()))]:
        print(category, *(values[name] for name in ("冲突", "证据不足", "支持")), sep="\t")


if __name__ == "__main__":
    main()
