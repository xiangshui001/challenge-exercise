"""Karpathy JSON ingestion. Fail closed; never drop bad evaluation samples."""
from collections import Counter, defaultdict
from pathlib import Path
import random

from PIL import Image

from .io import (digest, environment, file_hash, new_directory, read_json,
                 read_jsonl, write_json, write_jsonl)


def validate(images, captions, *, standard=True):
    if not images or not captions:
        raise ValueError("Empty image/caption manifest")
    image_ids = [r["image_id"] for r in images]
    caption_ids = [r["caption_id"] for r in captions]
    for name, ids in (("image", image_ids), ("caption", caption_ids)):
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"Empty, non-string or duplicate {name} ID")
    by_id = {r["image_id"]: r for r in images}
    counts = Counter()
    for c in captions:
        if not isinstance(c["text"], str) or not c["text"].strip():
            raise ValueError(f"Empty text: {c['caption_id']}")
        if c["image_id"] not in by_id:
            raise ValueError(f"Missing positive image: {c['caption_id']}")
        if c["split"] != by_id[c["image_id"]]["split"]:
            raise ValueError("Cross-split positive mapping")
        counts[c["image_id"]] += 1
    if any(counts[i] != 5 for i in image_ids):
        raise ValueError("Every image must have exactly five independent captions")
    sizes = Counter(r["split"] for r in images)
    if set(sizes) != {"val", "test"}:
        raise ValueError("Both val and test splits are required")
    if standard and sizes != {"val": 1000, "test": 1000}:
        raise ValueError(f"Standard protocol requires 1000 val + 1000 test images, got {dict(sizes)}")
    hashes = defaultdict(list)
    for r in images:
        hashes[r["sha256"]].append(r)
    duplicates = []
    for rows in hashes.values():
        if len({r["split"] for r in rows}) > 1:
            raise ValueError("Identical image bytes across val/test")
        if len(rows) > 1:
            duplicates.append([r["image_id"] for r in rows])
    repeated = Counter(r["text"] for r in captions)
    return {"images_per_split": dict(sizes), "captions_per_split":
            dict(Counter(r["split"] for r in captions)),
            "duplicate_image_content_groups": duplicates,
            "repeated_caption_text_groups": sum(n > 1 for n in repeated.values())}


def validation_groups(images):
    ids = sorted(r["image_id"] for r in images if r["split"] == "val")
    if len(ids) < 2 or len(ids) % 2:
        raise ValueError("Validation requires an even image count, at least two")
    random.Random(2026).shuffle(ids)
    half = len(ids) // 2
    return {"seed": 2026, "val_tune": sorted(ids[:half]),
            "val_confirm": sorted(ids[half:]), "full_gallery": True}


def fingerprint(images, captions, groups, source):
    # Relative source filenames plus byte hashes, independent of machine paths.
    clean_images = [{k: v for k, v in r.items() if k != "path"} for r in images]
    return digest({"images": clean_images, "captions": captions,
                   "validation_groups": groups, "source": source})


def prepare(annotation, image_root, output, source_revision, *, small=False):
    if not source_revision or not source_revision.strip():
        raise ValueError("An explicit dataset source/version description is required")
    out = new_directory(output)
    images, captions = [], []
    try:
        source = {"annotation_sha256": file_hash(annotation),
                  "source_revision": source_revision,
                  "id_rule": "str(imgid); str(sentid), fallback image_id:caption_index"}
        raw = read_json(annotation)
        if raw.get("dataset", "flickr30k").lower() != "flickr30k":
            raise ValueError("Expected Flickr30K Karpathy annotation")
        root = Path(image_root).resolve()
        for row in raw["images"]:
            if row["split"] not in {"val", "test"}:
                continue
            iid = str(row["imgid"])
            relative = str(Path(row.get("filepath", "")) / row["filename"])
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Image path escapes image root: {relative}")
            with Image.open(path) as im:
                im.load()  # verify decode, not just the header
            images.append({"image_id": iid, "path": str(path), "filename": relative,
                           "split": row["split"], "sha256": file_hash(path)})
            if "sentids" in row and row["sentids"] != [s["sentid"] for s in row["sentences"]]:
                raise ValueError(f"sentids/sentences order mismatch: {iid}")
            for j, sentence in enumerate(row["sentences"]):
                sid = str(sentence["sentid"]) if "sentid" in sentence else f"{iid}:{j}"
                captions.append({"caption_id": sid, "image_id": iid,
                                 "text": sentence["raw"], "split": row["split"]})
        images.sort(key=lambda r: r["image_id"])
        captions.sort(key=lambda r: r["caption_id"])
        checks = validate(images, captions, standard=not small)
        groups = validation_groups(images)
        audit = {"status": "passed", "standard_protocol": not small,
                 "source": source, "checks": checks,
                 "data_sha256": fingerprint(images, captions, groups, source),
                 "record": environment()}
        write_jsonl(out / "images.jsonl", images)
        write_jsonl(out / "captions.jsonl", captions)
        write_json(out / "validation_groups.json", groups)
        write_json(out / "data_audit.json", audit)
        return audit
    except Exception as exc:
        write_json(out / "data_audit.json", {"status": "failed", "error": str(exc)})
        raise


def load_data(directory, *, verify_files=False):
    directory = Path(directory)
    audit = read_json(directory / "data_audit.json")
    if audit.get("status") != "passed":
        raise ValueError("Data audit did not pass")
    images = read_jsonl(directory / "images.jsonl")
    captions = read_jsonl(directory / "captions.jsonl")
    groups = read_json(directory / "validation_groups.json")
    validate(images, captions, standard=audit["standard_protocol"])
    if groups != validation_groups(images):
        raise ValueError("Validation group mapping was changed")
    if fingerprint(images, captions, groups, audit["source"]) != audit["data_sha256"]:
        raise ValueError("Manifest was changed after audit; rerun prepare")
    if verify_files:
        for row in images:
            if file_hash(row["path"]) != row["sha256"]:
                raise ValueError(f"Image changed after audit: {row['image_id']}")
    return images, captions, groups, audit


def split_rows(images, captions, split, pilot_images=None):
    images = [r for r in images if r["split"] == split]
    if pilot_images is not None:
        if split != "val" or pilot_images < 1 or pilot_images > len(images):
            raise ValueError("Pilot is a fixed prefix of validation images only")
        images = images[:pilot_images]
    ids = {r["image_id"] for r in images}
    return images, [r for r in captions if r["image_id"] in ids]
