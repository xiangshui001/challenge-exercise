"""Validated FP32 caches with explicit ID order, provenance and file digests."""
from pathlib import Path
import numpy as np
from .data import load_data, split_rows
from .io import file_hash, read_json, write_json


def check_features(features, ids):
    if features.dtype != np.float32 or features.ndim != 2 or features.shape[0] != len(ids) or features.shape[1] < 1:
        raise ValueError("Expected an FP32 feature matrix with matching ID count")
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("Empty/duplicate cached IDs")
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite cached features")
    if not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-4):
        raise ValueError("Features must be L2 normalized")


def save_cache(directory, image_features, text_features, metadata):
    directory = Path(directory)
    check_features(image_features, metadata["image_ids"])
    check_features(text_features, metadata["caption_ids"])
    if image_features.shape[1] != text_features.shape[1]:
        raise ValueError("Image/text feature dimensions differ")
    np.save(directory / "images.npy", image_features, allow_pickle=False)
    np.save(directory / "texts.npy", text_features, allow_pickle=False)
    metadata = {**metadata, "feature_sha256": {name: file_hash(directory / name)
                 for name in ("images.npy", "texts.npy")}}
    write_json(directory / "metadata.json", metadata)  # completion marker, written last


def load_cache(directory, data_directory):
    directory = Path(directory)
    meta = read_json(directory / "metadata.json")
    images, captions, _, audit = load_data(data_directory)
    if meta["data_sha256"] != audit["data_sha256"]:
        raise ValueError("Cache belongs to a different audited dataset")
    images, captions = split_rows(images, captions, meta["split"], meta["pilot_images"])
    if meta["image_ids"] != [r["image_id"] for r in images] or meta["caption_ids"] != [r["caption_id"] for r in captions]:
        raise ValueError("Cache ID order does not match manifest")
    if meta["standard_protocol"] != (audit["standard_protocol"] and meta["pilot_images"] is None):
        raise ValueError("Cache protocol flag mismatch")
    arrays = []
    for name, ids in (("images.npy", meta["image_ids"]), ("texts.npy", meta["caption_ids"])):
        if file_hash(directory / name) != meta["feature_sha256"][name]:
            raise ValueError("Feature file checksum changed")
        a = np.load(directory / name, allow_pickle=False)
        check_features(a, ids)
        arrays.append(a)
    if arrays[0].shape[1] != arrays[1].shape[1]:
        raise ValueError("Image/text feature dimensions differ")
    return arrays[0], arrays[1], meta, images, captions


def model_identity(meta):
    return {k: meta[k] for k in ("method", "model_id", "revision", "dtype", "processor",
                                "packages", "code_sha256")}
