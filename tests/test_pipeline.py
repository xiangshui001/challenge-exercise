"""End-to-end CPU regression using generated data/features, NEVER real model results."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from exp5.cache import model_identity, save_cache
from exp5.data import load_data, prepare, split_rows
from exp5.experiment import (load_frozen, methods, pilot_evaluate, predictions,
                             recompute, select, test as final_test)
from exp5.io import digest, environment, read_json, write_json
from exp5.metrics import summarize
from exp5.models import encode
from test_data_cache import fixture


def synthetic_cache(root, data, method, split, *, frozen=None, pilot_images=None):
    images, captions, _, audit = load_data(data)
    images, captions = split_rows(images, captions, split, pilot_images)
    # Circle embeddings create a unique nearest image for each caption.
    angles = np.linspace(0, 2 * np.pi, len(images), endpoint=False)
    vectors = np.stack((np.cos(angles), np.sin(angles)), axis=1).astype(np.float32)
    features = {r["image_id"]: v for r, v in zip(images, vectors)}
    text = np.stack([features[r["image_id"]] for r in captions])
    # B0 deliberately rotates the text association to make B1's repair measurable.
    if method == "B0":
        text = -text
    record = environment()
    meta = {"method": method, "model_id": f"synthetic-{method}", "revision": "f" * 40,
            "dtype": "float32", "processor": {"synthetic": True}, "packages": record["packages"],
            "code_sha256": record["code_sha256"], "data_sha256": audit["data_sha256"],
            "split": split, "pilot_images": pilot_images,
            "standard_protocol": audit["standard_protocol"] and pilot_images is None,
            "image_ids": [r["image_id"] for r in images],
            "caption_ids": [r["caption_id"] for r in captions], "record": record}
    if frozen:
        meta["frozen_sha256"] = digest(load_frozen(frozen))
    out = root / f"{method}-{split}"; out.mkdir()
    save_cache(out, vectors, text, meta)
    return out


class PipelineTests(unittest.TestCase):
    def test_full_synthetic_validation_freeze_test_recompute(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotation, pictures = fixture(root, count=1000)
            data = root / "manifests"
            prepare(annotation, pictures, data, "SYNTHETIC generated test fixture, not Flickr30K")
            b0 = synthetic_cache(root, data, "B0", "val")
            b1 = synthetic_cache(root, data, "B1", "val")
            validation = root / "validation"
            result = select(data, b0, b1, validation, fusion=True)
            self.assertEqual(result["selected_method"], "B1")
            frozen = validation / "frozen.json"
            lock = load_frozen(frozen)
            self.assertEqual(lock["confirmation"]["t2i"]["query_count"], 2500)
            self.assertEqual(lock["confirmation"]["i2t"]["query_count"], 500)
            self.assertEqual(len(lock["selection"]), 7)
            with self.assertRaises(FileExistsError):
                select(data, b0, b1, validation)
            t0 = synthetic_cache(root, data, "B0", "test", frozen=frozen)
            t1 = synthetic_cache(root, data, "B1", "test", frozen=frozen)
            out = root / "final"
            final = final_test(data, t0, t1, frozen, out)
            self.assertTrue(final["target_met"])
            self.assertAlmostEqual(final["selected_t2i_recall_at_10"], 1)
            for method in ("B0", "B1"):
                for direction in ("t2i", "i2t"):
                    computed = recompute(out / f"{method}.{direction}.queries.jsonl")
                    expected = next(r for r in read_json(out / "run_record.json")["metrics"]
                                    if r["method"] == method and r["direction"] == direction)
                    self.assertEqual(computed["query_count"], 5000 if direction == "t2i" else 1000)
                    self.assertEqual(computed["ndcg_at_10"], expected["ndcg_at_10"])
            comparisons = read_json(out / "comparisons.json")
            self.assertEqual(comparisons["B1-B0.t2i"]["repairs"], 5000)
            self.assertEqual(comparisons["B1-B0.t2i"]["regressions"], 0)
            # Changing the test model's processor must invalidate the run.
            meta = read_json(t1 / "metadata.json")
            meta["processor"]["different"] = True
            write_json(t1 / "metadata.json", meta)
            with self.assertRaises(ValueError):
                final_test(data, t0, t1, frozen, root / "invalid")
            # Changing the frozen file without updating the digest also fails.
            wrapper = read_json(frozen); wrapper["config"]["selected_method"] = "B0"
            write_json(frozen, wrapper)
            with self.assertRaises(ValueError):
                load_frozen(frozen)

    def test_small_pilot_cannot_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotation, pictures = fixture(root)
            data = root / "manifests"
            prepare(annotation, pictures, data, "synthetic", small=True)
            b0 = synthetic_cache(root, data, "B0", "val")
            b1 = synthetic_cache(root, data, "B1", "val")
            result = pilot_evaluate(data, b1, root / "pilot")
            self.assertEqual(result["stage"], "pilot_only")
            self.assertFalse(read_json(root / "pilot/run_record.json")["standard_protocol"])
            with self.assertRaises(ValueError):
                select(data, b0, b1, root / "not-frozen")

    def test_no_implicit_downloads_or_unfrozen_test(self):
        with self.assertRaises(ValueError):
            encode("absent", "absent", "B0", "main", "val")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotation, pictures = fixture(root)
            data = root / "manifests"
            prepare(annotation, pictures, data, "synthetic", small=True)
            # Fails before attempting to import torch, load a model, or create output.
            with self.assertRaises(ValueError):
                encode(data, root / "no-test", "B0", "f" * 40, "test")
            self.assertFalse((root / "no-test").exists())


if __name__ == "__main__":
    unittest.main()
