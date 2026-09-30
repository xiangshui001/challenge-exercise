from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image

from exp5.cache import load_cache, save_cache
from exp5.data import load_data, prepare, split_rows, validate, validation_groups
from exp5.io import read_json, write_json
from exp5.models import require_gpu, text_options


def fixture(root, count=2):
    image_root = root / "pictures"
    image_root.mkdir()
    images = []
    for split, base in (("val", 0), ("test", count)):
        for i in range(count):
            idx = base + i
            name = f"{idx:04}.png"
            Image.new("RGB", (4, 4), (idx % 256, idx // 256, 10)).save(image_root / name)
            sentences = [{"sentid": idx * 5 + j, "raw": f"An image {idx}, description {j}"} for j in range(5)]
            images.append({"imgid": idx, "filename": name, "split": split, "sentences": sentences,
                           "sentids": [s["sentid"] for s in sentences]})
    annotation = root / "dataset.json"
    write_json(annotation, {"dataset": "flickr30k", "images": images})
    return annotation, image_root


class DataCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.annotation, self.pictures = fixture(self.root)
        self.data = self.root / "manifests"
        prepare(self.annotation, self.pictures, self.data, "synthetic-test-only", small=True)
        self.images, self.captions, self.groups, self.audit = load_data(self.data)

    def tearDown(self):
        self.tmp.cleanup()

    def test_audit_and_validation_groups(self):
        self.assertFalse(self.audit["standard_protocol"])
        self.assertEqual(self.audit["checks"]["captions_per_split"], {"val": 10, "test": 10})
        self.assertFalse(set(self.groups["val_tune"]) & set(self.groups["val_confirm"]))
        self.assertEqual(self.groups, validation_groups(self.images))
        with self.assertRaises(ValueError):
            validate(self.images, self.captions, standard=True)

    def test_duplicate_id_cross_split_missing_empty(self):
        with self.assertRaises(ValueError):
            validate(self.images + self.images[:1], self.captions, standard=False)
        for key, value in (("image_id", "absent"), ("text", " "), ("split", "test")):
            captions = deepcopy(self.captions)
            captions[0][key] = value
            with self.assertRaises(ValueError):
                validate(self.images, captions, standard=False)
        images = deepcopy(self.images)
        images[-1]["sha256"] = images[0]["sha256"]
        with self.assertRaises(ValueError):
            validate(images, self.captions, standard=False)

    def test_corrupt_image_fails_no_silent_drop(self):
        (self.pictures / "0000.png").write_bytes(b"corrupt")
        out = self.root / "failed"
        with self.assertRaises(OSError):
            prepare(self.annotation, self.pictures, out, "synthetic", small=True)
        self.assertEqual(read_json(out / "data_audit.json")["status"], "failed")
        self.assertFalse((out / "images.jsonl").exists())

    def test_image_modified_since_audit(self):
        Image.new("RGB", (4, 4), (99, 99, 99)).save(self.pictures / "0000.png")
        with self.assertRaises(ValueError):
            load_data(self.data, verify_files=True)

    def test_manifest_modified(self):
        groups = deepcopy(self.groups)
        groups["val_tune"], groups["val_confirm"] = groups["val_confirm"], groups["val_tune"]
        write_json(self.data / "validation_groups.json", groups)
        with self.assertRaises(ValueError):
            load_data(self.data)

    def test_mapping_and_relative_path_attack(self):
        raw = read_json(self.annotation)
        raw["images"][0]["sentids"].reverse()
        write_json(self.annotation, raw)
        with self.assertRaises(ValueError):
            prepare(self.annotation, self.pictures, self.root / "bad-order", "test", small=True)
        raw["images"][0]["filename"] = "../escape.png"
        write_json(self.annotation, raw)
        with self.assertRaises(ValueError):
            prepare(self.annotation, self.pictures, self.root / "bad-path", "test", small=True)

    def make_cache(self):
        images, captions = split_rows(self.images, self.captions, "val")
        meta = {"image_ids": [r["image_id"] for r in images],
                "caption_ids": [r["caption_id"] for r in captions],
                "data_sha256": self.audit["data_sha256"], "split": "val",
                "pilot_images": None, "standard_protocol": False}
        image = np.tile(np.array([[1, 0]], dtype=np.float32), (len(images), 1))
        text = np.tile(np.array([[1, 0]], dtype=np.float32), (len(captions), 1))
        out = self.root / "cache"; out.mkdir()
        save_cache(out, image, text, meta)
        return out

    def test_cache_roundtrip_and_tamper(self):
        out = self.make_cache()
        image, text, meta, _, _ = load_cache(out, self.data)
        self.assertEqual(image.shape, (2, 2))
        self.assertEqual(text.shape, (10, 2))
        image[0] = [0, 1]
        np.save(out / "images.npy", image)
        with self.assertRaises(ValueError):
            load_cache(out, self.data)

    def test_cache_order_mismatch(self):
        out = self.make_cache()
        meta = read_json(out / "metadata.json")
        meta["image_ids"].reverse()
        write_json(out / "metadata.json", meta)
        with self.assertRaises(ValueError):
            load_cache(out, self.data)

    def test_separate_text_processors(self):
        clip, co = text_options("B0", ["A DOG"])
        siglip, so = text_options("B1", ["A DOG"])
        self.assertEqual(clip, ["A DOG"])
        self.assertEqual(siglip, ["a dog"])
        self.assertEqual(co["max_length"], 77)
        self.assertEqual(so["max_length"], 64)
        self.assertEqual(so["padding"], "max_length")

    def test_gpu_requires_one_explicit_device(self):
        from unittest.mock import patch
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""}):
            with self.assertRaises(ValueError):
                require_gpu(None, "cuda:0")
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1"}):
            with self.assertRaises(ValueError):
                require_gpu(None, "cuda:0")


if __name__ == "__main__":
    unittest.main()
