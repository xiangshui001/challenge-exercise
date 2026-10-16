"""CPU result checks and guards that must run before any model/GPU work."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from models.siglip_model import resolve_revision
from scripts.siglip_smoke import checked_similarities, select_gpu

ROOT = Path(__file__).resolve().parents[1]


class SiglipSmokeTests(unittest.TestCase):
    def test_cosine_values_and_invalid_features(self):
        self.assertEqual(checked_similarities([[1, 0]], [[1, 0], [-1, 0]]), [1.0, -1.0])
        for images, texts in (([[1, 0]], [[1, 0]]), ([[1, 0]], [[1], [-1]]),
                              ([[0, 0]], [[1, 0], [-1, 0]]),
                              ([[1, 0]], [[np.nan, 0], [-1, 0]]),
                              ([[2, 0]], [[1, 0], [-1, 0]])):
            with self.subTest(images=images, texts=texts), self.assertRaises(ValueError):
                checked_similarities(images, texts)

    def test_unpinned_offline_loading_is_rejected(self):
        self.assertEqual(resolve_revision("A" * 40), "a" * 40)
        for revision in (None, "main", "a" * 39):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                resolve_revision(revision)

    def test_selected_index_maps_to_uuid_and_preserves_other_environment(self):
        result = subprocess.CompletedProcess([], 0,
                   "3, GPU-example, NVIDIA A100X, bus, 570.211.01, 81920, 0, 0\n", "")
        with patch("scripts.siglip_smoke.subprocess.run", return_value=result), patch.dict("os.environ"):
            import os
            gpu = select_gpu(3)
            self.assertEqual(gpu["index"], "3")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-example")
        with patch("scripts.siglip_smoke.subprocess.run", return_value=result):
            with self.assertRaises(ValueError):
                select_gpu(2)

    def test_invalid_cli_never_creates_or_overwrites_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "run"
            command = [sys.executable, str(ROOT / "scripts/siglip_smoke.py"), "--out", str(out)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Offline loading requires", result.stderr)
            self.assertFalse(out.exists())
            out.mkdir()
            sentinel = out / "keep.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            result = subprocess.run(command + ["--allow-download"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
