"""CPU-only checks for the distinct SigLIP-base validation path."""
import unittest
from unittest.mock import patch

from exp5.cli import parser
from exp5.models import MODEL_IDS, text_options
from exp5.experiment import baseline_evaluate


class SiglipBaseBaselineTests(unittest.TestCase):
    def test_distinct_model_identity(self):
        self.assertEqual(MODEL_IDS["S0"], "google/siglip-base-patch16-224")
        self.assertNotEqual(MODEL_IDS["S0"], MODEL_IDS["B1"])

    def test_native_siglip_text_processing(self):
        texts, opts = text_options("S0", ["A Red Square"])
        self.assertEqual(texts, ["a red square"])
        self.assertEqual(opts["max_length"], 64)
        self.assertEqual(opts["padding"], "max_length")

    def test_cli_accepts_s0_and_1014_validation(self):
        a = parser().parse_args([
            "prepare", "--annotations", "a.json", "--image-root", "images",
            "--source-revision", "rev", "--out", "out", "--expected-val-images", "1014"])
        self.assertEqual(a.expected_val_images, 1014)
        e = parser().parse_args([
            "encode", "--data", "data", "--out", "cache", "--method", "S0",
            "--revision", "a" * 40, "--split", "val"])
        self.assertEqual(e.method, "S0")
        b = parser().parse_args([
            "baseline-evaluate", "--data", "data", "--cache", "cache", "--out", "out"])
        self.assertEqual(b.command, "baseline-evaluate")

    @patch("exp5.experiment.load_cache")
    def test_disallows_old_or_pilot_cache(self, load_cache):
        for method, standard in (("B1", True), ("S0", False)):
            with self.subTest(method=method, standard=standard):
                load_cache.return_value = (
                    None, None,
                    {"method": method, "split": "val", "standard_protocol": standard},
                    None, None)
                with self.assertRaisesRegex(ValueError, "S0 baseline requires"):
                    baseline_evaluate("data", "cache", "out")
