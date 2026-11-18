"""Regression checks on the fixed follow-up evidence, including incomplete runs."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import shutil
import unittest

from followup.summarize import summarize
from followup.review import compare


ROOT = Path(__file__).resolve().parents[1]/"results"/"followup-20261119"


class FollowupResults(unittest.TestCase):
    def test_summary_from_raw_predictions(self):
        expected=json.loads((ROOT/"summary.json").read_text())
        self.assertEqual(summarize(ROOT),expected)

    def test_missing_prediction_rejected(self):
        with TemporaryDirectory() as tmp:
            copy=Path(tmp)/ROOT.name
            shutil.copytree(ROOT,copy)
            old=copy.parent/"m0-20261102"
            old.mkdir()
            for name in ("manifest.jsonl","image-audit.json"):
                shutil.copyfile(ROOT.parent/old.name/name,old/name)
            file=copy/"qwen-spatial"/"predictions.jsonl"
            lines=file.read_text().splitlines()
            file.write_text("\n".join(lines[1:])+"\n")
            with self.assertRaises(AssertionError):
                summarize(copy)

    def test_unfilled_review_remains_pending(self):
        with TemporaryDirectory() as tmp:
            export=Path(tmp)/"export.json"
            export.write_text(json.dumps({"answers":{"R01":{"judgments":["2"]},"R03":{"judgments":["T"]}}}))
            result=compare(export,ROOT/"ai-reference.jsonl")
            self.assertEqual(result["counts"],{"agree":1,"pending":31})
