"""Regression checks on the published real evidence, without models or private data."""
import json
from pathlib import Path
import shutil,tempfile,unittest
from m0.validate import validate
from m0.summarize import summarize

ROOT=Path(__file__).resolve().parents[1]/'results/m0-20261102'


class PublicEvidence(unittest.TestCase):
    def test_complete_real_result_identity(self):
        result=validate(ROOT)
        self.assertEqual(result['counts']['qwen'],dict(rows=840,ok=840))

    def test_gold_tampering_fails(self):
        with tempfile.TemporaryDirectory() as d:
            shutil.copytree(ROOT,Path(d)/'result');root=Path(d)/'result'
            path=root/'qwen-predictions.jsonl';rows=path.read_text().splitlines()
            row=json.loads(rows[0]);row['gold']='neutral' if row['gold']!='neutral' else 'entailment'
            rows[0]=json.dumps(row);path.write_text('\n'.join(rows)+'\n')
            with self.assertRaisesRegex(ValueError,'Gold/group mismatch'):validate(root)

    def test_missing_final_arm_fails(self):
        with tempfile.TemporaryDirectory() as d:
            shutil.copytree(ROOT,Path(d)/'result');root=Path(d)/'result'
            path=root/'qwen-predictions.jsonl';path.write_text('\n'.join(path.read_text().splitlines()[:-1])+'\n')
            with self.assertRaisesRegex(ValueError,'Missing or extra'):validate(root)

    def test_smoke_double_count_rejected(self):
        p=ROOT/'siglip-predictions.jsonl'
        with self.assertRaisesRegex(ValueError,'Duplicate predictions'):summarize([p,p])


if __name__=='__main__':unittest.main()
