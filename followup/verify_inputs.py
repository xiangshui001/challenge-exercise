"""Check the released arm identities against rebuilt, private manifests."""
import argparse
import hashlib
import json
from pathlib import Path
from followup.run import NATURAL, SPATIAL, REFERENCE, questions
from m0.prepare import sha


def verify(manifests, results):
    counts={}
    for name, digest in (("natural", NATURAL), ("spatial", SPATIAL), ("reference", REFERENCE)):
        path=manifests/(name+".jsonl")
        assert sha(path)==digest
        expected={}
        for line in path.read_text(encoding="utf8").splitlines():
            row=json.loads(line)
            if row["benchmark"]=="scpp":
                continue
            for arm,text,show,gold in questions(row):
                expected[(row["sample_id"],arm)]=(hashlib.sha256(text.encode()).hexdigest(),gold,show)
        predictions=(results/("qwen-"+name)/"predictions.jsonl").read_text().splitlines()
        for line in predictions:
            r=json.loads(line)
            triple=expected.pop((r["sample_id"],r["arm"]))
            assert triple==(r["prompt_sha256"],r["gold"],r["image_sha256"] is not None)
        assert not expected
        counts[name]=len(predictions)
    return counts


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("manifests", type=Path)
    p.add_argument("results", type=Path)
    a=p.parse_args()
    print(json.dumps(verify(a.manifests,a.results)))

