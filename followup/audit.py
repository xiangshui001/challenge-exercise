"""Validate both frozen manifests and all images before inference."""
import json
from pathlib import Path
from PIL import Image
from m0.prepare import save, sha
from followup.run import NATURAL, SPATIAL

def audit():
    old = json.loads(Path("results/m0-20261102/image-audit.json").read_text())["files"]
    hashes = {v["sha256"]:k for k,v in old.items()}
    registered = []
    duplicates = []
    for name, expected in [("natural", NATURAL), ("spatial", SPATIAL)]:
        path = Path(name+".jsonl")
        assert sha(path) == expected
        for line in path.read_text().splitlines():
            r = json.loads(line)
            file = Path("images")/r["images"][0]
            digest = sha(file)
            if digest in hashes:
                duplicates.append([r["sample_id"],hashes[digest]])
            hashes[digest] = r["sample_id"]
            with Image.open(file) as im:
                im.load()
            registered.append({k:v for k,v in r.items() if k!="texts"} | {"image_sha256":digest,
                "text_sha256":[__import__("hashlib").sha256(t.encode()).hexdigest() for t in r["texts"]]})
    save("audit.json", {"duplicates":duplicates, "registered":registered})
    if duplicates:
        raise ValueError("Duplicate image bytes require preregistered exclusion")
    save("freeze.json", {"manifest_sha256":NATURAL})

if __name__ == "__main__":
    audit()

