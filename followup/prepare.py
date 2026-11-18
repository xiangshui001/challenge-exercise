"""Freeze new units from the original annotations before running any models."""
import argparse
import csv
import json
import random
from pathlib import Path

from m0.prepare import sample_strata, sha

SEED = 20261119


def natural_units(scpp, esnli, old):
    used = {im for line in old.read_text().splitlines() for im in json.loads(line)["images"]}
    sugar = {}
    for path in sorted(scpp.glob("*.json")):
        pool = []
        for r in json.loads(path.read_text(encoding="utf8")):
            name = "coco/" + r["filename"]
            if name not in used:
                pool.append(dict(sample_id=f"scpp/{path.stem}/{r['id']}", images=[name],
                                 texts=[r["caption"], r["caption2"], r["negative_caption"]],
                                 benchmark="scpp", category=path.stem, gold=[0, 1]))
        sugar[path.stem] = pool
    assert len(sugar) == 5
    labels = {k: [] for k in ("contradiction", "entailment", "neutral")}
    with esnli.open(encoding="utf8", newline="") as stream:
        for r in csv.DictReader(stream):
            name = "flickr/" + r["Flickr30kID"]
            if name not in used:
                label = r["gold_label"]
                labels[label].append(dict(sample_id="esnli/"+r["pairID"], images=[name],
                                          texts=[r["hypothesis"]], benchmark="esnli",
                                          category=label, gold=label))
    rng = random.Random(SEED)
    return sample_strata(sugar, 100, 0, rng) + sample_strata(labels, 60, 0, rng)


def spatial_units(path):
    scenes = sorted(json.loads(path.read_text())["scenes"], key=lambda s: s["image_index"])
    rng = random.Random(SEED)
    rng.shuffle(scenes)
    rows = []
    for scene in scenes:
        names = [" ".join(o[k] for k in ("size", "color", "material", "shape")) for o in scene["objects"]]
        pairs = [(a, b) for b, left in enumerate(scene["relationships"]["left"]) for a in left
                 if b in scene["relationships"]["right"][a] and names.count(names[a]) == names.count(names[b]) == 1]
        if not pairs:
            continue
        a, b = rng.choice(pairs)
        texts = [f"Is the {names[a]} to the left of the {names[b]}?",
                 f"Is the {names[b]} to the left of the {names[a]}?"]
        gold = ["yes", "no"]
        if rng.randrange(2):
            texts.reverse()
            gold.reverse()
        rows.append(dict(sample_id=f"clevr/{scene['image_index']}", images=["clevr/"+scene["image_filename"]],
                         texts=texts, gold=gold, benchmark="clevr", split="confirmation", category="left"))
        if len(rows) == 100:
            break
    assert len(rows) == 100
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("scpp", "esnli", "old", "scenes", "out"):
        parser.add_argument("--"+name, type=Path)
    args = parser.parse_args()
    rows = spatial_units(args.scenes) if args.scenes else natural_units(args.scpp, args.esnli, args.old)
    with args.out.open("x", encoding="utf8", newline="\n") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False)+"\n")
    print(len(rows), sha(args.out))

