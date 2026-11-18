"""Rebuild private, shuffled review materials and the fixed sense-selection input."""
import argparse
import json
from pathlib import Path
import shutil

from followup.review import build


def prepare(old, images, knowledge, evidence, out):
    original={r["sample_id"]:r for r in map(json.loads, old.read_text(encoding="utf8").splitlines())}
    choices=json.loads((evidence/"selected-senses.json").read_text())
    facts=json.loads(knowledge.read_text(encoding="utf8"))
    review=out/"review"
    review.mkdir(parents=True,exist_ok=True)
    queue=[]
    with (out/"reference.jsonl").open("x",encoding="utf8",newline="\n") as stream:
        for line in (evidence/"ai-reference.jsonl").read_text(encoding="utf8").splitlines():
            fixed=json.loads(line)
            code=fixed["id"]
            key=fixed["sample_id"]
            r=original[key]
            names=[]
            for j,pos in enumerate(fixed["image_order"]):
                src=images/r["images"][pos]
                dest=review/f"{code}-{j}{src.suffix}"
                shutil.copyfile(src,dest)
                names.append(dest.name)
            q=dict(id=code, task=r["benchmark"],texts=[r["texts"][j] for j in fixed["text_order"]],
                   images=names,human_judgment=None,human_evidence=None)
            if code in choices:
                q["candidate_senses"]=facts[key]
                q["human_selected_senses"]=None
                selected=[f for f in facts[key] if f["sense"] in choices[code]]
                assert len(selected)==len(choices[code])
                stream.write(json.dumps(r|dict(benchmark="reference",facts=selected),ensure_ascii=False)+"\n")
            queue.append(q)
    path=review/"human-queue.json"
    path.write_text(json.dumps(queue,ensure_ascii=False,indent=2),encoding="utf8")
    build(path)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("old","images","knowledge","evidence","out"):
        p.add_argument("--"+name, type=Path,required=True)
    prepare(**vars(p.parse_args()))

