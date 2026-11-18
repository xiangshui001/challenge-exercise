"""Prompt sensitivity and left/right reversal with original frozen checkpoints."""
import argparse
from pathlib import Path
import json
import os
import subprocess
import threading
import time
import hashlib

from m0 import run_dual
from m0.knowledge import prompt, render
from m0.prepare import sha, save
from m0.validate import WEIGHTS

NATURAL = "ad526677c38589de2e4d2afdf0ac200185e669c48a76cd6eb9ba756d8e720096"
SPATIAL = "ba255e8115525800370511d78130a1fdf21227b5f97e12df8ac451517073a2b6"
REFERENCE = "29c5ba57a3da3247a13539f23e7a0fcae3f0bfba9a2f5952b0875d0552369a5e"
LABELS = ("entailment", "contradiction", "neutral")
PROMPTS = {
    "paraphrase": "Decide from the picture alone. Choose entailment if it supports the entire statement, contradiction if it shows evidence against the statement, and neutral if there is too little evidence. Do not assume unseen events, intentions or identities. Dictionary facts alone would not establish what is in the picture. Statement: {h}\nAnswer exactly one label: entailment, contradiction, or neutral.",
    "explicit": "Decide from the picture alone. Use entailment if the whole statement is supported. Use contradiction when a visible detail is incompatible with the statement, such as an incompatible color of the same object. Use neutral when the picture neither establishes nor rules out the statement, such as an unseen destination. Missing evidence alone is not a contradiction. Do not infer hidden intentions, identities or events. Statement: {h}\nAnswer exactly one label: entailment, contradiction, or neutral."
}


def smi(query):
    return subprocess.check_output(["nvidia-smi", query, "--format=csv,noheader,nounits"], text=True).splitlines()


def idle_gpu(index):
    gpus = [x.split(", ") for x in smi("--query-gpu=index,uuid,memory.used,utilization.gpu")]
    gpu = next(g for g in gpus if int(g[0]) == index)
    def processes():
        return [x for x in smi("--query-compute-apps=gpu_uuid,pid") if x.startswith(gpu[1]+",")]
    if processes() or int(gpu[2]) > 64 or int(gpu[3]) != 0:
        raise RuntimeError("Selected GPU is busy")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu[1]
    def watch():
        while True:
            time.sleep(2)
            if len(processes()) > 1:
                print("Foreign compute process appeared; releasing our GPU", flush=True)
                os._exit(75)
    threading.Thread(target=watch, daemon=True).start()
    return gpu[1]


def questions(row):
    if row["benchmark"] == "reference":
        for arm in ("K0", "K1", "K2", "Kshuffle"):
            yield arm, prompt(row["texts"][0], render(row["facts"], arm)), True, row["gold"]
    elif row["benchmark"] == "esnli":
        h = row["texts"][0]
        yield "original", prompt(h, ""), True, row["gold"]
        for arm, template in PROMPTS.items():
            yield arm, template.format(h=h), True, row["gold"]
    else:
        for j, (text, label) in enumerate(zip(row["texts"], row["gold"])):
            for arm, show in (("image", True), ("no_image", False)):
                yield f"{arm}/{j}", text+" Answer exactly yes or no.", show, label


def run_qwen(a, rows):
    gpu = idle_gpu(a.gpu)
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    torch.manual_seed(2026)
    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained(a.model_dir, local_files_only=True, use_fast=False,
                                               min_pixels=256*28*28, max_pixels=262144)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.model_dir, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
    meta = run_dual.metadata(a.model_dir, a.manifest, gpu)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    with (out/"predictions.jsonl").open("x") as log:
        for row in rows[:a.limit]:
            path = Path(a.images)/row["images"][0]
            with Image.open(path) as image:
                rgb = image.convert("RGB")
            digest = sha(path)
            for arm, text, show, gold in questions(row):
                content = ([{"type": "image"}] if show else []) + [{"type":"text", "text":text}]
                chat = processor.apply_chat_template([{"role":"user", "content":content}], tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[chat], images=[rgb] if show else None, return_tensors="pt", padding=True).to("cuda:0")
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=16, do_sample=False)
                raw = processor.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                pred = raw.lower().rstrip(".")
                valid = ("yes", "no") if row["benchmark"] == "clevr" else LABELS
                record = dict(sample_id=row["sample_id"], arm=arm, gold=gold, prediction=pred if pred in valid else "invalid",
                              raw_output=raw, prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                              image_sha256=digest if show else None)
                log.write(json.dumps(record)+"\n")
            log.flush()
    meta.update(seconds=time.monotonic()-start, units=len(rows[:a.limit]), code=sha(__file__))
    save(out/"run.json", meta)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for arg in ("model", "manifest", "model-dir", "images", "out"):
        p.add_argument("--"+arg, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    digest = sha(a.manifest)
    if digest not in {NATURAL, SPATIAL, REFERENCE}:
        raise ValueError("Unregistered input")
    for name, expected in WEIGHTS[a.model].items():
        if sha(Path(a.model_dir)/name) != expected:
            raise ValueError("Unexpected weights")
    if a.model in run_dual.SPECS:
        assert digest == NATURAL
        a.split = "all"
        run_dual.run(a, gpu_selector=idle_gpu)
    else:
        rows = [json.loads(l) for l in Path(a.manifest).read_text().splitlines()]
        run_qwen(a, [r for r in rows if r["benchmark"] in ("esnli","clevr","reference")])

