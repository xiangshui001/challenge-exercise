"""Separate native model/processor adapters. Imported only by the encode command."""
from contextlib import ExitStack
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np
from PIL import Image

from .cache import model_identity, save_cache
from .data import load_data, split_rows
from .io import digest, environment, new_directory, read_json, write_json

MODEL_IDS = {"B0": "openai/clip-vit-base-patch16",
             "B1": "google/siglip2-so400m-patch14-384"}


def text_options(method, texts):
    if method == "B1":
        return [t.lower() for t in texts], {"padding": "max_length", "max_length": 64,
                                            "truncation": True}
    if method == "B0":
        return texts, {"padding": True, "max_length": 77, "truncation": True}
    raise ValueError("Only B0/CLIP and B1/SigLIP2 are supported")


def require_gpu(torch, device):
    if device == "cpu":
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly one assigned GPU UUID/index before Python starts")
    if device != "cuda:0":
        raise ValueError("The one visible physical GPU is logical cuda:0")


def encode(data_directory, output, method, revision, split, *, device="cuda:0",
           dtype="bfloat16", image_batch=16, text_batch=64, pilot_images=None,
           allow_download=False, frozen=None):
    if method not in MODEL_IDS or not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("A supported method and immutable 40-character model revision are required")
    if min(image_batch, text_batch) < 1:
        raise ValueError("Batch sizes must be positive")
    images, captions, _, audit = load_data(data_directory, verify_files=True)
    images, captions = split_rows(images, captions, split, pilot_images)
    if split == "test" and (frozen is None or not audit["standard_protocol"]):
        raise ValueError("Test encoding requires a frozen validation config and standard dataset")
    if split == "test":
        from .experiment import load_frozen
        lock = load_frozen(frozen)
        identity = lock["models"][method]
        current = environment()
        if (lock["data_sha256"] != audit["data_sha256"] or identity["revision"] != revision
                or identity["dtype"] != dtype or identity["model_id"] != MODEL_IDS[method]
                or identity["code_sha256"] != current["code_sha256"]
                or identity["packages"] != current["packages"]):
            raise ValueError("Test model/data/environment does not match freeze (checked before loading)")
    # Lazy imports keep the entire CPU evaluation path free of torch/transformers.
    import torch
    from transformers import AutoModel, AutoProcessor, CLIPModel

    require_gpu(torch, device)
    if device == "cpu" and dtype != "float32":
        raise ValueError("Use float32 for CPU model inference")
    if dtype == "bfloat16" and device != "cpu" and not torch.cuda.is_bf16_supported():
        raise ValueError("Selected GPU does not support BF16")
    out = new_directory(output)
    try:
        record = environment()
        # No automatic multi-GPU device_map. No trust_remote_code. SDPA only.
        args = {"revision": revision, "local_files_only": not allow_download}
        processor = AutoProcessor.from_pretrained(MODEL_IDS[method], use_fast=False, **args)
        klass = CLIPModel if method == "B0" else AutoModel
        load_start = time.perf_counter()
        model = klass.from_pretrained(MODEL_IDS[method], torch_dtype=getattr(torch, dtype),
                                      attn_implementation="sdpa", **args).to(device).eval()
        load_seconds = time.perf_counter() - load_start
        resolved = getattr(model.config, "_commit_hash", None)
        if resolved != revision:
            raise ValueError("Loaded model revision does not match the pinned revision")
        processor_info = {"class": type(processor).__name__,
                          "image_processor": processor.image_processor.to_dict(),
                          "tokenizer_class": type(processor.tokenizer).__name__,
                          "text": text_options(method, [""])[1],
                          "lowercase": method == "B1", "image_fast": False}
        meta = {"method": method, "model_id": MODEL_IDS[method], "revision": resolved,
                "dtype": dtype, "processor": processor_info, "packages": record["packages"],
                "code_sha256": record["code_sha256"], "data_sha256": audit["data_sha256"],
                "split": split, "pilot_images": pilot_images,
                "standard_protocol": audit["standard_protocol"] and pilot_images is None,
                "image_ids": [r["image_id"] for r in images],
                "caption_ids": [r["caption_id"] for r in captions], "record": record}
        if split == "test":
            from .experiment import load_frozen
            lock = load_frozen(frozen)
            if lock["data_sha256"] != audit["data_sha256"] or lock["models"][method] != model_identity(meta):
                raise ValueError("Test data/model/processor/environment differs from frozen validation")
            meta["frozen_sha256"] = digest(lock)
        if device != "cpu":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        truncated = 0

        def extract(kind, rows, batch_size):
            nonlocal truncated
            features = []
            for offset in range(0, len(rows), batch_size):
                batch = rows[offset:offset + batch_size]
                if kind == "image":
                    with ExitStack() as stack:
                        opened = [stack.enter_context(Image.open(r["path"])) for r in batch]
                        inputs = processor(images=[im.convert("RGB") for im in opened], return_tensors="pt")
                else:
                    texts, options = text_options(method, [r["text"] for r in batch])
                    lengths = processor.tokenizer(texts, truncation=False, padding=False)["input_ids"]
                    truncated += sum(len(ids) > options["max_length"] for ids in lengths)
                    inputs = processor(text=texts, return_tensors="pt", **options)
                inputs = {k: v.to(device=device, dtype=getattr(torch, dtype)) if v.is_floating_point()
                          else v.to(device) for k, v in inputs.items()}
                with torch.inference_mode():
                    raw = (model.get_image_features(**inputs) if kind == "image"
                           else model.get_text_features(**inputs)).float()
                    if not torch.isfinite(raw).all() or (raw.norm(dim=-1) == 0).any():
                        raise ValueError("Nonfinite/zero model features")
                    f = torch.nn.functional.normalize(raw, dim=-1)
                features.append(f.cpu().numpy())
                print(f"{method} {split} {kind}: {offset + len(batch)}/{len(rows)}", flush=True)
            return np.concatenate(features).astype(np.float32)

        image_features = extract("image", images, image_batch)
        text_features = extract("text", captions, text_batch)
        if device != "cpu":
            torch.cuda.synchronize()
        meta["cost"] = {"mode": "measured", "encoding_wall_seconds": time.perf_counter() - start,
                        "model_load_seconds": load_seconds,
                        "encoding_includes_load_or_download": False,
                        "encoding_includes_processor_and_transfer": True,
                        "cuda_event_seconds": None,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device != "cpu" else None,
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device != "cpu" else None,
                        "image_batch": image_batch, "text_batch": text_batch,
                        "truncated_captions": truncated, "device": device,
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "gpu_name": torch.cuda.get_device_name(0) if device != "cpu" else None}
        freeze = subprocess.run([__import__("sys").executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True, check=True, timeout=30)
        (out / "requirements-lock.txt").write_text(freeze.stdout, encoding="utf-8")
        save_cache(out, image_features, text_features, meta)
        return {"status": "encoded", "method": method, "split": split, "cost": meta["cost"]}
    except Exception as exc:
        write_json(out / "failed.json", {"status": "failed", "error": str(exc)})
        raise
