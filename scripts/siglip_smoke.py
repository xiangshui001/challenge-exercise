#!/usr/bin/env python3
"""Short SigLIP-base interface check, with an explicit GPU choice and fresh output."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def select_gpu(index):
    if index < 0:
        raise ValueError("GPU index must be nonnegative")
    columns = ["index", "uuid", "name", "pci.bus_id", "driver_version",
               "memory.total", "memory.used", "utilization.gpu"]
    raw = subprocess.run(["nvidia-smi", "-i", str(index),
                          "--query-gpu=" + ",".join(columns),
                          "--format=csv,noheader,nounits"], capture_output=True,
                         text=True, check=True, timeout=10).stdout
    rows = [[item.strip() for item in row] for row in csv.reader(io.StringIO(raw))]
    if len(rows) != 1 or len(rows[0]) != len(columns) or int(rows[0][0]) != index:
        raise ValueError("nvidia-smi did not uniquely identify the requested GPU")
    gpu = dict(zip(columns, rows[0]))
    if not gpu["uuid"].startswith("GPU-"):
        raise ValueError("Expected a full physical GPU UUID")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    print(f"Selected GPU {index}: {gpu['name']} ({gpu['uuid']}); logical cuda:0", flush=True)
    return gpu


def checked_similarities(image_features, text_features):
    """Validate the result independently on CPU, without importing torch."""
    import numpy as np
    arrays = [np.asarray(array, dtype=np.float32) for array in (image_features, text_features)]
    if (any(array.ndim != 2 or not array.size for array in arrays)
            or arrays[0].shape[0] != 1 or arrays[1].shape[0] != 2
            or arrays[0].shape[1] != arrays[1].shape[1]):
        raise ValueError("Expected one image and two captions with the same feature dimension")
    for array in arrays:
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite features")
        if not np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-5):
            raise ValueError("Features must be L2 normalized")
    return (arrays[0] @ arrays[1].T)[0].tolist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path,
                        help="Local RGB image; if absent, create a synthetic red square for interface checks only")
    parser.add_argument("--text", nargs=2, metavar=("CAPTION_1", "CAPTION_2"),
                        help="Two English captions; required when --image is given")
    parser.add_argument("--gpu-index", type=int, default=2,
                        help="Physical nvidia-smi index for this run; default 2, freely selectable")
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--revision", help="Immutable model SHA for offline repeats; with --allow-download a branch is resolved once")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--out", type=Path, required=True, help="New local output directory, never overwritten")
    args = parser.parse_args(argv)
    if args.image is not None and not args.image.is_file():
        parser.error("--image must name an existing local file")
    if args.image is not None and args.text is None:
        parser.error("Supply two --text captions for your local image")
    if args.text is not None and any(not text.strip() for text in args.text):
        parser.error("Captions cannot be empty")
    if args.out.exists():
        parser.error("--out already exists; choose a new directory")
    if args.gpu_index < 0 or (args.device == "cpu" and args.precision != "float32"):
        parser.error("Use a nonnegative GPU index; CPU mode requires float32")
    # Reject an unpinned offline invocation before touching the GPU or output.
    sys.path.insert(0, str(ROOT))
    from models.siglip_model import MODEL_ID, resolve_revision
    if not args.allow_download:
        try:
            resolve_revision(args.revision)
        except ValueError as exc:
            parser.error(str(exc))
    args.out.mkdir(parents=True, exist_ok=False)
    from exp5.io import environment, file_hash, write_json
    try:
        from PIL import Image
        synthetic = args.image is None
        image_path = args.image or args.out / "synthetic-red.png"
        if synthetic:
            Image.new("RGB", (224, 224), (255, 0, 0)).save(image_path)
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            image.load()
        texts = args.text or ["a red square", "a blue square"]
        gpu = select_gpu(args.gpu_index) if args.device == "cuda:0" else None
        from models.siglip_model import SiglipEncoder
        encoder = SiglipEncoder(args.revision, allow_download=args.allow_download,
                                device=args.device, precision=args.precision)
        torch = encoder.torch
        if gpu:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            image_features, text_features, shapes = encoder.features([image], texts)
            image_array = image_features.cpu().numpy()
            text_array = text_features.cpu().numpy()
        if gpu:
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - start
        similarities = checked_similarities(image_array, text_array)
        parameters = [{"name": name, "shape": list(parameter.shape),
                       "numel": parameter.numel()} for name, parameter in encoder.model.named_parameters()]
        write_json(args.out / "parameter_names.json", parameters)
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                text=True, check=True, timeout=30)
        (args.out / "requirements-lock.txt").write_text(freeze.stdout, encoding="utf-8")
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, timeout=10)
        result = {
            "status": "smoke_passed", "scope": "interface_only_not_retrieval_evaluation",
            "model_id": MODEL_ID, "model_revision": encoder.revision,
            "pretrained_without_finetuning": True, "synthetic_image": synthetic,
            "image_sha256": file_hash(image_path), "texts": texts,
            "cosine_similarities": similarities, "score_is_probability": False,
            "input_shapes": shapes, "image_feature_shape": list(image_array.shape),
            "text_feature_shape": list(text_array.shape), "feature_dtype": str(image_array.dtype),
            "parameter_count": sum(parameter["numel"] for parameter in parameters),
            "parameter_names_file": "parameter_names.json",
            "device": args.device, "precision": args.precision, "gpu_at_selection": gpu,
            "cost": {"model_load_and_download_on_cpu_seconds": encoder.load_seconds,
                     "inference_wall_seconds": inference_seconds,
                     "inference_includes_processor_transfer_and_cpu_copy": True,
                     "peak_allocated_bytes_including_loaded_model": torch.cuda.max_memory_allocated() if gpu else None,
                     "peak_reserved_bytes_including_loaded_model": torch.cuda.max_memory_reserved() if gpu else None},
            "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "source_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in
                               [ROOT / "models/siglip_model.py", Path(__file__).resolve(), ROOT / "pyproject.toml"]},
            "environment": environment(),
        }
        write_json(args.out / "smoke.json", result)
        print(json.dumps({key: result[key] for key in ("status", "scope", "model_revision",
                          "synthetic_image", "input_shapes", "image_feature_shape",
                          "text_feature_shape", "cosine_similarities", "cost")}, ensure_ascii=False, indent=2))
        print(f"Saved {args.out / 'smoke.json'}; process exits and releases its GPU allocation.")
    except Exception as exc:
        write_json(args.out / "failed.json", {"status": "failed", "error_type": type(exc).__name__,
                                            "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
