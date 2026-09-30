"""Explicit subcommands: importing or installing this package starts no jobs."""
import argparse
import json
import sys


def parser():
    p = argparse.ArgumentParser(description="Experiment five baseline CLI (planning YAML is NOT executable)")
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Audit local Karpathy data, no downloads")
    prep.add_argument("--annotations", required=True)
    prep.add_argument("--image-root", required=True)
    prep.add_argument("--source-revision", required=True)
    prep.add_argument("--out", required=True)
    prep.add_argument("--small", action="store_true", help="Nonstandard test fixture only; cannot freeze")
    enc = sub.add_parser("encode", help="Explicit model inference; offline unless --allow-download")
    enc.add_argument("--data", required=True)
    enc.add_argument("--out", required=True)
    enc.add_argument("--method", choices=("B0", "B1"), required=True)
    enc.add_argument("--revision", required=True, help="Immutable Hugging Face 40-character commit SHA")
    enc.add_argument("--split", choices=("val", "test"), required=True)
    enc.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    enc.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    enc.add_argument("--image-batch", type=int, default=16)
    enc.add_argument("--text-batch", type=int, default=64)
    enc.add_argument("--pilot-images", type=int)
    enc.add_argument("--allow-download", action="store_true")
    enc.add_argument("--frozen")
    pilot = sub.add_parser("pilot-evaluate", help="Diagnostic reduced-gallery evaluation, never formal")
    pilot.add_argument("--data", required=True)
    pilot.add_argument("--cache", required=True)
    pilot.add_argument("--out", required=True)
    selection = sub.add_parser("select", help="Tune on val_tune, confirm once, freeze")
    final = sub.add_parser("test", help="Evaluate independent test under frozen config")
    for cmd in (selection, final):
        cmd.add_argument("--data", required=True)
        cmd.add_argument("--b0", required=True)
        cmd.add_argument("--b1", required=True)
        cmd.add_argument("--out", required=True)
    selection.add_argument("--fusion", action="store_true", help="Consider predeclared weighted RRF grid")
    final.add_argument("--frozen", required=True)
    rec = sub.add_parser("recompute", help="Recalculate summary from per-query predictions")
    rec.add_argument("predictions")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    from . import data, experiment
    try:
        if args.command == "prepare":
            result = data.prepare(args.annotations, args.image_root, args.out, args.source_revision, small=args.small)
        elif args.command == "encode":
            from .models import encode
            result = encode(args.data, args.out, args.method, args.revision, args.split,
                            device=args.device, dtype=args.dtype, image_batch=args.image_batch,
                            text_batch=args.text_batch, pilot_images=args.pilot_images,
                            allow_download=args.allow_download, frozen=args.frozen)
        elif args.command == "pilot-evaluate":
            result = experiment.pilot_evaluate(args.data, args.cache, args.out)
        elif args.command == "select":
            result = experiment.select(args.data, args.b0, args.b1, args.out, fusion=args.fusion)
        elif args.command == "test":
            result = experiment.test(args.data, args.b0, args.b1, args.frozen, args.out)
        else:
            result = experiment.recompute(args.predictions)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, KeyError, OSError, RuntimeError, ImportError) as exc:
        print(f"exp5: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
