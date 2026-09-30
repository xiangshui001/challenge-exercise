#!/usr/bin/env python3
"""Resolve an assigned nvidia-smi index to UUID before starting Python/torch."""
import argparse
import csv
import io
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Launch one exp5 encode on a single assigned GPU")
    parser.add_argument("--index", type=int, default=2, help="Physical index from nvidia-smi, default 2")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments after -- passed to exp5 encode")
    args = parser.parse_args()
    extra = args.args[1:] if args.args[:1] == ["--"] else args.args
    if not extra:
        parser.error("Supply encode arguments after --")
    # This helper cannot launch arbitrary shell commands and never changes running jobs.
    try:
        raw = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,pci.bus_id,memory.total,memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             check=True, timeout=10).stdout
        rows = [[v.strip() for v in r] for r in csv.reader(io.StringIO(raw))]
        matches = [r for r in rows if int(r[0]) == args.index]
        if len(matches) != 1:
            raise ValueError(f"GPU index {args.index} was not uniquely found")
        gpu = matches[0]
        print(f"Selected physical GPU {gpu[0]}: {gpu[2]} UUID={gpu[1]} bus={gpu[3]} "
              f"memory={gpu[5]}/{gpu[4]} MiB utilization={gpu[6]}%", flush=True)
        print("This GPU becomes logical cuda:0. Existing allocation is preserved.", flush=True)
        child_env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=gpu[1])
        result = subprocess.run([sys.executable, "-m", "exp5", "encode", *extra], env=child_env)
        raise SystemExit(result.returncode)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"GPU selection failed: {exc}\n")


if __name__ == "__main__":
    main()
