"""Deterministic serialization, checksums and local provenance."""
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False,
                                            allow_nan=False) + "\n" for r in rows),
                          encoding="utf-8")


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot silently write an empty result table")
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def new_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def code_hash():
    return digest({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def environment():
    packages = {name: importlib.metadata.version(name) for name in
                ("numpy", "Pillow")}
    for name in ("torch", "transformers", "sentencepiece", "protobuf"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total,memory.used,utilization.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        smi = None
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "command_argv": sys.argv, "python": platform.python_version(),
            "platform": platform.platform(), "packages": packages,
            "code_sha256": code_hash(), "nvidia_smi": smi}
