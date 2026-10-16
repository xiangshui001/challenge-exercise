#!/usr/bin/env python3
"""Read local Git/layout status; no network, model loading or server login."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

DIRECTORIES = ("data", "models", "train", "evaluation", "configs", "checkpoints",
               "exp5", "scripts", "tests", "docs")
SUFFIXES = {".py", ".md", ".toml", ".json", ".yaml", ".yml", ".bib"}
EXCLUDED = ("data/local/", "models/pretrained/", "checkpoints/", "cache/", "runs/",
            "private_sources/", "external/")


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True, timeout=10).stdout


def public_source(name):
    path = Path(name)
    if name == "checkpoints/README.md":
        return True
    if any(name.startswith(p) for p in EXCLUDED):
        return False
    if path.name.startswith(".env") or path.suffix in {".key", ".pem"}:
        return False
    return path.suffix in SUFFIXES or name == ".gitignore"


def report(root):
    root = Path(root).resolve()
    commit = git(root, "rev-parse", "HEAD").strip()
    branch = git(root, "branch", "--show-current").strip()
    changes = git(root, "status", "--porcelain", "--untracked-files=normal").splitlines()
    names = git(root, "ls-files", "-z").split("\0")
    hashes, missing = {}, []
    for name in sorted(n for n in names if n and public_source(n)):
        path = root / name
        if not path.is_file() or path.is_symlink():
            missing.append(name)
            continue
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    directories = {name: (root / name).is_dir() for name in DIRECTORIES}
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit, "branch": branch or "detached",
            "worktree_clean": not changes, "worktree_changes": changes,
            "directories_present": directories, "layout_complete": all(directories.values()),
            "tracked_source_sha256": hashes, "missing_or_symlink_sources": missing,
            "scope": "relative layout and tracked public sources; local data/weights excluded"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--out", help="Optional local report, e.g. runs/sync/current.json")
    args = parser.parse_args()
    try:
        result = report(args.root)
        text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.out:
            output = Path(args.out)
            if output.exists():
                parser.error("Report exists; use a new filename to preserve earlier evidence")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
        print(text, end="")
    except (OSError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"Cannot read project state: {exc}\n")


if __name__ == "__main__":
    main()
