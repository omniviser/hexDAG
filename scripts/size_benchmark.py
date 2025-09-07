"""Size benchmark."""

import argparse
import importlib
import json
import os
import platform
import sys
from pathlib import Path


def dist_path_for(pkg: str) -> Path:
    """Distribution path."""
    mod = importlib.import_module(pkg)
    p = Path(getattr(mod, "__file__", "")).resolve()
    return p.parent if p.is_file() else p


def folder_size_bytes(path: Path) -> int:
    """Size of folder."""
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            fp = Path(root) / fn
            try:
                total += fp.stat().st_size
            except FileNotFoundError:
                pass
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    path = dist_path_for(args.package)
    size = folder_size_bytes(path)
    out = {
        "package": args.package,
        "dist_path": str(path),
        "size_bytes": size,
        "size_kib": round(size / 1024, 2),
        "size_mib": round(size / 1024 / 1024, 2),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}: {out}")
