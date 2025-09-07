"""Measure execution time."""

import argparse
import importlib
import json
import platform
import sys
import time


def measure(pkg: str):
    """Measure execution time."""
    t0 = time.perf_counter()
    mod = importlib.import_module(pkg)
    t1 = time.perf_counter()
    return {
        "package": pkg,
        "module": getattr(mod, "__name__", pkg),
        "import_time_s": round(t1 - t0, 6),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = measure(args.package)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.out}: {res}")
