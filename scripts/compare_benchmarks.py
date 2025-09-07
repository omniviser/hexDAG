"""Compare benchmarks."""

import argparse
import json
import sys
from pathlib import Path


def load_index(dir_path: Path):
    """Load benchmark index."""
    data = {}
    for p in dir_path.glob("*.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                j = json.load(f)
            data[p.stem] = j
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] Skipping {p.name}: {exc}", file=sys.stderr)
    return data


def pct_increase(old, new):
    """Percent increase."""
    if old == 0:
        return float("inf") if new > 0 else 0.0
    return (new - old) / old * 100.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--current-dir", required=True)
    ap.add_argument("--max-import-increase", type=float, default=20.0)
    ap.add_argument("--max-size-increase", type=float, default=10.0)
    args = ap.parse_args()

    baseline = load_index(Path(args.baseline_dir))
    current = load_index(Path(args.current_dir))

    if not baseline:
        print("No baseline artifacts found. Skipping regression check.")
        sys.exit(0)

    failures = []
    for key, cur in current.items():
        base = baseline.get(key)
        if not base:
            print(f"[WARN] No baseline for {key}, skipping.")
            continue

        bi = float(base.get("import_time_s", 0))
        ci = float(cur.get("import_time_s", 0))
        imp_delta = pct_increase(bi, ci)

        bs = int(base.get("size_bytes", 0))
        cs = int(cur.get("size_bytes", 0))
        size_delta = pct_increase(bs, cs)

        print(f"{key}: import +{imp_delta:.1f}%  size +{size_delta:.1f}%")
        if imp_delta > args.max_import_increase:
            failures.append(f"{key}: import_time +{imp_delta:.1f}% > {args.max_import_increase}%")
        if size_delta > args.max_size_increase:
            failures.append(f"{key}: size +{size_delta:.1f}% > {args.max_size_increase}%")

    if failures:
        print("REGRESSION DETECTED:")
        for f in failures:
            print(" -", f)
        sys.exit(1)

    print("No regressions beyond thresholds.")
