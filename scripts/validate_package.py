#!/usr/bin/env python3
"""Validate package before publishing to PyPI."""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Validate the package can be built and published."""
    root = Path(__file__).parent.parent

    print("Building package...")
    result = subprocess.run(["python", "-m", "build"], cwd=root)
    if result.returncode != 0:
        print("Build failed!")
        return 1

    print("\nValidating with twine...")
    result = subprocess.run(
        ["python", "-m", "twine", "check", "dist/*"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        return 1

    required = ["LICENSE", "README.md", "pyproject.toml"]
    for f in required:
        if not (root / f).exists():
            print(f"Missing required file: {f}")
            return 1

    print("\nPackage validation passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
