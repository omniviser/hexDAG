#!/usr/bin/env python3
"""Rename a package for TestPyPI publishing.

TestPyPI requires unique package names, so we rename packages before
building for TestPyPI. This replaces the inline `sed` commands that
were previously duplicated across three separate release workflows.
"""

import argparse
import re
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rename a package for TestPyPI.")
    parser.add_argument("--pyproject", required=True, help="Path to pyproject.toml")
    parser.add_argument("--test-name", required=True, help="TestPyPI package name")
    args = parser.parse_args()

    path = Path(args.pyproject)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    content = path.read_text()
    new_content = re.sub(
        r'^name = ".*"',
        f'name = "{args.test_name}"',
        content,
        count=1,
        flags=re.MULTILINE,
    )

    if content == new_content:
        print(f"Warning: no name field found in {path}", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    path.write_text(new_content)
    print(f"Renamed package to {args.test_name} in {path}")  # noqa: T201


if __name__ == "__main__":
    sys.exit(main() or 0)
