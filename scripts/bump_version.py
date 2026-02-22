#!/usr/bin/env python3
"""Bump the version in a pyproject.toml file.

Simple version bumping without commit history scanning.

Rules:
  dev:    0.7.0       -> 0.8.0.dev1   (bump minor, add .dev1)
          0.8.0.dev1  -> 0.8.0.dev2   (increment dev number)
  stable: 0.8.0.dev2  -> 0.8.0        (strip .devN suffix)
          0.8.0       -> error         (must go through dev first)
"""

import argparse
import re
import sys
from pathlib import Path


def parse_version(version: str) -> tuple[int, int, int, int | None]:
    """Parse version string into (major, minor, patch, dev_number|None)."""
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:\.dev(\d+))?$", version)
    if not m:
        print(f"Error: cannot parse version '{version}'", file=sys.stderr)  # noqa: T201
        sys.exit(1)
    major, minor, patch = int(m[1]), int(m[2]), int(m[3])
    dev = int(m[4]) if m[4] is not None else None
    return major, minor, patch, dev


def bump(version: str, release_type: str) -> str:
    """Compute the next version."""
    major, minor, patch, dev = parse_version(version)

    if release_type == "dev":
        if dev is not None:
            # 0.8.0.dev1 -> 0.8.0.dev2
            return f"{major}.{minor}.{patch}.dev{dev + 1}"
        # 0.7.0 -> 0.8.0.dev1
        return f"{major}.{minor + 1}.{patch}.dev1"

    if release_type == "stable":
        if dev is not None:
            # 0.8.0.dev2 -> 0.8.0
            return f"{major}.{minor}.{patch}"
        print(  # noqa: T201
            f"Error: version '{version}' is already stable. Run a dev release first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Error: unknown release_type '{release_type}'", file=sys.stderr)  # noqa: T201
    sys.exit(1)


def update_pyproject(path: Path, old_version: str, new_version: str) -> None:
    """Replace all version = "old" with version = "new" in pyproject.toml."""
    content = path.read_text()
    updated = content.replace(f'version = "{old_version}"', f'version = "{new_version}"')
    if content == updated:
        print(f"Warning: no version replacement made in {path}", file=sys.stderr)  # noqa: T201
    path.write_text(updated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bump version in pyproject.toml.")
    parser.add_argument(
        "--pyproject",
        required=True,
        help="Path to pyproject.toml",
    )
    parser.add_argument(
        "--release-type",
        required=True,
        choices=("dev", "stable"),
    )
    args = parser.parse_args()

    path = Path(args.pyproject)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    # Extract current version from [project] section
    content = path.read_text()
    m = re.search(r'^version = "([^"]+)"', content, re.MULTILINE)
    if not m:
        print(f"Error: no version field found in {path}", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    old_version = m[1]
    new_version = bump(old_version, args.release_type)

    update_pyproject(path, old_version, new_version)
    print(new_version)  # noqa: T201


if __name__ == "__main__":
    sys.exit(main() or 0)
