#!/usr/bin/env python3
"""Documentation build scripts."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


# Find the project root (where pyproject.toml is)
def find_project_root() -> Path:
    """Find the project root directory."""
    current = Path.cwd()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()  # Fallback to current directory


PROJECT_ROOT = find_project_root()


def build_docs() -> int:
    """Build the documentation."""
    source_dir = PROJECT_ROOT / "docs" / "source"
    build_dir = PROJECT_ROOT / "docs" / "build" / "html"
    result = subprocess.run(
        ["sphinx-build", "-b", "html", str(source_dir), str(build_dir)],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode


def clean_docs() -> int:
    """Clean the documentation build directory."""
    build_dir = PROJECT_ROOT / "docs" / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
        print(f"Cleaned {build_dir}")
    else:
        print(f"{build_dir} does not exist")
    return 0


def rebuild_docs() -> int:
    """Clean and rebuild the documentation."""
    clean_docs()
    return build_docs()


def check_docs() -> int:
    """Build docs with warnings as errors."""
    source_dir = PROJECT_ROOT / "docs" / "source"
    build_dir = PROJECT_ROOT / "docs" / "build" / "html"
    result = subprocess.run(
        ["sphinx-build", "-W", "-b", "html", str(source_dir), str(build_dir)],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode


def build() -> None:
    """Entry point for docs-build command."""
    sys.exit(build_docs())


def clean() -> None:
    """Entry point for docs-clean command."""
    sys.exit(clean_docs())


def rebuild() -> None:
    """Entry point for docs-rebuild command."""
    sys.exit(rebuild_docs())


def check() -> None:
    """Entry point for docs-check command."""
    sys.exit(check_docs())


def main() -> int:
    """Main entry point for hexdag-docs command."""
    parser = argparse.ArgumentParser(description="Documentation build tools")
    parser.add_argument(
        "command",
        choices=["build", "clean", "rebuild", "check", "autobuild"],
        help="Command to run",
    )

    args = parser.parse_args()

    commands = {
        "build": build_docs,
        "clean": clean_docs,
        "rebuild": rebuild_docs,
        "check": check_docs,
    }

    return commands[args.command]()


if __name__ == "__main__":
    sys.exit(main())
