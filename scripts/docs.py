#!/usr/bin/env python3
"""Documentation build scripts."""

import argparse
import shutil
import subprocess  # nosec B404
import sys
from collections.abc import Sequence
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


def _run_sphinx(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Execute sphinx-build with a trusted executable path."""
    sphinx_executable = shutil.which("sphinx-build")
    if sphinx_executable is None:
        raise FileNotFoundError(
            "sphinx-build executable not found. Install Sphinx to build documentation."
        )
    return subprocess.run(  # nosec B603
        [sphinx_executable, *args],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        check=False,
    )


def build_docs() -> int:
    """Build the documentation."""
    source_dir = PROJECT_ROOT / "docs" / "source"
    build_dir = PROJECT_ROOT / "docs" / "build" / "html"
    result = _run_sphinx(["-b", "html", str(source_dir), str(build_dir)])
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
    result = _run_sphinx(["-W", "-b", "html", str(source_dir), str(build_dir)])
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
