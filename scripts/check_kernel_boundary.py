"""Enforce the kernel boundary (syscall table).

User-space code (hexdag/api/, hexdag/cli/, hexdag/mcp_server.py) must
only import from ``hexdag.kernel`` — never from kernel submodules like
``hexdag.kernel.orchestration.models``.

Kernel-space code (hexdag/kernel/, hexdag/stdlib/, hexdag/compiler/,
hexdag/drivers/) may freely import from kernel submodules.

Usage::

    uv run python scripts/check_kernel_boundary.py
    uv run python scripts/check_kernel_boundary.py --verbose
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Root of the hexdag package
_HEXDAG_ROOT = Path(__file__).resolve().parent.parent / "hexdag"

# User-space directories/files that must respect the boundary
_USER_SPACE = [
    _HEXDAG_ROOT / "api",
    _HEXDAG_ROOT / "cli",
    _HEXDAG_ROOT / "mcp_server.py",
]

# Pattern that matches deep kernel submodule imports:
#   from hexdag.kernel.something import X
#   import hexdag.kernel.something
# But NOT:
#   from hexdag.kernel import X
#   import hexdag.kernel
_DEEP_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+hexdag\.kernel\.\S+\s+import|import\s+hexdag\.kernel\.\S+)"
)

# Allow TYPE_CHECKING-guarded imports (they don't execute at runtime)
_TYPE_CHECKING_RE = re.compile(r"^\s*if\s+TYPE_CHECKING\s*:")


def _check_file(path: Path, verbose: bool = False) -> list[tuple[Path, int, str]]:
    """Check a single file for kernel boundary violations.

    Returns list of (path, line_number, line_text) tuples.
    """
    violations: list[tuple[Path, int, str]] = []

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return violations

    in_type_checking_block = False
    indent_level: int | None = None

    for lineno_0, line in enumerate(lines):
        lineno = lineno_0 + 1

        # Track TYPE_CHECKING blocks
        if _TYPE_CHECKING_RE.match(line):
            in_type_checking_block = True
            indent_level = len(line) - len(line.lstrip())
            continue

        # Detect when we leave a TYPE_CHECKING block
        if in_type_checking_block and line.strip():
            current_indent = len(line) - len(line.lstrip())
            if (
                indent_level is not None
                and current_indent <= indent_level
                and not line.strip().startswith(("else", "elif"))
            ):
                in_type_checking_block = False
                indent_level = None

        # Skip violations inside TYPE_CHECKING blocks
        if in_type_checking_block:
            continue

        # Skip comments and string-only lines
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue

        if _DEEP_IMPORT_RE.match(line):
            violations.append((path, lineno, line.rstrip()))
            if verbose:
                print(f"  VIOLATION {path.relative_to(_HEXDAG_ROOT.parent)}:{lineno}: {stripped}")

    return violations


def main() -> int:
    """Run the kernel boundary check."""
    parser = argparse.ArgumentParser(description="Check kernel boundary enforcement")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show each violation")
    args = parser.parse_args()

    all_violations: list[tuple[Path, int, str]] = []

    for entry in _USER_SPACE:
        if entry.is_file():
            all_violations.extend(_check_file(entry, verbose=args.verbose))
        elif entry.is_dir():
            for py_file in sorted(entry.rglob("*.py")):
                all_violations.extend(_check_file(py_file, verbose=args.verbose))

    if all_violations:
        print(f"\n{len(all_violations)} kernel boundary violation(s) found!\n")
        if not args.verbose:
            for path, lineno, line in all_violations:
                rel = path.relative_to(_HEXDAG_ROOT.parent)
                print(f"  {rel}:{lineno}: {line.strip()}")
        print(
            "\nUser-space code must import from 'hexdag.kernel', "
            "not from kernel submodules.\n"
            "Example: from hexdag.kernel import Orchestrator  "
            "(not from hexdag.kernel.orchestration.orchestrator)"
        )
        return 1

    print("Kernel boundary check passed: no violations found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
