"""Check for lazy (function-level) imports in core hexdag packages.

Detects import statements inside function or method bodies that should
be at the module top level.  Intentional lazy imports can be annotated
with ``# lazy: <reason>`` to suppress the warning.

By default only scans the core packages (kernel/, compiler/, stdlib/)
where top-level imports are enforced.  Use ``--all`` to scan everything.

Usage::

    uv run python scripts/check_lazy_imports.py            # core packages only
    uv run python scripts/check_lazy_imports.py --all      # entire hexdag/
    uv run python scripts/check_lazy_imports.py --verbose  # show allowed lazy imports

Exit codes::

    0 — no violations
    1 — violations found
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# Core packages where top-level imports are strictly enforced
CORE_DIRS = [
    BASE_DIR / "hexdag" / "kernel",
    BASE_DIR / "hexdag" / "compiler",
    BASE_DIR / "hexdag" / "stdlib",
]

# All hexdag packages (use with --all)
ALL_DIRS = [BASE_DIR / "hexdag"]

# Functions that are known lazy-loading patterns (e.g. module __getattr__)
ALLOWED_FUNCTIONS = {"__getattr__"}


def _is_import(node: ast.stmt) -> bool:
    return isinstance(node, (ast.Import, ast.ImportFrom))


def _has_lazy_comment(
    source_lines: list[str],
    lineno: int,
    end_lineno: int | None = None,
) -> str | None:
    """Return the reason string if any line in the import has a ``# lazy: <reason>`` comment.

    For multi-line imports (``from foo import (\\n    bar,  # lazy: reason\\n)``),
    we check every line from ``lineno`` to ``end_lineno``.
    """
    if lineno < 1 or lineno > len(source_lines):
        return None
    last = end_lineno if end_lineno else lineno
    for ln in range(lineno, min(last + 1, len(source_lines) + 1)):
        line = source_lines[ln - 1]
        idx = line.find("# lazy:")
        if idx != -1:
            return line[idx + 7 :].strip() or "unspecified"
    return None


def _is_type_checking_guard(node: ast.If) -> bool:
    """Check if an ``if`` block is a ``TYPE_CHECKING`` guard."""
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def check_file(filepath: Path, verbose: bool = False) -> list[str]:
    """Return list of violation messages for a single file."""
    source = filepath.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []

    source_lines = source.splitlines()
    violations: list[str] = []
    rel_path = filepath.relative_to(BASE_DIR)

    def _walk_body(body: list[ast.stmt], in_function: str | None) -> None:
        for node in body:
            # Skip TYPE_CHECKING guards entirely
            if isinstance(node, ast.If) and _is_type_checking_guard(node):
                continue

            # Check imports inside functions/methods
            if in_function and _is_import(node):
                # Skip known lazy-loading patterns
                if in_function in ALLOWED_FUNCTIONS:
                    if verbose:
                        print(
                            f"  [skip] {rel_path}:{node.lineno} "
                            f"in {in_function}() — allowed pattern"
                        )
                    continue

                reason = _has_lazy_comment(source_lines, node.lineno, node.end_lineno)
                if reason:
                    if verbose:
                        print(
                            f"  [allowed] {rel_path}:{node.lineno} "
                            f"in {in_function}() — lazy: {reason}"
                        )
                else:
                    violations.append(
                        f"{rel_path}:{node.lineno}: "
                        f"import inside {in_function}() — "
                        f"move to top level or add '# lazy: <reason>'"
                    )
                continue

            # Recurse into function/method bodies
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_name = node.name
                _walk_body(node.body, fn_name)
                continue

            # Recurse into class bodies (methods are inside classes)
            if isinstance(node, ast.ClassDef):
                _walk_body(node.body, in_function)
                continue

            # Recurse into try/except inside functions
            if isinstance(node, ast.Try) and in_function:
                _walk_body(node.body, in_function)
                for handler in node.handlers:
                    _walk_body(handler.body, in_function)
                continue

            # Recurse into if/else inside functions
            if isinstance(node, ast.If) and in_function:
                _walk_body(node.body, in_function)
                _walk_body(node.orelse, in_function)

    _walk_body(tree.body, None)
    return violations


def main() -> int:
    """Run the lazy import checker."""
    parser = argparse.ArgumentParser(description="Check for lazy imports in hexdag/")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show allowed lazy imports")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scan all hexdag/ packages (default: only kernel/, compiler/, stdlib/)",
    )
    args = parser.parse_args()

    scan_dirs = ALL_DIRS if args.all else CORE_DIRS
    all_violations: list[str] = []

    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        for py_file in sorted(scan_dir.rglob("*.py")):
            violations = check_file(py_file, verbose=args.verbose)
            all_violations.extend(violations)

    if all_violations:
        print(f"\n{'=' * 60}")
        print(f"Found {len(all_violations)} lazy import violation(s):")
        print(f"{'=' * 60}\n")
        for v in all_violations:
            print(f"  {v}")
        print(
            "\nFix: move imports to module top level, "
            "or add '# lazy: <reason>' comment to suppress."
        )
        return 1

    print("No lazy import violations found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
