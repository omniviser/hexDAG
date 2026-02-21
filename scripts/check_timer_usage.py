#!/usr/bin/env python3
"""Check that duration measurement uses Timer, not inline time.time()/perf_counter().

Enforces the M3 audit convention: use Timer from hexdag/core/utils/node_timer.py
for performance timing. Only wall-clock data timestamps may use time.time().

Usage:
    uv run python scripts/check_timer_usage.py
    uv run python scripts/check_timer_usage.py --path hexdag_plugins/azure/
    uv run python scripts/check_timer_usage.py --verbose
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# Files where time.time()/time.perf_counter() is legitimately needed
ALLOWLIST: dict[str, set[str]] = {
    # Timer implementation itself
    "hexdag/core/utils/node_timer.py": {"time.perf_counter", "time.time"},
    # Wall-clock timestamps for data storage (not duration measurement)
    "hexdag/builtin/adapters/memory/state_memory.py": {"time.time"},
    "hexdag/builtin/adapters/memory/session_memory.py": {"time.time"},
}

# Pre-existing violations from before the M3 audit.
# These produce warnings but don't fail the hook.
GRANDFATHERED: set[str] = {
    # Core
    "hexdag/api/execution.py",
    "hexdag/studio/server/routes/execute.py",
    "hexdag/builtin/adapters/openai/openai_adapter.py",
    "hexdag/adapters/executors/local_executor.py",
    "hexdag/builtin/nodes/tool_call_node.py",
    # Plugins — pre-existing timer/timestamp usage
    "hexdag_plugins/azure/adapters/blob.py",
    "hexdag_plugins/azure/adapters/cosmos.py",
    "hexdag_plugins/azure/adapters/keyvault.py",
    "hexdag_plugins/azure/adapters/openai.py",
    "hexdag_plugins/azure/tests/test_azure_keyvault_adapter.py",
    "hexdag_plugins/storage/adapters/file/local.py",
    "hexdag_plugins/storage/adapters/vector/pgvector.py",
}

DEFAULT_EXCLUDES: list[str] = [
    "hexdag/studio/",
    "hexdag_plugins/",
    "__pycache__/",
]


class TimerUsageChecker(ast.NodeVisitor):
    """AST visitor to detect time.time() and time.perf_counter() calls."""

    FLAGGED_CALLS: set[str] = {"time.time", "time.perf_counter"}

    def __init__(self, filename: str) -> None:
        """Initialize checker.

        Args
        ----
            filename: Path to the file being checked
        """
        self.filename = filename
        self.violations: list[tuple[int, int, str]] = []  # (line, col, call_name)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        """Visit function calls to detect time.time()/perf_counter()."""
        call_name = self._get_call_name(node)
        if call_name in self.FLAGGED_CALLS:
            self.violations.append((node.lineno, node.col_offset, call_name))
        self.generic_visit(node)

    def _get_call_name(self, node: ast.Call) -> str | None:
        """Extract full dotted name of a function call."""
        if isinstance(node.func, ast.Attribute):
            parts: list[str] = []
            current: ast.expr = node.func
            while isinstance(current, ast.Attribute):
                parts.insert(0, current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.insert(0, current.id)
            return ".".join(parts) if parts else None
        return None


def check_file(
    file_path: Path,
    relative_to: Path,
    verbose: bool = False,
) -> tuple[list[str], list[str]]:
    """Check a single file for timer usage violations.

    Returns (errors, warnings) tuple.
    """
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError):
        return [], []

    relative_path = str(file_path.relative_to(relative_to))
    allowed_calls = ALLOWLIST.get(relative_path, set())
    is_grandfathered = relative_path in GRANDFATHERED

    checker = TimerUsageChecker(relative_path)
    checker.visit(tree)

    errors = []
    warnings = []

    for line, col, call_name in checker.violations:
        if call_name in allowed_calls:
            continue

        msg = (
            f"  {relative_path}:{line}:{col}: {call_name}() — "
            f"Use Timer from hexdag.core.utils.node_timer instead"
        )

        if is_grandfathered:
            warnings.append(f"  WARNING (grandfathered) {relative_path}:{line}: {call_name}()")
            if verbose:
                print(msg)
        else:
            errors.append(msg)

    return errors, warnings


def main() -> int:
    """Check for inline timer usage violations."""
    parser = argparse.ArgumentParser(
        description="Check Timer usage instead of inline time.time/perf_counter"
    )
    parser.add_argument(
        "--path",
        default="hexdag",
        help="Path to check (default: hexdag/)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional exclude patterns",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat grandfathered violations as errors too",
    )

    args = parser.parse_args()
    check_path = Path(args.path).resolve()

    if not check_path.exists():
        print(f"Error: Path does not exist: {check_path}")
        return 1

    repo_root = Path.cwd().resolve()

    # Determine excludes: use defaults for core, none for plugins
    if check_path.name == "hexdag" and "plugins" not in check_path.name:
        excludes = DEFAULT_EXCLUDES + args.exclude
    else:
        excludes = ["__pycache__/"] + args.exclude

    all_errors: list[str] = []
    all_warnings: list[str] = []

    for py_file in sorted(check_path.rglob("*.py")):
        relative = str(py_file.relative_to(repo_root))

        if any(relative.startswith(exc) for exc in excludes):
            continue

        errors, warnings = check_file(py_file, repo_root, verbose=args.verbose)
        all_errors.extend(errors)
        all_warnings.extend(warnings)

    if args.strict:
        all_errors.extend(all_warnings)
        all_warnings = []

    if all_warnings and args.verbose:
        print(f"\nGrandfathered warnings ({len(all_warnings)}):")
        for warning in all_warnings:
            print(warning)

    if all_errors:
        print(f"\n{'=' * 70}")
        print("TIMER USAGE VIOLATIONS")
        print(f"{'=' * 70}\n")
        for error in all_errors:
            print(error)
        if all_warnings:
            print(f"\n  ({len(all_warnings)} grandfathered warnings not shown, use --verbose)")
        print(f"\n{'=' * 70}")
        print(f"{len(all_errors)} new violation(s) found.")
        print("Use Timer from hexdag.core.utils.node_timer for duration measurement.")
        print(f"{'=' * 70}")
        return 1

    warning_msg = f" ({len(all_warnings)} grandfathered)" if all_warnings else ""
    print(f"No new timer usage violations{warning_msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
