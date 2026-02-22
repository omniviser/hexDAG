#!/usr/bin/env python3
"""Check that Protocol classes use ... (Ellipsis) not pass for method bodies.

Enforces the H2 audit convention: Protocol abstract methods use Ellipsis (...),
not pass statements, for consistent convention across all port definitions.

Usage:
    uv run python scripts/check_port_protocols.py
    uv run python scripts/check_port_protocols.py --path hexdag_plugins/storage/ports/
    uv run python scripts/check_port_protocols.py --verbose
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# Default path to scan for port protocols
DEFAULT_PATH = "hexdag/kernel/ports"


def _has_protocol_base(class_node: ast.ClassDef) -> bool:
    """Check if class inherits from Protocol."""
    for base in class_node.bases:
        if isinstance(base, ast.Name) and base.id == "Protocol":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "Protocol":
            return True
    return False


def _body_is_pass(body: list[ast.stmt]) -> bool:
    """Check if a method body is just 'pass' (optionally after a docstring)."""
    stmts = body
    # Skip leading docstring
    if stmts and isinstance(stmts[0], ast.Expr) and isinstance(stmts[0].value, ast.Constant):
        stmts = stmts[1:]
    # Check remaining is a single Pass
    return len(stmts) == 1 and isinstance(stmts[0], ast.Pass)


def _body_is_raise_not_implemented(body: list[ast.stmt]) -> bool:
    """Check if a method body raises NotImplementedError (anti-pattern for Protocols)."""
    stmts = body
    # Skip leading docstring
    if stmts and isinstance(stmts[0], ast.Expr) and isinstance(stmts[0].value, ast.Constant):
        stmts = stmts[1:]
    if len(stmts) != 1:
        return False
    stmt = stmts[0]
    if not isinstance(stmt, ast.Raise) or stmt.exc is None:
        return False
    exc = stmt.exc
    # raise NotImplementedError or raise NotImplementedError(...)
    if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
        return True
    if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
        return exc.func.id == "NotImplementedError"
    return False


def check_file(
    file_path: Path,
    relative_to: Path,
    verbose: bool = False,
) -> list[str]:
    """Check a single file for Protocol body convention violations.

    Returns list of error messages.
    """
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError):
        return []

    relative_path = str(file_path.relative_to(relative_to))
    errors = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not _has_protocol_base(node):
            continue

        if verbose:
            print(f"  Checking Protocol: {relative_path}:{node.lineno} {node.name}")

        # Check each method in the Protocol
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            if _body_is_pass(item.body):
                errors.append(
                    f"  {relative_path}:{item.lineno}: Protocol method "
                    f"'{node.name}.{item.name}' uses 'pass' instead of '...' (Ellipsis).\n"
                    f"    Fix: Replace 'pass' with '...'"
                )
            elif _body_is_raise_not_implemented(item.body):
                errors.append(
                    f"  {relative_path}:{item.lineno}: Protocol method "
                    f"'{node.name}.{item.name}' raises NotImplementedError.\n"
                    f"    Fix: Replace 'raise NotImplementedError(...)' with '...'"
                )

    return errors


def main() -> int:
    """Check Protocol classes for body convention violations."""
    parser = argparse.ArgumentParser(description="Check Protocol methods use ... not pass")
    parser.add_argument(
        "--path",
        default=DEFAULT_PATH,
        help=f"Path to check (default: {DEFAULT_PATH})",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()
    check_path = Path(args.path).resolve()

    if not check_path.exists():
        print(f"Error: Path does not exist: {check_path}")
        return 1

    repo_root = Path.cwd().resolve()
    all_errors: list[str] = []

    py_files = sorted(check_path.rglob("*.py"))
    for py_file in py_files:
        errors = check_file(py_file, repo_root, verbose=args.verbose)
        all_errors.extend(errors)

    if all_errors:
        print(f"\n{'=' * 70}")
        print("PORT PROTOCOL CONVENTION VIOLATIONS")
        print(f"{'=' * 70}\n")
        for error in all_errors:
            print(error)
        print(f"\n{'=' * 70}")
        print(f"{len(all_errors)} violation(s) found.")
        print("Protocol methods must use '...' (Ellipsis), not 'pass'.")
        print(f"{'=' * 70}")
        return 1

    file_count = len(py_files)
    print(f"All Protocol methods use '...' convention ({file_count} files checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
