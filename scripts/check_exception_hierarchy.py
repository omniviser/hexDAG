#!/usr/bin/env python3
"""Check that all exception classes in hexdag/ inherit from HexDAGError.

Enforces the M1 audit convention: all framework exceptions must inherit from
HexDAGError so that `except HexDAGError` catches everything.

Usage:
    uv run python scripts/check_exception_hierarchy.py
    uv run python scripts/check_exception_hierarchy.py --path hexdag_plugins/azure/
    uv run python scripts/check_exception_hierarchy.py --verbose
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# Known HexDAGError subclasses (from hexdag/core/exceptions.py + domain/dag.py etc.)
KNOWN_HEXDAG_ERRORS: set[str] = {
    "HexDAGError",
    "ConfigurationError",
    "ValidationError",
    "ParseError",
    "ResourceNotFoundError",
    "DependencyError",
    "TypeMismatchError",
    "OrchestratorError",
    "NodeValidationError",
    "DirectedGraphError",
    "CycleDetectedError",
    "MissingDependencyError",
    "DuplicateNodeError",
    "SchemaCompatibilityError",
    "ResolveError",
    "PyTagError",
    "ExpressionError",
    "IncludeTagError",
    "PromptTemplateError",
    "MissingVariableError",
    "YamlPipelineBuilderError",
    "ComponentInstantiationError",
    "BodyExecutorError",
    "NodeExecutionError",
    "NodeTimeoutError",
    "PipelineRunnerError",
}

# Stdlib exception bases that should NOT be sole parents of framework exceptions
STDLIB_EXCEPTIONS: set[str] = {
    "Exception",
    "BaseException",
    "ValueError",
    "TypeError",
    "RuntimeError",
    "KeyError",
    "AttributeError",
    "IOError",
    "OSError",
    "IndexError",
    "NotImplementedError",
    "LookupError",
    "ArithmeticError",
}

# File → class names that are allowed to not inherit from HexDAGError
ALLOWLIST: dict[str, set[str]] = {
    # Pydantic BaseModel, not an actual exception class
    "hexdag/studio/server/routes/validate.py": {"ValidationError"},
}

# Directories excluded from checking by default
DEFAULT_EXCLUDES: list[str] = [
    "hexdag/studio/",
    "hexdag_plugins/",
]


def _get_base_names(class_node: ast.ClassDef) -> list[str]:
    """Extract base class names from a ClassDef node."""
    names = []
    for base in class_node.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            # Handle dotted names like exceptions.HexDAGError
            parts: list[str] = []
            current: ast.expr = base
            while isinstance(current, ast.Attribute):
                parts.insert(0, current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.insert(0, current.id)
            names.append(".".join(parts))
    return names


def _is_exception_class(class_node: ast.ClassDef) -> bool:
    """Check if a class looks like an exception (name ends with Error/Exception)."""
    return class_node.name.endswith("Error") or class_node.name.endswith("Exception")


def check_file(
    file_path: Path,
    relative_to: Path,
    verbose: bool = False,
) -> list[str]:
    """Check a single file for exception hierarchy violations.

    Returns list of error messages.
    """
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError):
        return []

    relative_path = str(file_path.relative_to(relative_to))
    allowed_classes = ALLOWLIST.get(relative_path, set())

    errors = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not _is_exception_class(node):
            continue
        if node.name in allowed_classes:
            continue
        # Skip the HexDAGError base class itself
        if node.name == "HexDAGError":
            continue

        base_names = _get_base_names(node)
        # Check if any base is a known HexDAGError subclass
        has_hexdag_base = any(
            name in KNOWN_HEXDAG_ERRORS or name.endswith(".HexDAGError") for name in base_names
        )
        # Check if bases are only stdlib exceptions
        has_only_stdlib = all(name in STDLIB_EXCEPTIONS for name in base_names)

        if not has_hexdag_base and has_only_stdlib and base_names:
            bases_str = ", ".join(base_names)
            errors.append(
                f"  {relative_path}:{node.lineno}: class '{node.name}' inherits from "
                f"'{bases_str}' instead of HexDAGError or a HexDAGError subclass.\n"
                f"    Fix: Change to 'class {node.name}(HexDAGError):' or use an existing subclass."
            )

        if verbose and has_hexdag_base:
            print(f"  OK: {relative_path}:{node.lineno} {node.name}")

    return errors


def main() -> int:
    """Check all Python files for exception hierarchy violations."""
    parser = argparse.ArgumentParser(
        description="Check that exception classes inherit from HexDAGError"
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

    args = parser.parse_args()
    check_path = Path(args.path).resolve()

    if not check_path.exists():
        print(f"Error: Path does not exist: {check_path}")
        return 1

    # Determine repo root for relative paths
    repo_root = Path.cwd().resolve()

    # Build exclude list
    excludes = DEFAULT_EXCLUDES + args.exclude

    all_errors: list[str] = []

    for py_file in sorted(check_path.rglob("*.py")):
        relative = str(py_file.relative_to(repo_root))

        # Skip excluded paths
        if any(relative.startswith(exc) for exc in excludes):
            continue

        errors = check_file(py_file, repo_root, verbose=args.verbose)
        all_errors.extend(errors)

    if all_errors:
        print(f"\n{'=' * 70}")
        print("EXCEPTION HIERARCHY VIOLATIONS")
        print(f"{'=' * 70}\n")
        for error in all_errors:
            print(error)
        print(f"\n{'=' * 70}")
        print(f"{len(all_errors)} violation(s) found.")
        print("All framework exceptions must inherit from HexDAGError.")
        print(f"{'=' * 70}")
        return 1

    print("All exception classes properly inherit from HexDAGError")
    return 0


if __name__ == "__main__":
    sys.exit(main())
