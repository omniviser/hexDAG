#!/usr/bin/env python3
"""Check that adapter __init__ methods have explicit typed parameters.

Enforces the CLAUDE.md convention: all adapter/component __init__ methods MUST
use explicit typed parameters instead of **kwargs alone. This enables automatic
schema generation via SchemaGenerator.from_callable().

**kwargs-only signatures produce empty schemas in Studio UI, MCP server, and API.

Usage:
    uv run python scripts/check_init_params.py
    uv run python scripts/check_init_params.py --path hexdag_plugins/azure/
    uv run python scripts/check_init_params.py --verbose
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# Default paths to scan for adapters
DEFAULT_PATHS: list[str] = [
    "hexdag/builtin/adapters",
    "hexdag/adapters",
]

# Classes that are legitimately kwargs-only
ALLOWLIST: dict[str, set[str]] = {
    # Base classes that forward kwargs by design
    "hexdag/core/configurable.py": {"Configurable", "ConfigurableMacro"},
    "hexdag/core/yaml_macro.py": {"YamlMacro"},
    # Dynamic runtime-generated classes
    "hexdag/core/pipeline_builder/plugins/macro_definition.py": {"DynamicYamlMacro"},
    # Node factories: all config goes through __call__, not __init__
    "hexdag/builtin/nodes/llm_node.py": {"LLMNode"},
    "hexdag/builtin/nodes/agent_node.py": {"ReActAgentNode"},
    "hexdag/builtin/nodes/tool_call_node.py": {"ToolCallNode"},
}


def _is_kwargs_only_init(func_node: ast.FunctionDef) -> bool:
    """Check if an __init__ method has only self + **kwargs (no explicit params).

    Returns True if the signature is `def __init__(self, **kwargs)` with no
    other named parameters.
    """
    if func_node.name != "__init__":
        return False

    args = func_node.args

    # Count meaningful parameters (excluding 'self')
    regular_params = args.args[1:]  # Skip 'self'
    posonlyargs = args.posonlyargs
    kwonlyargs = args.kwonlyargs

    # If there are ANY explicit named parameters, it's fine
    if regular_params or posonlyargs or kwonlyargs:
        return False

    # Must have **kwargs to be a violation (if no params and no **kwargs, it's fine)
    return args.kwarg is not None


def check_file(
    file_path: Path,
    relative_to: Path,
    verbose: bool = False,
) -> list[str]:
    """Check a single file for kwargs-only __init__ violations.

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

        # Check each method looking for __init__
        for item in node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            if item.name != "__init__":
                continue

            if node.name in allowed_classes:
                if verbose:
                    print(f"  SKIP (allowlisted): {relative_path}:{node.lineno} {node.name}")
                continue

            if _is_kwargs_only_init(item):
                errors.append(
                    f"  {relative_path}:{item.lineno}: {node.name}.__init__ uses only **kwargs "
                    f"with no explicit parameters.\n"
                    f"    SchemaGenerator will produce an empty schema for this class.\n"
                    f"    Fix: Add explicit typed parameters "
                    f"(e.g., 'delay_seconds: float = 0.0')"
                )
            elif verbose:
                print(f"  OK: {relative_path}:{item.lineno} {node.name}.__init__")

    return errors


def main() -> int:
    """Check adapter classes for kwargs-only __init__ methods."""
    parser = argparse.ArgumentParser(description="Check adapter __init__ has explicit typed params")
    parser.add_argument(
        "--path",
        action="append",
        default=None,
        help="Path(s) to check (default: hexdag/builtin/adapters + hexdag/adapters)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()
    check_paths = [Path(p).resolve() for p in (args.path or DEFAULT_PATHS)]

    repo_root = Path.cwd().resolve()
    all_errors: list[str] = []
    files_checked = 0

    for check_path in check_paths:
        if not check_path.exists():
            if args.verbose:
                print(f"  Skipping non-existent path: {check_path}")
            continue

        for py_file in sorted(check_path.rglob("*.py")):
            relative = str(py_file.relative_to(repo_root))
            if "__pycache__" in relative:
                continue

            files_checked += 1
            errors = check_file(py_file, repo_root, verbose=args.verbose)
            all_errors.extend(errors)

    if all_errors:
        print(f"\n{'=' * 70}")
        print("KWARGS-ONLY __init__ VIOLATIONS")
        print(f"{'=' * 70}\n")
        for error in all_errors:
            print(error)
        print(f"\n{'=' * 70}")
        print(f"{len(all_errors)} violation(s) found ({files_checked} files checked).")
        print("Adapter __init__ must have explicit typed parameters for SchemaGenerator.")
        print(f"{'=' * 70}")
        return 1

    print(f"All adapter __init__ methods have explicit parameters ({files_checked} files checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
