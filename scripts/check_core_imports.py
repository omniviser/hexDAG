#!/usr/bin/env python3
"""Check that hexdag/core doesn't import from hexdag/adapters or hexdag/builtin.

This enforces the architectural principle that core contains pure abstractions
and should not depend on concrete implementations.

Configuration is loaded from .check_core_imports.yaml
"""

import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)


def load_config() -> dict[str, Any]:
    """Load configuration from .check_core_imports.yaml"""
    config_path = Path(".check_core_imports.yaml")

    if not config_path.exists():
        print(f"Error: Config file {config_path} not found")
        sys.exit(1)

    with Path.open(config_path) as f:
        return yaml.safe_load(f)


def check_file(
    file_path: Path,
    forbidden_patterns: list[str],
    allowed_exceptions: dict[str, list[str]],
) -> list[str]:
    """Check a single file for forbidden imports.

    Returns list of error messages (empty if no violations).
    """
    errors = []
    content = file_path.read_text()

    relative_path = str(file_path).replace(str(Path.cwd()) + "/", "")
    allowed_imports = set(allowed_exceptions.get(relative_path, []))

    for line_num, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()

        # Skip comments and docstrings
        if stripped.startswith("#") or '"""' in stripped or "'''" in stripped:
            continue

        for pattern in forbidden_patterns:
            if re.match(pattern, stripped):
                # Check if this specific import is allowed
                if stripped in allowed_imports:
                    continue

                errors.append(
                    f"{relative_path}:{line_num}: "
                    f"Core module imports from adapters/builtin: {stripped}\n"
                    f"  Core should only define abstractions (ports), not import implementations."
                )

    return errors


def main() -> int:
    """Check all Python files in hexdag/core for forbidden imports."""
    # Load configuration
    config = load_config()
    check_dir = Path(config["check_directory"])
    forbidden_patterns = config["forbidden_patterns"]
    allowed_exceptions = config.get("allowed_exceptions", {})

    if not check_dir.exists():
        print(f"Error: {check_dir} not found. Run from repository root.")
        return 1

    all_errors = []

    # Check all .py files in configured directory
    for py_file in check_dir.rglob("*.py"):
        errors = check_file(py_file, forbidden_patterns, allowed_exceptions)
        all_errors.extend(errors)

    if all_errors:
        print("❌ Architecture violation: Core importing from adapters/builtin\n")
        for error in all_errors:
            print(error)
        print("\n" + "=" * 70)
        print("ALLOWED EXCEPTIONS (see .check_core_imports.yaml):")
        for file_path, imports in allowed_exceptions.items():
            print(f"\n  {file_path}:")
            for imp in imports:
                print(f"    - {imp}")
        print("=" * 70)
        return 1

    print("✅ All core imports are clean (no adapters/builtin dependencies)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
