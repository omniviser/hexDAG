#!/usr/bin/env python3
"""Check that JSON schemas haven't been manually modified.

This script ensures schemas are always generated, never manually edited.
It compares committed schema files against freshly generated ones.

Exit codes:
    0: Schemas are up-to-date
    1: Schemas need regeneration or were manually modified
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Import schema generator
from scripts.generate_schemas import (
    generate_hexdag_config_schema,
    generate_pipeline_schema,
)

SCHEMAS_DIR = Path(__file__).parent.parent / "schemas"


def load_json_file(filepath: Path) -> dict | None:
    """Load JSON file, return None if doesn't exist."""
    try:
        with filepath.open() as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def normalize_json(data: dict) -> str:
    """Normalize JSON for comparison (sorted keys, consistent formatting)."""
    return json.dumps(data, sort_keys=True, indent=2)


def check_schema_file(name: str, current_file: Path, generator_func, **kwargs) -> bool:
    """Check if a schema file matches its generated version.

    Args:
        name: Schema name for display
        current_file: Path to current schema file
        generator_func: Function to generate the schema
        **kwargs: Arguments to pass to generator function

    Returns:
        True if schema is up-to-date, False otherwise
    """
    # Load current schema
    current_schema = load_json_file(current_file)
    if current_schema is None:
        print(f"✗ {name}: File doesn't exist or is invalid JSON")
        return False

    # Generate fresh schema
    try:
        generated_schema = generator_func(**kwargs)
    except Exception as e:
        print(f"✗ {name}: Failed to generate schema: {e}")
        return False

    # Compare normalized versions
    current_normalized = normalize_json(current_schema)
    generated_normalized = normalize_json(generated_schema)

    if current_normalized != generated_normalized:
        print(f"✗ {name}: Schema is out of sync with generated version")
        print(f"   File: {current_file}")
        print("   Please run: uv run python scripts/generate_schemas.py")
        return False

    print(f"✓ {name}: Up-to-date")
    return True


def main() -> int:
    """Check all schema files are up-to-date.

    Returns:
        0 if all schemas are valid, 1 otherwise
    """
    print("Checking JSON schemas are auto-generated (not manually edited)...")

    # Check each schema file
    checks = [
        (
            "Pipeline schema",
            SCHEMAS_DIR / "pipeline-schema.json",
            generate_pipeline_schema,
            {},
        ),
        (
            "HexDAG config schema",
            SCHEMAS_DIR / "hexdag-config-schema.json",
            generate_hexdag_config_schema,
            {},
        ),
    ]

    all_valid = True
    for name, filepath, generator, kwargs in checks:
        if not check_schema_file(name, filepath, generator, **kwargs):
            all_valid = False

    if not all_valid:
        print("\n" + "=" * 70)
        print("ERROR: Schema files are out of sync!")
        print("=" * 70)
        print("\nSchemas must be auto-generated, never manually edited.")
        print("To fix this, run:\n")
        print("    uv run python scripts/generate_schemas.py")
        print("\nThen stage and commit the regenerated files.")
        return 1

    print("\n✓ All schemas are up-to-date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
