#!/usr/bin/env python3
"""Verify that the YAML pipeline schema and builder code are synchronized.

Detects two kinds of drift:
1. **Schema-only properties** — defined in ``schemas/pipeline-schema.yaml`` under
   ``spec.properties`` but never referenced in builder/plugin/config code.  Users
   can write schema-valid YAML that silently does nothing.
2. **Code-only properties** — processed by the builder or ``PipelineConfig`` but
   absent from the schema.  Users get no IDE autocompletion or validation for
   these properties.

Configuration is loaded from ``.check_yaml_schema_sync.yaml`` (allowlist).

Usage:
    uv run python scripts/check_yaml_schema_sync.py
    uv run python scripts/check_yaml_schema_sync.py --verbose
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)


# ============================================================================
# Config
# ============================================================================

_CONFIG_FILE = ".check_yaml_schema_sync.yaml"

_SCHEMA_PATH = Path("schemas/pipeline-schema.yaml")

_BUILDER_PATHS: list[str] = [
    "hexdag/compiler/yaml_builder.py",
    "hexdag/compiler/yaml_validator.py",
    "hexdag/kernel/domain/pipeline_config.py",
    "hexdag/compiler/plugins/node_entity.py",
    "hexdag/compiler/plugins/macro_entity.py",
    "hexdag/compiler/plugins/macro_definition.py",
    "hexdag/compiler/plugins/config_definition.py",
    "hexdag/compiler/preprocessing/env_vars.py",
    "hexdag/compiler/preprocessing/include.py",
    "hexdag/compiler/preprocessing/template.py",
]


def _load_config() -> dict[str, Any]:
    """Load allowlist configuration.

    Returns
    -------
    dict
        Configuration with ``schema_only_allowed`` and ``code_only_allowed`` keys.
    """
    config_path = Path(_CONFIG_FILE)
    if not config_path.exists():
        return {"schema_only_allowed": [], "code_only_allowed": []}

    with Path.open(config_path) as f:
        return yaml.safe_load(f) or {}


# ============================================================================
# Schema extraction
# ============================================================================


def _extract_spec_properties(schema_path: Path) -> dict[str, str]:
    """Extract top-level ``spec.properties`` from the pipeline schema.

    Returns
    -------
    dict[str, str]
        Mapping of property name → description.
    """
    content = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    spec_props = content.get("properties", {}).get("spec", {}).get("properties", {})
    return {name: prop.get("description", "(no description)") for name, prop in spec_props.items()}


def _extract_node_level_properties(schema_path: Path) -> set[str]:
    """Extract all node-level property names from ``$defs`` in the schema.

    Collects properties from the base ``Node`` definition (including
    top-level and ``spec.properties``) as well as each concrete node type's
    ``spec.properties``.  This allows the sync checker to recognise
    node-level ``spec.get(...)`` accesses in builder code without
    false-flagging them as missing pipeline-level properties.

    Returns
    -------
    set[str]
        Set of all node-level property names.
    """
    content = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    defs = content.get("$defs", {})
    props: set[str] = set()

    for def_schema in defs.values():
        # Top-level node properties (kind, metadata, spec, dependencies, wait_for)
        for name in def_schema.get("properties", {}):
            props.add(name)

        # spec.properties within each node type
        spec_schema = def_schema.get("properties", {}).get("spec", {})
        for name in spec_schema.get("properties", {}):
            props.add(name)

    return props


# ============================================================================
# Code scanning
# ============================================================================


def _scan_code_for_property(prop: str, builder_sources: dict[str, str]) -> list[str]:
    """Check if *prop* appears in builder/config code as a dict key access.

    Looks for patterns like:
    - ``spec.get("prop"`` or ``spec["prop"]``
    - ``"prop":`` in Pydantic model fields
    - ``prop =`` as Pydantic Field assignment
    - ``get("prop"`` in dict lookups

    Returns
    -------
    list[str]
        List of file paths where the property is referenced.
    """
    found_in: list[str] = []

    # Patterns that indicate the property is handled
    patterns = [
        # Dict key access: .get("prop" or ["prop"]
        re.compile(rf'["\']({re.escape(prop)})["\']'),
        # Pydantic field: prop: ... = Field(
        re.compile(rf"^\s+{re.escape(prop)}\s*:", re.MULTILINE),
    ]

    for file_path, source in builder_sources.items():
        for pattern in patterns:
            if pattern.search(source):
                found_in.append(file_path)
                break

    return found_in


def _extract_code_properties(builder_sources: dict[str, str]) -> set[str]:
    """Extract **pipeline-level** spec property names that the code processes.

    Only checks ``yaml_builder.py`` and ``pipeline_config.py`` for
    pipeline-level ``spec.get(...)`` access.  Node-level spec access in plugin
    files (e.g. ``body``, ``mode``) is excluded — those are inner node
    properties, not pipeline ``spec`` properties.

    Returns
    -------
    set[str]
        Set of property names referenced in code.
    """
    # Match spec.get("something" or spec["something"]
    spec_get_pattern = re.compile(r'spec\.get\(\s*["\'](\w+)["\']')
    spec_bracket_pattern = re.compile(r'spec\[["\'](\w+)["\']\]')

    # Pydantic model fields in PipelineConfig
    pydantic_field_pattern = re.compile(r"^\s+(\w+)\s*:\s*(?:dict|list|str|int|Any)", re.MULTILINE)
    # Pattern to extract the PipelineConfig class body only
    pipeline_config_class_pattern = re.compile(
        r"^class PipelineConfig\b.*?(?=\nclass |\Z)", re.MULTILINE | re.DOTALL
    )

    # Only scan pipeline-level files (not node plugins)
    pipeline_level_files = {
        "hexdag/compiler/yaml_builder.py",
        "hexdag/kernel/domain/pipeline_config.py",
    }

    properties: set[str] = set()

    for _file_path, source in builder_sources.items():
        if _file_path not in pipeline_level_files:
            continue
        for match in spec_get_pattern.finditer(source):
            properties.add(match.group(1))
        for match in spec_bracket_pattern.finditer(source):
            properties.add(match.group(1))
        # Only look for Pydantic fields within the PipelineConfig class
        if "pipeline_config" in _file_path:
            class_match = pipeline_config_class_pattern.search(source)
            if class_match:
                class_body = class_match.group(0)
                for match in pydantic_field_pattern.finditer(class_body):
                    properties.add(match.group(1))

    return properties


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    """Check YAML schema ↔ builder sync and return exit code.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(description="Check YAML pipeline schema ↔ builder code sync")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if not _SCHEMA_PATH.exists():
        print(f"Error: Schema not found: {_SCHEMA_PATH}")
        return 1

    config = _load_config()
    schema_only_allowed = set(config.get("schema_only_allowed", []) or [])
    code_only_allowed = set(config.get("code_only_allowed", []) or [])

    # Load schema properties
    schema_props = _extract_spec_properties(_SCHEMA_PATH)
    if args.verbose:
        print(f"📋 Schema spec properties ({len(schema_props)}):")
        for name, desc in sorted(schema_props.items()):
            print(f"   {name}: {desc}")

    # Load builder source code
    builder_sources: dict[str, str] = {}
    for path_str in _BUILDER_PATHS:
        path = Path(path_str)
        if path.exists():
            builder_sources[path_str] = path.read_text(encoding="utf-8")
        elif args.verbose:
            print(f"   Skipping non-existent: {path_str}")

    # Check schema → code (properties in schema but not handled by code)
    schema_only: list[tuple[str, str]] = []
    for prop, desc in sorted(schema_props.items()):
        found_in = _scan_code_for_property(prop, builder_sources)
        if not found_in and prop not in schema_only_allowed:
            schema_only.append((prop, desc))
        elif args.verbose:
            files = ", ".join(found_in)
            print(f"   ✓ {prop} → found in: {files}")

    # Load node-level properties from $defs (base Node + concrete node types)
    node_level_props = _extract_node_level_properties(_SCHEMA_PATH)

    # Check code → schema (properties handled by code but not in schema)
    code_props = _extract_code_properties(builder_sources)
    code_only: list[str] = []
    for prop in sorted(code_props):
        if prop not in schema_props and prop not in code_only_allowed:
            # Exclude common non-spec dict accesses and built-in config fields
            if prop in {
                "kind",
                "metadata",
                "spec",
                "apiVersion",
                "nodes",
                "model_config",
                "default_factory",
                "description",
            }:
                continue
            # Exclude node-level properties defined in $defs
            if prop in node_level_props:
                if args.verbose:
                    print(f"   ✓ {prop} → node-level property (in $defs)")
                continue
            code_only.append(prop)
        elif args.verbose and prop in schema_props:
            print(f"   ✓ {prop} → in both schema and code")

    # Report results
    has_issues = bool(schema_only) or bool(code_only)

    if schema_only:
        count = len(schema_only)
        print(f"\n❌ Schema-only properties ({count}) — in schema but not in builder:")
        print("=" * 80)
        for prop, desc in schema_only:
            print(f"   spec.{prop}: {desc}")
        print()
        print("   Users can write schema-valid YAML with these properties,")
        print("   but they will be silently ignored.")

    if code_only:
        count = len(code_only)
        print(f"\n❌ Code-only properties ({count}) — in builder but not in schema:")
        print("=" * 80)
        for prop in code_only:
            print(f"   spec.{prop}")
        print()
        print("   Builder processes these but they have no schema definition.")
        print("   Users get no IDE autocompletion or validation.")

    if not has_issues:
        n = len(schema_props)
        print(f"✅ YAML schema and builder are synchronized ({n} spec properties)")
        return 0

    print("\n💡 To fix:")
    print("   1. Add handling in builder for schema-only properties")
    print("   2. Add schema entries for code-only properties")
    print("   3. Or add to .check_yaml_schema_sync.yaml allowlist")

    return 1


if __name__ == "__main__":
    sys.exit(main())
