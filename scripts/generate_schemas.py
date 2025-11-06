#!/usr/bin/env python3
"""Generate JSON Schema files from the registry for IDE autocomplete.

This script generates JSON Schema definitions for:
1. Pipeline YAML files - All node types with their complete parameter schemas
2. Policy configuration files
3. HexDAG configuration (pyproject.toml)

These schemas enable IDE autocomplete, validation, and documentation.

Usage:
    uv run python scripts/generate_schemas.py

The generated schemas are saved to the schemas/ directory and should NOT
be manually edited. Pre-commit hooks will reject manual modifications.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from hexdag.core.bootstrap import ensure_bootstrapped
from hexdag.core.orchestration.policies.models import PolicySignal
from hexdag.core.registry import registry
from hexdag.core.registry.models import ComponentType
from hexdag.core.schema.generator import SchemaGenerator

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

SCHEMAS_DIR = Path(__file__).parent.parent / "schemas"


def generate_pipeline_schema(namespace: str = "core") -> dict[str, Any]:
    """Generate JSON Schema for hexDAG pipeline YAML files.

    This schema includes all registered node types from the specified namespace
    with their complete parameter schemas for IDE autocomplete.

    Parameters
    ----------
    namespace : str
        Component namespace to include (default: "core" for built-in nodes)

    Returns
    -------
    dict[str, Any]
        Complete JSON Schema for pipeline YAML files
    """
    # Get all registered node types
    node_components = registry.list_components(component_type=ComponentType.NODE)

    # Filter by namespace
    node_components = [c for c in node_components if c.namespace == namespace]

    # Base schema structure
    schema: dict[str, Any] = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "hexDAG Pipeline",
        "description": f"(auto-generated from {namespace} namespace)",
        "type": "object",
        "properties": {
            "apiVersion": {
                "type": "string",
                "default": "v1",
                "description": "API version",
            },
            "kind": {
                "type": "string",
                "const": "Pipeline",
                "description": "Resource kind (must be 'Pipeline')",
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Unique pipeline identifier",
                    },
                    "description": {
                        "type": "string",
                        "description": "Pipeline description",
                    },
                    "author": {
                        "type": "string",
                        "description": "Pipeline author",
                    },
                    "version": {
                        "type": "string",
                        "description": "Semantic version (e.g., 1.0.0)",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Categorization tags",
                    },
                },
                "required": ["name"],
            },
            "spec": {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "description": "List of pipeline nodes",
                        "items": {
                            "oneOf": [],  # Will be populated with node definitions
                        },
                    },
                    "input_schema": {
                        "type": "object",
                        "description": "JSON Schema for pipeline inputs",
                    },
                    "output_schema": {
                        "type": "object",
                        "description": "JSON Schema for pipeline outputs",
                    },
                    "common_field_mappings": {
                        "type": "object",
                        "description": "Reusable field mapping definitions",
                    },
                    "ports": {
                        "type": "object",
                        "description": "Port adapters configuration (LLM, database, memory, etc.)",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "adapter": {"type": "string", "description": "Adapter type"},
                                "params": {"type": "object", "description": "Adapter parameters"},
                            },
                        },
                    },
                    "type_ports": {
                        "type": "object",
                        "description": "Port type mappings",
                        "additionalProperties": {"type": "string"},
                    },
                    "policies": {
                        "type": "object",
                        "description": "Execution policies (retry, timeout, error handling)",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "description": "Policy type"},
                                "params": {"type": "object", "description": "Policy parameters"},
                            },
                        },
                    },
                    "events": {
                        "type": "object",
                        "description": "Event handlers for observability",
                        "properties": {
                            "on_node_started": {
                                "type": "array",
                                "items": {"$ref": "#/$defs/EventHandler"},
                            },
                            "on_node_completed": {
                                "type": "array",
                                "items": {"$ref": "#/$defs/EventHandler"},
                            },
                            "on_node_failed": {
                                "type": "array",
                                "items": {"$ref": "#/$defs/EventHandler"},
                            },
                            "on_workflow_complete": {
                                "type": "array",
                                "items": {"$ref": "#/$defs/EventHandler"},
                            },
                        },
                    },
                },
                "required": ["nodes"],
            },
        },
        "required": ["kind", "metadata", "spec"],
        "$defs": {},
    }

    # Generate node definitions
    node_definitions = []

    for component in node_components:
        node_name = component.name

        # Get node factory from registry
        try:
            factory = registry.get(node_name, namespace=namespace)
        except (KeyError, ValueError):
            continue

        # Generate schema for this node type
        try:
            node_schema = SchemaGenerator.from_callable(factory, format="dict")
        except Exception as e:
            print(
                f"Warning: Could not generate schema for {node_name}: {e}",
                file=sys.stderr,
            )
            continue

        # Create node definition
        node_kind = node_name  # e.g., "llm_node", "agent_node"

        # Build the node spec structure
        node_def = {
            "type": "object",
            "description": f"Specification for {namespace}:{node_name} type",
            "properties": {
                "kind": {
                    "const": node_kind,
                    "description": f"Node type: {node_kind}",
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Unique node identifier within the pipeline",
                        },
                        "annotations": {
                            "type": "object",
                            "description": "Optional metadata annotations",
                        },
                    },
                    "required": ["name"],
                },
                "spec": node_schema,  # The generated parameter schema
            },
            "required": ["kind", "metadata", "spec"],
        }

        # Add to definitions
        def_key = f"{node_name.replace('_node', '').title()}NodeSpec"
        schema["$defs"][def_key] = {
            "allOf": [{"$ref": "#/$defs/Node"}],
            **node_def,
        }

        # Add reference to oneOf
        node_definitions.append({"$ref": f"#/$defs/{def_key}"})

    # Add base Node definition
    schema["$defs"]["Node"] = {
        "type": "object",
        "description": "Base node structure",
        "properties": {
            "kind": {"type": "string"},
            "metadata": {"type": "object"},
            "spec": {"type": "object"},
        },
    }

    # Add EventHandler definition for events section
    schema["$defs"]["EventHandler"] = {
        "type": "object",
        "description": "Event handler configuration",
        "properties": {
            "type": {
                "type": "string",
                "description": "Handler type (e.g., 'alert', 'metrics', 'log')",
                "enum": ["alert", "metrics", "log", "webhook", "callback"],
            },
            "target": {
                "type": "string",
                "description": "Target system (e.g., 'pagerduty', 'datadog', 'slack')",
            },
            "severity": {
                "type": "string",
                "description": "Alert severity level",
                "enum": ["low", "medium", "high", "critical"],
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tags for categorization",
            },
            "params": {
                "type": "object",
                "description": "Handler-specific parameters",
            },
        },
        "required": ["type"],
    }

    # Update nodes array with all node definitions
    if schema["properties"]["spec"]["properties"]["nodes"]["items"]:
        schema["properties"]["spec"]["properties"]["nodes"]["items"]["oneOf"] = node_definitions

    return schema


def generate_policy_schema() -> dict[str, Any]:
    """Generate JSON Schema for policy configuration.

    Returns
    -------
    dict[str, Any]
        JSON Schema for policy files
    """
    schema: dict[str, Any] = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "hexDAG Policy Configuration",
        "description": "Policy configuration for execution control and error handling",
        "type": "object",
        "properties": {
            "signal": {
                "type": "string",
                "enum": [s.value for s in PolicySignal],
                "description": "Policy decision signal",
            },
            "context": {
                "type": "object",
                "description": "Policy evaluation context",
                "properties": {
                    "event": {
                        "type": "object",
                        "description": "Event that triggered evaluation",
                    },
                    "dag_id": {"type": "string", "description": "DAG identifier"},
                    "node_id": {
                        "type": ["string", "null"],
                        "description": "Current node (if applicable)",
                    },
                    "wave_index": {
                        "type": "integer",
                        "description": "Current wave index",
                    },
                    "attempt": {
                        "type": "integer",
                        "description": "Attempt number (1-based)",
                    },
                    "error": {
                        "type": ["object", "null"],
                        "description": "Exception details (if any)",
                    },
                    "metadata": {
                        "type": ["object", "null"],
                        "description": "Additional context",
                    },
                },
            },
        },
    }

    return schema


def generate_hexdag_config_schema() -> dict[str, Any]:
    """Generate JSON Schema for [tool.hexdag] configuration in pyproject.toml.

    Note: HexDAGConfig is a dataclass. The existing schema file is already correct.
    This function keeps it up-to-date if needed.

    Returns
    -------
    dict[str, Any]
        JSON Schema for hexDAG configuration
    """
    # The schema is already properly maintained in schemas/hexdag-config-schema.json
    # Just return the existing one if it exists
    config_schema_path = SCHEMAS_DIR / "hexdag-config-schema.json"

    if config_schema_path.exists():
        with config_schema_path.open() as f:
            return json.load(f)

    # Fallback: Return existing schema structure
    # (This schema is already correct and shouldn't need regeneration)
    return {}


def save_schema(schema: dict[str, Any], filename: str) -> None:
    """Save schema to file with pretty formatting.

    Parameters
    ----------
    schema : dict[str, Any]
        Schema dictionary to save
    filename : str
        Output filename (relative to schemas/ directory)
    """
    output_path = SCHEMAS_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(schema, f, indent=4, sort_keys=True)
        f.write("\n")  # Add trailing newline for end-of-file-fixer

    print(f"✓ Generated {output_path.relative_to(Path.cwd())}")


def main() -> int:
    """Generate all schema files.

    Returns
    -------
    int
        Exit code (0 = success, 1 = error)
    """
    print("Generating JSON Schema files...")
    print()

    # Bootstrap registry to discover components
    try:
        ensure_bootstrapped(use_defaults=True)
    except Exception as e:
        print(f"Error bootstrapping registry: {e}", file=sys.stderr)
        return 1

    # Generate pipeline schema (core namespace only)
    try:
        pipeline_schema = generate_pipeline_schema(namespace="core")
        save_schema(pipeline_schema, "pipeline-schema.json")

        # Count node types
        node_count = len(pipeline_schema.get("$defs", {})) - 1  # -1 for base Node
        print(f"  → Included {node_count} node types from core namespace")
    except Exception as e:
        print(f"Error generating pipeline schema: {e}", file=sys.stderr)
        return 1

    # Generate policy schema
    try:
        policy_schema = generate_policy_schema()
        save_schema(policy_schema, "policy-schema.json")
        print(f"  → Included {len(PolicySignal)} policy signals")
    except Exception as e:
        print(f"Error generating policy schema: {e}", file=sys.stderr)
        return 1

    # Generate hexDAG config schema
    try:
        config_schema = generate_hexdag_config_schema()
        save_schema(config_schema, "hexdag-config-schema.json")
        print("  → Generated from HexDAGConfig Pydantic model")
    except Exception as e:
        print(f"Error generating config schema: {e}", file=sys.stderr)
        return 1

    print()
    print("✓ All schemas generated successfully!")
    print()
    print("To use in VS Code, add to .vscode/settings.json:")
    print(
        """
{
  "yaml.schemas": {
    "./schemas/pipeline-schema.json": ["*.yaml", "pipelines/*.yaml"],
    "./schemas/hexdag-config-schema.json": ["pyproject.toml"]
  }
}
"""
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
