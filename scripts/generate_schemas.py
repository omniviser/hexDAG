#!/usr/bin/env python3
"""Generate JSON Schema files from node factories for IDE autocomplete.

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

import importlib
import json
import pkgutil
import re
import sys
from pathlib import Path
from typing import Any

from hexdag.kernel.schema.generator import SchemaGenerator

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

SCHEMAS_DIR = Path(__file__).parent.parent / "schemas"


def _to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case (e.g. LLMNode -> llm_node)."""
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
    return name.lower()


def _discover_factory_classes() -> dict[str, type[Any]]:
    """Auto-discover all BaseNodeFactory subclasses from hexdag.stdlib.nodes.

    Returns a mapping of snake_case alias to factory class, e.g.:
        {"function_node": FunctionNode, "llm_node": LLMNode, ...}
    """
    from hexdag.stdlib.nodes.base_node_factory import BaseNodeFactory

    factories: dict[str, type[Any]] = {}
    package = importlib.import_module("hexdag.stdlib.nodes")

    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"hexdag.stdlib.nodes.{module_info.name}")
        except ImportError:
            continue

        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseNodeFactory)
                and attr is not BaseNodeFactory
            ):
                snake_name = _to_snake_case(attr_name)
                factories[snake_name] = attr

    # Add agent_node alias for ReActAgentNode (matches YAML convention)
    if "re_act_agent_node" in factories:
        factories["agent_node"] = factories.pop("re_act_agent_node")

    return factories


# Auto-discovered node factories — no manual registration needed
NODE_FACTORIES: dict[str, type[Any]] = _discover_factory_classes()


def generate_pipeline_schema() -> dict[str, Any]:
    """Generate JSON Schema for hexDAG pipeline YAML files.

    This schema includes all node types with their complete parameter schemas
    for IDE autocomplete.

    Returns
    -------
    dict[str, Any]
        Complete JSON Schema for pipeline YAML files
    """
    # Base schema structure
    schema: dict[str, Any] = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "hexDAG Pipeline",
        "description": "(auto-generated from builtin nodes)",
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
                        "description": "Port adapters configuration (LLM, database, memory)",
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

    for node_name, factory_class in NODE_FACTORIES.items():
        # Instantiate factory to get the __call__ method for schema generation
        try:
            factory_instance = factory_class()
        except Exception as e:
            print(
                f"Warning: Could not instantiate {node_name}: {e}",
                file=sys.stderr,
            )
            continue

        # Generate schema for this node type
        try:
            node_schema = SchemaGenerator.from_callable(factory_instance, format="dict")
        except Exception as e:
            print(
                f"Warning: Could not generate schema for {node_name}: {e}",
                file=sys.stderr,
            )
            continue

        # Build the node spec structure
        node_def = {
            "type": "object",
            "description": f"Specification for {node_name} type",
            "properties": {
                "kind": {
                    "const": node_name,
                    "description": f"Node type: {node_name}",
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


def generate_hexdag_config_schema() -> dict[str, Any]:
    """Generate JSON Schema for [tool.hexdag] configuration in pyproject.toml.

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

    # Fallback: Return minimal schema structure
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "hexDAG Configuration",
        "description": "Configuration for hexDAG in pyproject.toml",
        "type": "object",
    }


def save_schema(schema: dict[str, Any], filename: str, format: str = "json") -> None:
    """Save schema to file with pretty formatting.

    Parameters
    ----------
    schema : dict[str, Any]
        Schema dictionary to save
    filename : str
        Output filename (relative to schemas/ directory)
    format : str
        Output format: "json" or "yaml"
    """
    import io

    import yaml

    output_path = SCHEMAS_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate content to a string buffer first
    buffer = io.StringIO()
    if format == "yaml":
        yaml.dump(schema, buffer, sort_keys=False, default_flow_style=False)
    else:
        json.dump(schema, buffer, indent=4, sort_keys=True)

    content = buffer.getvalue()

    # Ensure exactly one trailing newline (for end-of-file-fixer compatibility)
    content = content.rstrip("\n") + "\n"

    output_path.write_text(content)
    print(f"✓ Generated {output_path.relative_to(Path.cwd())}")


def main() -> int:
    """Generate all schema files.

    Returns
    -------
    int
        Exit code (0 = success, 1 = error)
    """
    print("Generating schema files (JSON + YAML)...")
    print()

    # Generate pipeline schema
    try:
        pipeline_schema = generate_pipeline_schema()
        save_schema(pipeline_schema, "pipeline-schema.json", format="json")
        save_schema(pipeline_schema, "pipeline-schema.yaml", format="yaml")

        # Count node types
        node_count = len(pipeline_schema.get("$defs", {})) - 2  # -2 for Node and EventHandler
        print(f"  → Included {node_count} node types")
    except Exception as e:
        print(f"Error generating pipeline schema: {e}", file=sys.stderr)
        return 1

    # Generate hexDAG config schema
    try:
        config_schema = generate_hexdag_config_schema()
        save_schema(config_schema, "hexdag-config-schema.json", format="json")
        print("  → Generated from existing config schema")
    except Exception as e:
        print(f"Error generating config schema: {e}", file=sys.stderr)
        return 1

    print()
    print("✓ All schemas generated successfully (JSON + YAML)!")
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
