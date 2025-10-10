#!/usr/bin/env python3
"""Generate JSON schemas dynamically from registry and Pydantic models.

This script generates up-to-date JSON schemas for:
1. Pipeline manifests (from node factory schemas in registry)
2. Policy configuration (from policy models)
3. HexDAG configuration (from config models)

Run this script when:
- Adding new node types
- Changing node parameters
- Updating configuration models
- Before releases
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Bootstrap the registry to get all node types
from hexdag.core.bootstrap import ensure_bootstrapped
from hexdag.core.orchestration.policies.models import (
    PolicySignal,
    SubscriberType,
)
from hexdag.core.registry import registry
from hexdag.core.registry.models import ComponentType

SCHEMAS_DIR = Path(__file__).parent.parent / "schemas"


def generate_pipeline_schema(namespace: str = "core") -> dict[str, Any]:
    """Generate pipeline schema dynamically from registry node factories.

    Args:
        namespace: Registry namespace to generate schema for (default: "core")
    """
    # Get all registered node types from specified namespace
    node_components = registry.list_components(component_type=ComponentType.NODE)

    # Filter by namespace and extract node types
    node_types = {
        comp.name.removesuffix("_node")
        for comp in node_components
        if comp.name.endswith("_node") and comp.namespace == namespace
    }

    # Build node spec definitions from registry schemas
    node_spec_defs: dict[str, Any] = {}
    for node_type in sorted(node_types):
        factory_name = f"{node_type}_node"
        try:
            schema = registry.get_schema(factory_name, namespace=namespace, format="dict")
            if schema and isinstance(schema, dict):
                node_spec_defs[f"{node_type.title()}NodeSpec"] = {
                    "description": f"Specification for {namespace}:{node_type}_node type",
                    "allOf": [{"$ref": "#/$defs/Node"}],
                    "properties": {
                        "spec": {
                            "type": "object",
                            "properties": schema.get("properties", {}),
                            "required": schema.get("required", []),
                        }
                    },
                }
        except Exception:
            # Skip nodes without schemas
            continue

    # Build the complete schema
    namespace_note = f" (namespace: {namespace})" if namespace != "core" else ""
    schema_id_suffix = f"-{namespace}" if namespace != "core" else ""
    description = (
        f"JSON Schema for HexDAG declarative pipeline YAML manifests "
        f"(auto-generated from {namespace} namespace)"
    )
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": f"https://hexdag.dev/schemas/pipeline{schema_id_suffix}.json",
        "title": f"HexDAG Pipeline Configuration{namespace_note}",
        "description": description,
        "type": "object",
        "required": ["kind", "metadata", "spec"],
        "properties": {
            "apiVersion": {
                "type": "string",
                "description": "API version for the pipeline manifest",
                "default": "v1",
                "enum": ["v1"],
            },
            "kind": {
                "type": "string",
                "description": "Resource type - must be 'Pipeline'",
                "const": "Pipeline",
            },
            "metadata": {
                "type": "object",
                "description": "Pipeline metadata and descriptive information",
                "required": ["name"],
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Unique identifier for the pipeline",
                        "pattern": "^[a-z0-9]([a-z0-9-]*[a-z0-9])?$",
                        "minLength": 1,
                        "maxLength": 253,
                    },
                    "description": {"type": "string"},
                    "author": {"type": "string"},
                    "version": {
                        "type": "string",
                        "pattern": "^\\d+\\.\\d+(\\.\\d+)?$",
                    },
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "annotations": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                },
            },
            "spec": {
                "type": "object",
                "required": ["nodes"],
                "properties": {
                    "input_schema": {
                        "type": "object",
                        "description": "JSON Schema defining expected pipeline inputs",
                    },
                    "common_field_mappings": {
                        "type": "object",
                        "description": "Reusable field mappings for DRY pipeline definitions",
                        "additionalProperties": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                    "nodes": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/Node"},
                        "minItems": 0,
                    },
                },
            },
        },
        "$defs": {
            "Node": {
                "type": "object",
                "required": ["kind", "metadata", "spec"],
                "properties": {
                    "kind": {
                        "type": "string",
                        "description": "Node type (e.g., 'llm_node', 'plugin:custom_node')",
                        "pattern": "^([a-z0-9-]+:)?[a-z_]+(_node)?$",
                    },
                    "metadata": {
                        "type": "object",
                        "required": ["name"],
                        "properties": {
                            "name": {
                                "type": "string",
                                "pattern": "^[a-z0-9_]+$",
                                "minLength": 1,
                            },
                            "annotations": {
                                "type": "object",
                                "additionalProperties": {"type": "string"},
                            },
                        },
                    },
                    "spec": {
                        "type": "object",
                        "properties": {
                            "dependencies": {
                                "type": "array",
                                "items": {"type": "string"},
                                "default": [],
                            }
                        },
                        "additionalProperties": True,
                    },
                },
            },
            **node_spec_defs,
        },
    }


def generate_policy_schema() -> dict[str, Any]:
    """Generate policy schema from policy models."""
    # Get enum values
    policy_signals = [signal.value for signal in PolicySignal]
    subscriber_types = [sub_type.value for sub_type in SubscriberType]

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "https://hexdag.dev/schemas/policy.json",
        "title": "HexDAG Policy Configuration",
        "description": "JSON Schema for HexDAG policy management (auto-generated)",
        "type": "object",
        "properties": {
            "policies": {
                "type": "array",
                "items": {"$ref": "#/$defs/Policy"},
            }
        },
        "$defs": {
            "Policy": {
                "type": "object",
                "required": ["name", "type", "handler"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "type": {
                        "type": "string",
                        "enum": subscriber_types,
                        "default": "user",
                    },
                    "handler": {"type": "string"},
                    "events": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "node_started",
                                "node_completed",
                                "node_failed",
                                "dag_started",
                                "dag_completed",
                                "dag_failed",
                            ],
                        },
                    },
                    "priority": {"type": "integer", "default": 0},
                },
            },
            "PolicyContext": {
                "type": "object",
                "description": "Context provided to policy handlers",
                "properties": {
                    "event": {"type": "object"},
                    "dag_id": {"type": "string"},
                    "node_id": {"type": ["string", "null"]},
                    "wave_index": {"type": "integer", "default": 0},
                    "attempt": {"type": "integer", "minimum": 1, "default": 1},
                    "error": {"type": ["object", "null"]},
                    "metadata": {
                        "type": ["object", "null"],
                        "additionalProperties": True,
                    },
                },
            },
            "PolicyResponse": {
                "type": "object",
                "required": ["signal"],
                "properties": {
                    "signal": {
                        "type": "string",
                        "enum": policy_signals,
                        "default": "proceed",
                    },
                    "data": {},
                    "metadata": {
                        "type": ["object", "null"],
                        "additionalProperties": True,
                    },
                },
            },
            "PolicySignals": {
                "type": "string",
                "enum": policy_signals,
            },
        },
    }


def generate_hexdag_config_schema() -> dict[str, Any]:
    """Generate HexDAG config schema from config models."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "https://hexdag.dev/schemas/hexdag-config.json",
        "title": "HexDAG Configuration",
        "description": "JSON Schema for [tool.hexdag] in pyproject.toml (auto-generated)",
        "type": "object",
        "properties": {
            "modules": {
                "type": "array",
                "items": {"type": "string"},
                "default": [
                    "hexdag.core.ports",
                    "hexdag.builtin.nodes",
                    "hexdag.builtin.tools.builtin_tools",
                ],
            },
            "plugins": {
                "type": "array",
                "items": {"type": "string"},
                "default": ["hexdag.builtin.adapters.local"],
            },
            "dev_mode": {"type": "boolean", "default": False},
            "logging": {"$ref": "#/$defs/LoggingConfig"},
            "settings": {"type": "object", "additionalProperties": True},
        },
        "$defs": {
            "LoggingConfig": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "string",
                        "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        "default": "INFO",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["console", "json", "structured", "dual", "rich"],
                        "default": "structured",
                    },
                    "output_file": {"type": ["string", "null"], "default": None},
                    "use_color": {"type": "boolean", "default": True},
                    "include_timestamp": {"type": "boolean", "default": True},
                    "use_rich": {"type": "boolean", "default": False},
                    "dual_sink": {"type": "boolean", "default": False},
                    "enable_stdlib_bridge": {"type": "boolean", "default": False},
                    "backtrace": {"type": "boolean", "default": True},
                    "diagnose": {"type": "boolean", "default": True},
                },
            }
        },
    }


def main() -> None:
    """Generate all schemas and write to files."""
    # Ensure schemas directory exists
    SCHEMAS_DIR.mkdir(exist_ok=True)

    # Bootstrap registry to load all node types
    print("Bootstrapping registry...")
    ensure_bootstrapped(use_defaults=True)

    # Generate pipeline schema for 'core' namespace
    # Note: This only includes builtin node types (function, llm, agent, loop, conditional)
    # Plugins can generate their own namespace-specific schemas using:
    #   generate_pipeline_schema(namespace="my-plugin")
    print("Generating pipeline schema (core namespace)...")
    pipeline_schema = generate_pipeline_schema(namespace="core")
    with (SCHEMAS_DIR / "pipeline-schema.json").open("w") as f:
        json.dump(pipeline_schema, f, indent=2)
    print(f"✓ Generated {SCHEMAS_DIR / 'pipeline-schema.json'}")

    # Generate policy schema
    print("Generating policy schema...")
    policy_schema = generate_policy_schema()
    with (SCHEMAS_DIR / "policy-schema.json").open("w") as f:
        json.dump(policy_schema, f, indent=2)
    print(f"✓ Generated {SCHEMAS_DIR / 'policy-schema.json'}")

    # Generate HexDAG config schema
    print("Generating HexDAG config schema...")
    config_schema = generate_hexdag_config_schema()
    with (SCHEMAS_DIR / "hexdag-config-schema.json").open("w") as f:
        json.dump(config_schema, f, indent=2)
    print(f"✓ Generated {SCHEMAS_DIR / 'hexdag-config-schema.json'}")

    print("\n✓ All schemas generated successfully!")


if __name__ == "__main__":
    main()
