"""MCP (Model Context Protocol) server for hexDAG.

Exposes hexDAG functionality as MCP tools for LLM-powered editors like Claude Code and Cursor.
This server enables LLMs to:
- Discover available components dynamically from the registry
- Build YAML pipelines with guided, structured approaches
- Validate pipeline configurations
- Generate pipeline templates

The server is **registry-aware** - all tools automatically reflect registered components,
including user plugins loaded from pyproject.toml or hexdag.toml.

Installation
------------
    uv add "hexdag[mcp]"

Usage
-----
Development mode::

    uv run mcp dev hexdag/mcp_server.py

Install for Claude Desktop/Cursor::

    uv run mcp install hexdag/mcp_server.py --name hexdag

Configuration
-------------
The MCP server loads hexDAG configuration from::

    1. HEXDAG_CONFIG_PATH environment variable (highest priority)
    2. hexdag.toml (project-specific)
    3. pyproject.toml with [tool.hexdag]
    4. .hexdag.toml (hidden config)
    5. Defaults (core + builtin components)

Example Claude Desktop config::

    {
      "mcpServers": {
        "hexdag": {
          "command": "uv",
          "args": ["run", "python", "-m", "hexdag.mcp_server"],
          "env": {
            "HEXDAG_CONFIG_PATH": "/path/to/hexdag.toml"  // Optional
          }
        }
      }
    }
"""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any

import yaml
from mcp.server.fastmcp import FastMCP

from hexdag.core.bootstrap import ensure_bootstrapped
from hexdag.core.pipeline_builder import YamlPipelineBuilder
from hexdag.core.registry import registry
from hexdag.core.registry.models import ComponentType
from hexdag.core.schema import SchemaGenerator

# Bootstrap registry on module load
# Uses HEXDAG_CONFIG_PATH env var if set, otherwise auto-discovers config
config_path = os.getenv("HEXDAG_CONFIG_PATH")
ensure_bootstrapped(config_path=config_path if config_path else None)

# Create MCP server
mcp = FastMCP(
    "hexDAG",
    dependencies=["pydantic>=2.0", "pyyaml>=6.0", "jinja2>=3.1.0"],
)


# ============================================================================
# Component Discovery Tools (Dynamic from Registry)
# ============================================================================


@mcp.tool()
def list_nodes() -> str:
    """List all available node types in the hexDAG registry.

    Returns detailed information about each node type including:
    - Node name and namespace
    - Description
    - Configuration schema

    Returns
    -------
        JSON string with available nodes grouped by namespace

    Examples
    --------
        >>> list_nodes()
        {
          "core": [
            {
              "name": "llm_node",
              "namespace": "core",
              "description": "LLM node with prompt templating",
              "subtype": "llm"
            },
            ...
          ]
        }
    """
    nodes_by_namespace: dict[str, list[dict[str, Any]]] = {}

    for name, metadata in registry._components.items():
        if metadata.component_type != ComponentType.NODE:
            continue

        namespace = metadata.namespace
        if namespace not in nodes_by_namespace:
            nodes_by_namespace[namespace] = []

        node_info = {
            "name": name,
            "namespace": namespace,
            "qualified_name": metadata.qualified_name,
            "description": metadata.description or "No description available",
            "subtype": str(metadata.subtype) if metadata.subtype else None,
        }

        nodes_by_namespace[namespace].append(node_info)

    return json.dumps(nodes_by_namespace, indent=2)


@mcp.tool()
def list_adapters(port_type: str | None = None) -> str:
    """List all available adapters in the hexDAG registry.

    Args
    ----
        port_type: Optional filter by port type (e.g., "llm", "memory", "database", "secret")

    Returns
    -------
        JSON string with available adapters grouped by port type

    Examples
    --------
        >>> list_adapters(port_type="llm")
        {
          "llm": [
            {
              "name": "openai",
              "namespace": "core",
              "port_type": "llm",
              "description": "OpenAI LLM adapter"
            }
          ]
        }
    """
    adapters_by_port: dict[str, list[dict[str, Any]]] = {}

    for name, metadata in registry._components.items():
        if metadata.component_type != ComponentType.ADAPTER:
            continue

        port = metadata.implements_port
        if not port:
            continue

        # Filter by port type if specified
        if port_type and port != port_type:
            continue

        if port not in adapters_by_port:
            adapters_by_port[port] = []

        adapter_info = {
            "name": name,
            "namespace": metadata.namespace,
            "qualified_name": metadata.qualified_name,
            "port_type": port,
            "description": metadata.description or "No description available",
        }

        adapters_by_port[port].append(adapter_info)

    return json.dumps(adapters_by_port, indent=2)


@mcp.tool()
def list_tools(namespace: str | None = None) -> str:
    """List all available tools in the hexDAG registry.

    Args
    ----
        namespace: Optional filter by namespace (e.g., "core", "user", "plugin")

    Returns
    -------
        JSON string with available tools and their schemas

    Examples
    --------
        >>> list_tools(namespace="core")
        {
          "core": [
            {
              "name": "tool_end",
              "namespace": "core",
              "description": "End agent execution"
            }
          ]
        }
    """
    tools_by_namespace: dict[str, list[dict[str, Any]]] = {}

    for name, metadata in registry._components.items():
        if metadata.component_type != ComponentType.TOOL:
            continue

        ns = metadata.namespace

        # Filter by namespace if specified
        if namespace and ns != namespace:
            continue

        if ns not in tools_by_namespace:
            tools_by_namespace[ns] = []

        tool_info = {
            "name": name,
            "namespace": ns,
            "qualified_name": metadata.qualified_name,
            "description": metadata.description or "No description available",
        }

        tools_by_namespace[ns].append(tool_info)

    return json.dumps(tools_by_namespace, indent=2)


@mcp.tool()
def list_macros() -> str:
    """List all available macros in the hexDAG registry.

    Macros are reusable pipeline templates that expand into subgraphs.

    Returns
    -------
        JSON string with available macros and their descriptions

    Examples
    --------
        >>> list_macros()
        [
          {
            "name": "reasoning_agent",
            "namespace": "core",
            "description": "ReAct reasoning agent pattern"
          }
        ]
    """
    macros: list[dict[str, Any]] = []

    for name, metadata in registry._components.items():
        if metadata.component_type != ComponentType.MACRO:
            continue

        macro_info = {
            "name": name,
            "namespace": metadata.namespace,
            "qualified_name": metadata.qualified_name,
            "description": metadata.description or "No description available",
        }
        macros.append(macro_info)

    return json.dumps(macros, indent=2)


@mcp.tool()
def list_policies() -> str:
    """List all available policies in the hexDAG registry.

    Returns
    -------
        JSON string with available policies and their descriptions

    Examples
    --------
        >>> list_policies()
        [
          {
            "name": "retry_policy",
            "namespace": "core",
            "description": "Retry failed operations"
          }
        ]
    """
    policies: list[dict[str, Any]] = []

    for name, metadata in registry._components.items():
        if metadata.component_type != ComponentType.POLICY:
            continue

        policy_info = {
            "name": name,
            "namespace": metadata.namespace,
            "qualified_name": metadata.qualified_name,
            "description": metadata.description or "No description available",
        }
        policies.append(policy_info)

    return json.dumps(policies, indent=2)


@mcp.tool()
def get_component_schema(
    component_type: str,
    name: str,
    namespace: str = "core",
) -> str:
    """Get detailed schema for a specific component.

    Args
    ----
        component_type: Type of component (node, adapter, tool, macro, policy)
        name: Component name
        namespace: Component namespace (default: "core")

    Returns
    -------
        JSON string with component schema including parameters and types

    Examples
    --------
        >>> get_component_schema("node", "llm_node", "core")
        {
          "name": "llm_node",
          "type": "node",
          "schema": {
            "prompt_template": {"type": "str", "required": true},
            "output_key": {"type": "str", "default": "result"}
          }
        }
    """
    try:
        # Get component from registry
        component_obj = registry.get(name, namespace=namespace)

        # Extract schema using SchemaGenerator
        schema = SchemaGenerator.extract_schema(component_obj)

        result = {
            "name": name,
            "namespace": namespace,
            "type": component_type,
            "schema": schema,
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps(
            {
                "error": str(e),
                "component_type": component_type,
                "name": name,
                "namespace": namespace,
            },
            indent=2,
        )


# ============================================================================
# Helper Functions
# ============================================================================


def _normalize_for_yaml(obj: Any) -> Any:
    """Recursively convert enum values to strings for YAML serialization.

    This ensures that enums are serialized using their .value attribute
    instead of their name, preventing validation errors when the YAML
    is loaded back.

    Args
    ----
        obj: Object to normalize (can be dict, list, enum, or primitive)

    Returns
    -------
        Normalized object with enums converted to their string values

    Examples
    --------
        >>> from enum import Enum
        >>> class Format(str, Enum):
        ...     MIXED = "mixed"
        >>> _normalize_for_yaml({"format": Format.MIXED})
        {'format': 'mixed'}
    """
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, dict):
        return {k: _normalize_for_yaml(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_normalize_for_yaml(item) for item in obj]
    else:
        return obj


# ============================================================================
# YAML Pipeline Building Tools
# ============================================================================


@mcp.tool()
def validate_yaml_pipeline(yaml_content: str) -> str:
    """Validate a YAML pipeline configuration.

    Args
    ----
        yaml_content: YAML pipeline configuration as a string

    Returns
    -------
        JSON string with validation results (success/errors)

    Examples
    --------
        >>> validate_yaml_pipeline(pipeline_yaml)
        {
          "valid": true,
          "message": "Pipeline is valid",
          "node_count": 3,
          "nodes": ["step1", "step2", "step3"]
        }
    """
    try:
        # Attempt to build the pipeline
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)

        return json.dumps(
            {
                "valid": True,
                "message": "Pipeline is valid",
                "node_count": len(graph.nodes),
                "nodes": [node.metadata.name for node in graph.nodes.values()],
                "ports": list(config.ports.keys()) if config.ports else [],
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps(
            {
                "valid": False,
                "error": str(e),
                "error_type": type(e).__name__,
            },
            indent=2,
        )


@mcp.tool()
def generate_pipeline_template(
    pipeline_name: str,
    description: str,
    node_types: list[str],
) -> str:
    """Generate a YAML pipeline template with specified node types.

    Args
    ----
        pipeline_name: Name for the pipeline
        description: Pipeline description
        node_types: List of node types to include (e.g., ["llm_node", "agent_node"])

    Returns
    -------
        YAML pipeline template as a string

    Examples
    --------
        >>> generate_pipeline_template(
        ...     "my-workflow",
        ...     "Example workflow",
        ...     ["llm_node", "function_node"]
        ... )
        apiVersion: hexdag/v1
        kind: Pipeline
        metadata:
          name: my-workflow
          description: Example workflow
        spec:
          nodes:
            - kind: llm_node
              metadata:
                name: llm_1
              spec:
                prompt_template: "Your prompt here: {{input}}"
                output_key: result
              dependencies: []
    """
    # Create basic pipeline structure
    pipeline = {
        "apiVersion": "hexdag/v1",
        "kind": "Pipeline",
        "metadata": {
            "name": pipeline_name,
            "description": description,
        },
        "spec": {
            "nodes": [],
        },
    }

    # Add node templates
    for i, node_type in enumerate(node_types, 1):
        # Remove '_node' suffix if present for cleaner names
        node_name_base = node_type.replace("_node", "")

        node_template = {
            "kind": node_type,
            "metadata": {
                "name": f"{node_name_base}_{i}",
            },
            "spec": _get_node_spec_template(node_type),
            "dependencies": [] if i == 1 else [f"{node_types[i-2].replace('_node', '')}_{i-1}"],
        }
        pipeline["spec"]["nodes"].append(node_template)

    # Normalize enums before serialization
    pipeline = _normalize_for_yaml(pipeline)
    return yaml.dump(pipeline, sort_keys=False, default_flow_style=False)


def _get_node_spec_template(node_type: str) -> dict[str, Any]:
    """Get a spec template for a given node type.

    Args
    ----
        node_type: Type of node (e.g., "llm_node", "agent_node")

    Returns
    -------
        Dict with common spec fields for the node type
    """
    templates = {
        "llm_node": {
            "prompt_template": "Your prompt here: {{input}}",
            "output_key": "result",
        },
        "agent_node": {
            "initial_prompt_template": "Task: {{task}}",
            "max_steps": 5,
            "output_key": "agent_result",
            "tools": [],
        },
        "function_node": {
            "fn": "your_module.your_function",
            "input_schema": {"param": "str"},
            "output_schema": {"result": "str"},
        },
        "conditional_node": {
            "condition": "{{input}} > 0",
            "true_path": [],
            "false_path": [],
        },
        "loop_node": {
            "loop_variable": "item",
            "items": "{{input_list}}",
            "body": [],
        },
    }

    return templates.get(node_type, {})


@mcp.tool()
def build_yaml_pipeline_interactive(
    pipeline_name: str,
    description: str,
    nodes: list[dict[str, Any]],
    ports: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Build a complete YAML pipeline with full specifications.

    This is the recommended tool for building complete pipelines with LLM assistance.

    Args
    ----
        pipeline_name: Name for the pipeline
        description: Pipeline description
        nodes: List of node specifications with full config
        ports: Optional port configurations (llm, memory, database, etc.)

    Returns
    -------
        Complete YAML pipeline configuration

    Examples
    --------
        >>> build_yaml_pipeline_interactive(
        ...     "analysis-pipeline",
        ...     "Analyze documents",
        ...     nodes=[
        ...         {
        ...             "kind": "llm_node",
        ...             "name": "analyzer",
        ...             "spec": {"prompt_template": "Analyze: {{input}}"},
        ...             "dependencies": []
        ...         }
        ...     ],
        ...     ports={
        ...         "llm": {
        ...             "adapter": "openai",
        ...             "config": {"api_key": "${OPENAI_API_KEY}", "model": "gpt-4"}
        ...         }
        ...     }
        ... )
        apiVersion: hexdag/v1
        kind: Pipeline
        metadata:
          name: analysis-pipeline
          description: Analyze documents
        spec:
          ports:
            llm:
              adapter: openai
              config:
                api_key: ${OPENAI_API_KEY}
                model: gpt-4
          nodes:
            - kind: llm_node
              metadata:
                name: analyzer
              spec:
                prompt_template: "Analyze: {{input}}"
              dependencies: []
    """
    pipeline = {
        "apiVersion": "hexdag/v1",
        "kind": "Pipeline",
        "metadata": {
            "name": pipeline_name,
            "description": description,
        },
        "spec": {
            "nodes": [],
        },
    }

    # Add ports if provided
    if ports:
        pipeline["spec"]["ports"] = ports

    # Add nodes
    for node_def in nodes:
        node = {
            "kind": node_def["kind"],
            "metadata": {
                "name": node_def["name"],
            },
            "spec": node_def["spec"],
            "dependencies": node_def.get("dependencies", []),
        }
        pipeline["spec"]["nodes"].append(node)

    # Normalize enums before serialization
    pipeline = _normalize_for_yaml(pipeline)
    return yaml.dump(pipeline, sort_keys=False, default_flow_style=False)


@mcp.tool()
def create_environment_pipelines(
    pipeline_name: str,
    description: str,
    nodes: list[dict[str, Any]],
    dev_ports: dict[str, dict[str, Any]] | None = None,
    staging_ports: dict[str, dict[str, Any]] | None = None,
    prod_ports: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Create dev, staging, and production YAML pipelines in one call.

    This generates up to 3 YAML files with environment-specific port configurations:
    - dev: Uses mock adapters (no API keys)
    - staging: Uses real APIs with staging credentials
    - prod: Uses real APIs with production credentials

    Args
    ----
        pipeline_name: Base name for pipelines (will be suffixed with -dev, -staging, -prod)
        description: Pipeline description
        nodes: Node specifications (shared across all environments)
        dev_ports: Port config for dev (defaults to mock adapters if not provided)
        staging_ports: Port config for staging (optional)
        prod_ports: Port config for production (optional)

    Returns
    -------
        JSON string with separate YAML content for each environment

    Examples
    --------
        >>> create_environment_pipelines(
        ...     "research-agent",
        ...     "Research agent",
        ...     nodes=[{"kind": "macro_invocation", ...}],
        ...     prod_ports={"llm": {"adapter": "core:openai", ...}}
        ... )
        {
          "dev": "apiVersion: hexdag/v1\\nkind: Pipeline...",
          "staging": "apiVersion: hexdag/v1\\nkind: Pipeline...",
          "prod": "apiVersion: hexdag/v1\\nkind: Pipeline..."
        }
    """
    ensure_bootstrapped()

    result = {}

    # Default dev ports to mock adapters
    if dev_ports is None:
        dev_ports = {
            "llm": {
                "adapter": "plugin:mock_llm",
                "config": {
                    "responses": [
                        "I'll search for information. INVOKE_TOOL: search(query='...')",
                        "Let me gather more details. INVOKE_TOOL: search(query='...')",
                        "Based on research, here are my findings: [Mock comprehensive answer]",
                    ],
                    "delay_seconds": 0.1,
                },
            },
            "tool_router": {
                "adapter": "plugin:mock_tool_router",
                "config": {"available_tools": ["search", "calculate"]},
            },
        }

    # Build dev environment
    pipeline_dev = {
        "apiVersion": "hexdag/v1",
        "kind": "Pipeline",
        "metadata": {
            "name": f"{pipeline_name}-dev",
            "description": f"{description} (DEV - Mock Adapters)",
        },
        "spec": {"ports": dev_ports, "nodes": []},
    }

    for node_def in nodes:
        node = {
            "kind": node_def["kind"],
            "metadata": {"name": node_def["name"]},
            "spec": node_def["spec"],
            "dependencies": node_def.get("dependencies", []),
        }
        pipeline_dev["spec"]["nodes"].append(node)

    pipeline_dev = _normalize_for_yaml(pipeline_dev)
    result["dev"] = yaml.dump(pipeline_dev, sort_keys=False, default_flow_style=False)

    # Build staging environment (if provided)
    if staging_ports:
        pipeline_staging = {
            "apiVersion": "hexdag/v1",
            "kind": "Pipeline",
            "metadata": {
                "name": f"{pipeline_name}-staging",
                "description": f"{description} (STAGING)",
            },
            "spec": {"ports": staging_ports, "nodes": []},
        }

        for node_def in nodes:
            node = {
                "kind": node_def["kind"],
                "metadata": {"name": node_def["name"]},
                "spec": node_def["spec"],
                "dependencies": node_def.get("dependencies", []),
            }
            pipeline_staging["spec"]["nodes"].append(node)

        pipeline_staging = _normalize_for_yaml(pipeline_staging)
        result["staging"] = yaml.dump(pipeline_staging, sort_keys=False, default_flow_style=False)

    # Build production environment (if provided)
    if prod_ports:
        pipeline_prod = {
            "apiVersion": "hexdag/v1",
            "kind": "Pipeline",
            "metadata": {
                "name": f"{pipeline_name}-prod",
                "description": f"{description} (PRODUCTION)",
            },
            "spec": {"ports": prod_ports, "nodes": []},
        }

        for node_def in nodes:
            node = {
                "kind": node_def["kind"],
                "metadata": {"name": node_def["name"]},
                "spec": node_def["spec"],
                "dependencies": node_def.get("dependencies", []),
            }
            pipeline_prod["spec"]["nodes"].append(node)

        pipeline_prod = _normalize_for_yaml(pipeline_prod)
        result["prod"] = yaml.dump(pipeline_prod, sort_keys=False, default_flow_style=False)

    return json.dumps(result, indent=2)


@mcp.tool()
def create_environment_pipelines_with_includes(
    pipeline_name: str,
    description: str,
    nodes: list[dict[str, Any]],
    dev_ports: dict[str, dict[str, Any]] | None = None,
    staging_ports: dict[str, dict[str, Any]] | None = None,
    prod_ports: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Create base + environment-specific YAML files using the include pattern.

    This generates a base YAML with shared nodes and separate environment configs
    that include the base file. This approach reduces duplication and makes it
    easier to maintain consistent logic across environments.

    Generated files:
    - base.yaml: Shared node definitions
    - dev.yaml: Includes base + dev ports (mock adapters)
    - staging.yaml: Includes base + staging ports (optional)
    - prod.yaml: Includes base + prod ports (optional)

    Args
    ----
        pipeline_name: Base name for pipelines
        description: Pipeline description
        nodes: Node specifications (shared across all environments)
        dev_ports: Port config for dev (defaults to mock adapters if not provided)
        staging_ports: Port config for staging (optional)
        prod_ports: Port config for production (optional)

    Returns
    -------
        JSON string with base YAML and environment-specific includes

    Examples
    --------
        >>> create_environment_pipelines_with_includes(
        ...     "research-agent",
        ...     "Research agent",
        ...     nodes=[{"kind": "macro_invocation", ...}],
        ...     prod_ports={"llm": {"adapter": "core:openai", ...}}
        ... )
        {
          "base": "apiVersion: hexdag/v1\\n...",
          "dev": "include: ./research_agent_base.yaml\\nports:\\n  llm:\\n    adapter: plugin:mock_llm",
          "prod": "include: ./research_agent_base.yaml\\nports:\\n  llm:\\n    adapter: core:openai"
        }
    """
    ensure_bootstrapped()

    result = {}

    # Default dev ports to mock adapters
    if dev_ports is None:
        dev_ports = {
            "llm": {
                "adapter": "plugin:mock_llm",
                "config": {
                    "responses": [
                        "I'll search for information. INVOKE_TOOL: search(query='...')",
                        "Let me gather more details. INVOKE_TOOL: search(query='...')",
                        "Based on research, here are my findings: [Mock comprehensive answer]",
                    ],
                    "delay_seconds": 0.1,
                },
            },
            "tool_router": {
                "adapter": "plugin:mock_tool_router",
                "config": {"available_tools": ["search", "calculate"]},
            },
        }

    # Build base YAML (nodes only)
    base_pipeline = {
        "apiVersion": "hexdag/v1",
        "kind": "Pipeline",
        "metadata": {
            "name": f"{pipeline_name}-base",
            "description": f"{description} (Base Configuration)",
        },
        "spec": {"nodes": []},
    }

    for node_def in nodes:
        node = {
            "kind": node_def["kind"],
            "metadata": {"name": node_def["name"]},
            "spec": node_def["spec"],
            "dependencies": node_def.get("dependencies", []),
        }
        base_pipeline["spec"]["nodes"].append(node)

    base_pipeline = _normalize_for_yaml(base_pipeline)
    result["base"] = yaml.dump(base_pipeline, sort_keys=False, default_flow_style=False)

    # Build dev environment (includes base + dev ports)
    dev_env = {
        "include": f"./{pipeline_name}_base.yaml",
        "metadata": {
            "name": f"{pipeline_name}-dev",
            "description": f"{description} (DEV - Mock Adapters)",
        },
        "ports": dev_ports,
    }
    dev_env = _normalize_for_yaml(dev_env)
    result["dev"] = yaml.dump(dev_env, sort_keys=False, default_flow_style=False)

    # Build staging environment (if provided)
    if staging_ports:
        staging_env = {
            "include": f"./{pipeline_name}_base.yaml",
            "metadata": {
                "name": f"{pipeline_name}-staging",
                "description": f"{description} (STAGING)",
            },
            "ports": staging_ports,
        }
        staging_env = _normalize_for_yaml(staging_env)
        result["staging"] = yaml.dump(staging_env, sort_keys=False, default_flow_style=False)

    # Build production environment (if provided)
    if prod_ports:
        prod_env = {
            "include": f"./{pipeline_name}_base.yaml",
            "metadata": {
                "name": f"{pipeline_name}-prod",
                "description": f"{description} (PRODUCTION)",
            },
            "ports": prod_ports,
        }
        prod_env = _normalize_for_yaml(prod_env)
        result["prod"] = yaml.dump(prod_env, sort_keys=False, default_flow_style=False)

    return json.dumps(result, indent=2)


@mcp.tool()
def explain_yaml_structure() -> str:
    """Explain the structure of hexDAG YAML pipelines.

    Returns
    -------
        Detailed explanation of YAML pipeline structure with examples
    """
    explanation = """
# hexDAG YAML Pipeline Structure

## Basic Structure

```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: pipeline-name
  description: What this pipeline does
spec:
  ports:       # Optional: Configure adapters
    llm:
      adapter: openai
      config:
        api_key: ${OPENAI_API_KEY}
        model: gpt-4
  nodes:       # Required: Pipeline nodes
    - kind: llm_node
      metadata:
        name: node_name
      spec:
        prompt_template: "Your prompt: {{input}}"
      dependencies: []
```

## Key Concepts

1. **apiVersion**: Always "hexdag/v1"
2. **kind**: Always "Pipeline" (or "Macro" for macro definitions)
3. **metadata**: Pipeline name and description
4. **spec**: Pipeline specification
   - **ports**: Adapter configurations (llm, memory, database, secret)
   - **nodes**: List of processing nodes

## Node Structure

Each node has:
- **kind**: Node type (llm_node, agent_node, function_node, etc.)
- **metadata.name**: Unique identifier for the node
- **spec**: Node-specific configuration
- **dependencies**: List of node names this node depends on

## Available Node Types

- **llm_node**: LLM interactions with prompt templates
- **agent_node**: ReAct agents with tool access
- **function_node**: Execute Python functions
- **conditional_node**: Conditional execution paths
- **loop_node**: Iterative processing

## Templating

Use Jinja2 syntax for dynamic values:
- `{{variable}}` - Reference node outputs or inputs
- `{{node_name.output_key}}` - Reference specific node outputs
- `${ENV_VAR}` - Environment variables (resolved at build or runtime)

## Port Configuration

```yaml
ports:
  llm:
    adapter: openai
    config:
      api_key: ${OPENAI_API_KEY}
      model: gpt-4
  memory:
    adapter: in_memory
  database:
    adapter: sqlite
    config:
      database_path: ./data.db
```

## Dependencies

Dependencies define execution order:
```yaml
nodes:
  - metadata:
      name: step1
    dependencies: []  # Runs first

  - metadata:
      name: step2
    dependencies: [step1]  # Runs after step1

  - metadata:
      name: step3
    dependencies: [step1, step2]  # Runs after both
```

## Secret Handling

Secret-like environment variables are deferred to runtime:
- `*_API_KEY`, `*_SECRET`, `*_TOKEN`, `*_PASSWORD`
- Allows building pipelines without secrets present
- Runtime injection via SecretPort → Memory

## Best Practices

1. Use descriptive node names
2. Add comprehensive descriptions
3. Leverage environment variables for secrets
4. Keep pipelines modular and reusable
5. Validate before execution using validate_yaml_pipeline()
6. Use macro_invocation for reusable patterns
"""
    return explanation


# Run the server
if __name__ == "__main__":
    mcp.run()
