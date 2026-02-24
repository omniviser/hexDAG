"""MCP (Model Context Protocol) server for hexDAG.

Exposes hexDAG functionality as MCP tools for LLM-powered editors like Claude Code and Cursor.
This server uses the unified hexdag.api layer for all business logic.

Installation
------------
    uv add "hexdag[mcp]"

Usage
-----
Development mode::

    uv run mcp dev hexdag/mcp_server.py

Install for Claude Desktop/Cursor::

    uv run mcp install hexdag/mcp_server.py --name hexdag

Example Claude Desktop config::

    {
      "mcpServers": {
        "hexdag": {
          "command": "uv",
          "args": ["run", "python", "-m", "hexdag.mcp_server"],
          "env": {
            "HEXDAG_PLUGIN_PATHS": "/path/to/custom/adapters:/path/to/custom/nodes"
          }
        }
      }
    }

Custom Plugin Paths
-------------------
Set HEXDAG_PLUGIN_PATHS environment variable to discover custom adapters/nodes.
Multiple paths are separated by the OS path separator (`:` on Unix, `;` on Windows).

Example::

    export HEXDAG_PLUGIN_PATHS="./my_adapters:./my_nodes"
    uv run mcp dev hexdag/mcp_server.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import orjson
from mcp.server.fastmcp import FastMCP

from hexdag import api

# Configure user plugin paths from environment variable
# This allows MCP users to discover custom adapters/nodes
if plugin_paths := os.environ.get("HEXDAG_PLUGIN_PATHS"):
    from hexdag.kernel import set_user_plugin_paths

    paths = [Path(p) for p in plugin_paths.split(os.pathsep) if p]
    set_user_plugin_paths(paths)

# Create MCP server
mcp = FastMCP(
    "hexDAG",
    dependencies=["pydantic>=2.0", "pyyaml>=6.0", "jinja2>=3.1.0"],
)


# ============================================================================
# VFS Tools — unified path-based introspection
# ============================================================================

# Create a VFS instance for component and process introspection.
# /lib/ is always available; /proc/* providers need lib instances
# which are not wired here (they'll be added when the MCP server
# supports process management).
_vfs = api.vfs.create_vfs()


@mcp.tool()  # type: ignore[misc]
async def vfs_read(path: str) -> str:
    """Read content at a VFS path.

    Use this to inspect individual components, pipeline runs, scheduled
    tasks, or entity states.

    Paths
    -----
    - ``/lib/nodes/llm_node`` — JSON detail for a specific node
    - ``/lib/nodes/llm_node/schema`` — JSON schema for a node
    - ``/lib/adapters/OpenAIAdapter`` — adapter detail
    - ``/lib/tools/tool_end`` — tool detail
    - ``/lib/macros/ReasoningAgentMacro`` — macro detail
    - ``/lib/tags/!py`` — tag detail
    - ``/proc/runs/<run_id>`` — pipeline run detail
    - ``/proc/scheduled/<task_id>`` — scheduled task detail
    - ``/proc/entities/<type>/<id>`` — entity state

    Parameters
    ----------
    path : str
        Absolute VFS path (e.g. ``/lib/nodes/llm_node``)

    Returns
    -------
        JSON content at the path
    """
    return await api.vfs.read_path(_vfs, path)


@mcp.tool()  # type: ignore[misc]
async def vfs_list(path: str) -> str:
    """List entries in a VFS directory.

    Use this to discover available components, runs, or entities.

    Paths
    -----
    - ``/lib/`` — list entity types (nodes, adapters, macros, tools, tags)
    - ``/lib/nodes/`` — list all available nodes
    - ``/lib/adapters/`` — list all adapters
    - ``/lib/tools/`` — list all tools
    - ``/lib/macros/`` — list all macros
    - ``/lib/tags/`` — list all YAML tags
    - ``/proc/runs/`` — list pipeline runs
    - ``/proc/scheduled/`` — list scheduled tasks
    - ``/proc/entities/`` — list entity types

    Parameters
    ----------
    path : str
        Absolute VFS directory path (e.g. ``/lib/nodes/``)

    Returns
    -------
        JSON array of entries with name, entry_type, and path
    """
    entries = await api.vfs.list_path(_vfs, path)
    return orjson.dumps(entries, option=orjson.OPT_INDENT_2).decode()


@mcp.tool()  # type: ignore[misc]
async def vfs_stat(path: str) -> str:
    """Get metadata about a VFS path.

    Returns structured metadata including description, entity type,
    module path, capabilities, and tags (e.g. is_builtin, port_type).

    Parameters
    ----------
    path : str
        Absolute VFS path (e.g. ``/lib/adapters/OpenAIAdapter``)

    Returns
    -------
        JSON metadata dict
    """
    result = await api.vfs.stat_path(_vfs, path)
    return orjson.dumps(result, option=orjson.OPT_INDENT_2).decode()


# ============================================================================
# Documentation Tools
# ============================================================================


@mcp.tool()  # type: ignore[misc]
def get_syntax_reference() -> str:
    """Get reference for hexDAG YAML syntax including variable references.

    Returns comprehensive documentation on:
    - $input.field - Reference initial pipeline input
    - {{node.output}} - Jinja2 template for node outputs
    - ${ENV_VAR} - Environment variables
    - input_mapping syntax and usage

    Returns
    -------
        Detailed syntax reference documentation
    """
    return api.documentation.get_syntax_reference()


@mcp.tool()  # type: ignore[misc]
def explain_yaml_structure() -> str:
    """Explain the structure of hexDAG YAML pipelines.

    Returns
    -------
        YAML structure documentation
    """
    return api.documentation.explain_yaml_structure()


@mcp.tool()  # type: ignore[misc]
def get_type_reference() -> str:
    """Get reference for hexDAG YAML output_schema type system.

    Returns documentation on:
    - Supported types: str, int, float, bool, list, dict, Any
    - Nullable types with ? suffix: str?, int?, float?, bool?, list?, dict?
    - When to use nullable vs required types
    - Examples for LLM output parsing

    Returns
    -------
        Type system reference documentation
    """
    return api.documentation.get_type_reference()


@mcp.tool()  # type: ignore[misc]
def get_custom_adapter_guide() -> str:
    """Get a comprehensive guide for creating custom adapters.

    Returns documentation on:
    - Creating adapters with the @adapter decorator
    - Secret handling with the secrets parameter
    - Using custom adapters in YAML pipelines
    - Testing patterns for adapters

    Returns
    -------
        Detailed guide for creating custom adapters
    """
    return api.documentation.get_custom_adapter_guide()


@mcp.tool()  # type: ignore[misc]
def get_custom_node_guide() -> str:
    """Get a comprehensive guide for creating custom nodes.

    Returns documentation on:
    - Creating nodes with the @node decorator
    - Node factory pattern
    - Input/output schemas
    - Using custom nodes in YAML pipelines

    Returns
    -------
        Detailed guide for creating custom nodes
    """
    return api.documentation.get_custom_node_guide()


@mcp.tool()  # type: ignore[misc]
def get_custom_tool_guide() -> str:
    """Get a guide for creating custom tools for agents.

    Returns documentation on:
    - Creating tools as Python functions
    - Tool schemas and descriptions
    - Registering tools with agents

    Returns
    -------
        Detailed guide for creating custom tools
    """
    return api.documentation.get_custom_tool_guide()


@mcp.tool()  # type: ignore[misc]
def get_extension_guide(component_type: str | None = None) -> str:
    """Get a guide for extending hexDAG with custom components.

    Parameters
    ----------
    component_type : str | None
        Optional specific component type: "adapter", "node", "tool"
        If None, returns overview of all extension types.

    Returns
    -------
        Extension guide documentation
    """
    return api.documentation.get_extension_guide(component_type)


# ============================================================================
# Execution Tools
# ============================================================================


@mcp.tool()  # type: ignore[misc]
async def execute_pipeline(
    yaml_content: str,
    inputs: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> str:
    """Execute a YAML pipeline and return results.

    This is useful for testing pipelines during development.
    For production execution, use the hexDAG CLI or programmatic API.

    Parameters
    ----------
    yaml_content : str
        YAML pipeline configuration as a string
    inputs : dict | None
        Initial input values for the pipeline
    environment : dict | None
        Optional environment configuration with port adapters.
        Format: {"ports": {"llm": {"adapter": "mock_llm"}, ...}}
        If None, uses mock adapters for safe testing.
    timeout : float
        Execution timeout in seconds (default: 30.0)

    Returns
    -------
    str
        JSON with execution results:
        {
            "success": bool,
            "nodes": [{"name", "status", "output", "error", "duration_ms"}],
            "final_output": Any,
            "error": str | None,
            "duration_ms": float
        }

    Examples
    --------
    Execute a simple pipeline::

        result = execute_pipeline('''
        apiVersion: hexdag/v1
        kind: Pipeline
        metadata:
          name: test
        spec:
          nodes:
            - kind: data_node
              metadata:
                name: start
              spec:
                value: "hello world"
              dependencies: []
        ''')
    """
    # Parse environment configuration to create ports
    ports = None
    if environment and "ports" in environment:
        ports = api.execution.create_ports_from_config(environment["ports"])

    result = await api.execution.execute(
        yaml_content=yaml_content,
        inputs=inputs or {},
        ports=ports,
        timeout=timeout,
    )
    return orjson.dumps(result, option=orjson.OPT_INDENT_2).decode()


@mcp.tool()  # type: ignore[misc]
def dry_run_pipeline(yaml_content: str, inputs: dict[str, Any] | None = None) -> str:
    """Analyze a pipeline without executing it.

    Returns execution plan, dependency order, and wave structure.
    Useful for understanding how a pipeline will execute.

    Parameters
    ----------
    yaml_content : str
        YAML pipeline configuration as a string
    inputs : dict | None
        Input values (used for analysis, not execution)

    Returns
    -------
    str
        JSON with analysis:
        {
            "valid": bool,
            "execution_order": [str],
            "node_count": int,
            "waves": [[str]],  # Nodes that can run in parallel
            "dependency_map": {node: {dependencies, wave}},
            "error": str | None
        }

    Examples
    --------
    Analyze execution order::

        result = dry_run_pipeline('''
        apiVersion: hexdag/v1
        kind: Pipeline
        metadata:
          name: test
        spec:
          nodes:
            - kind: data_node
              metadata:
                name: a
              spec:
                value: 1
              dependencies: []
            - kind: data_node
              metadata:
                name: b
              spec:
                value: 2
              dependencies: [a]
        ''')
        # Returns: {"execution_order": ["a", "b"], "waves": [["a"], ["b"]], ...}
    """
    result = api.execution.dry_run(yaml_content, inputs)
    return orjson.dumps(result, option=orjson.OPT_INDENT_2).decode()


# ============================================================================
# Validation Tools
# ============================================================================


@mcp.tool()  # type: ignore[misc]
def validate_yaml_pipeline(yaml_content: str) -> str:
    """Validate a YAML pipeline configuration.

    Args
    ----
        yaml_content: YAML pipeline configuration as a string

    Returns
    -------
        JSON string with validation results (success/errors)
    """
    result = api.validation.validate(yaml_content, lenient=False)
    return orjson.dumps(result, option=orjson.OPT_INDENT_2).decode()


@mcp.tool()  # type: ignore[misc]
def validate_yaml_pipeline_lenient(yaml_content: str) -> str:
    """Validate YAML pipeline structure without requiring environment variables.

    Use this for CI/CD validation where secrets aren't available.
    This validates structure only, without instantiating adapters.

    Validates:
    - YAML syntax
    - Node structure and dependencies
    - Port configuration format
    - Manifest format (apiVersion, kind, metadata, spec)

    Does NOT validate:
    - Environment variable values
    - Adapter instantiation
    - Module path resolution

    Args
    ----
        yaml_content: YAML pipeline configuration as a string

    Returns
    -------
        JSON string with validation results (success/errors/warnings)
    """
    result = api.validation.validate(yaml_content, lenient=True)
    return orjson.dumps(result, option=orjson.OPT_INDENT_2).decode()


# ============================================================================
# Pipeline Manipulation Tools
# ============================================================================


@mcp.tool()  # type: ignore[misc]
def init_pipeline(name: str, description: str = "") -> str:
    """Create a new minimal pipeline YAML configuration.

    Creates an empty pipeline structure ready for adding nodes.

    Parameters
    ----------
    name : str
        Pipeline name (used in metadata.name)
    description : str, optional
        Pipeline description

    Returns
    -------
    str
        JSON with {success: bool, yaml_content: str}
    """
    result = api.pipeline.init(name, description)
    return orjson.dumps(result, option=orjson.OPT_INDENT_2).decode()


@mcp.tool()  # type: ignore[misc]
def add_node_to_pipeline(yaml_content: str, node_config: dict[str, Any]) -> str:
    """Add a node to an existing pipeline YAML.

    Parameters
    ----------
    yaml_content : str
        Existing pipeline YAML as string
    node_config : dict[str, Any]
        Node configuration with keys:
        - kind: Node type (e.g., "llm_node", "function_node")
        - name: Unique node identifier
        - spec: Node-specific configuration dict
        - dependencies: Optional list of dependency node names

    Returns
    -------
    str
        JSON with {success: bool, yaml_content: str, warnings: list, node_count: int}
    """
    # Validate required fields
    if "kind" not in node_config:
        return orjson.dumps(
            {"success": False, "error": "node_config must have 'kind' field"},
            option=orjson.OPT_INDENT_2,
        ).decode()
    if "name" not in node_config:
        return orjson.dumps(
            {"success": False, "error": "node_config must have 'name' field"},
            option=orjson.OPT_INDENT_2,
        ).decode()

    result = api.pipeline.add_node(
        yaml_content,
        kind=node_config["kind"],
        name=node_config["name"],
        spec=node_config.get("spec", {}),
        dependencies=node_config.get("dependencies", []),
    )
    return orjson.dumps(result, option=orjson.OPT_INDENT_2).decode()


@mcp.tool()  # type: ignore[misc]
def remove_node_from_pipeline(yaml_content: str, node_name: str) -> str:
    """Remove a node from a pipeline YAML.

    Parameters
    ----------
    yaml_content : str
        Existing pipeline YAML as string
    node_name : str
        Name of the node to remove

    Returns
    -------
    str
        JSON with {success: bool, yaml_content: str, warnings: list}
        Warns if other nodes depend on the removed node.
    """
    result = api.pipeline.remove_node(yaml_content, node_name)
    return orjson.dumps(result, option=orjson.OPT_INDENT_2).decode()


@mcp.tool()  # type: ignore[misc]
def update_node_config(yaml_content: str, node_name: str, config_updates: dict[str, Any]) -> str:
    """Update a node's configuration in pipeline YAML.

    Parameters
    ----------
    yaml_content : str
        Existing pipeline YAML as string
    node_name : str
        Name of the node to update
    config_updates : dict[str, Any]
        Updates to apply:
        - spec: Dict of spec fields to merge/update
        - dependencies: New dependencies list (replaces existing)
        - kind: New node type (use with caution)

    Returns
    -------
    str
        JSON with {success: bool, yaml_content: str}
    """
    result = api.pipeline.update_node(
        yaml_content,
        node_name,
        spec=config_updates.get("spec"),
        dependencies=config_updates.get("dependencies"),
        kind=config_updates.get("kind"),
    )
    return orjson.dumps(result, option=orjson.OPT_INDENT_2).decode()


@mcp.tool()  # type: ignore[misc]
def list_pipeline_nodes(yaml_content: str) -> str:
    """List all nodes in a pipeline with their dependencies.

    Parameters
    ----------
    yaml_content : str
        Pipeline YAML as string

    Returns
    -------
    str
        JSON with:
        {
            success: bool,
            pipeline_name: str,
            node_count: int,
            nodes: [{name, kind, dependencies, dependents}],
            execution_order: [str]
        }
    """
    result = api.pipeline.list_nodes(yaml_content)
    return orjson.dumps(result, option=orjson.OPT_INDENT_2).decode()


# ============================================================================
# Template Generation Tools (keep original implementations for now)
# ============================================================================


@mcp.tool()  # type: ignore[misc]
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
        YAML pipeline template as string
    """

    # Start with init
    result = api.pipeline.init(pipeline_name, description)
    if not result["success"]:
        return orjson.dumps(
            {"success": False, "error": result.get("error")}, option=orjson.OPT_INDENT_2
        ).decode()

    yaml_content = result["yaml_content"]

    # Add nodes for each type
    for i, node_type in enumerate(node_types):
        node_name = f"{node_type.replace('_node', '')}_{i + 1}"
        deps = [f"{node_types[i - 1].replace('_node', '')}_{i}"] if i > 0 else []

        add_result = api.pipeline.add_node(
            yaml_content,
            kind=node_type,
            name=node_name,
            spec={},  # Empty spec - user should fill in
            dependencies=deps,
        )

        if add_result["success"]:
            yaml_content = add_result["yaml_content"]

    return yaml_content


@mcp.tool()  # type: ignore[misc]
def build_yaml_pipeline_interactive(
    pipeline_name: str,
    description: str,
    nodes: list[dict[str, Any]],
    ports: dict[str, Any] | None = None,
) -> str:
    """Build a complete YAML pipeline from structured input.

    Parameters
    ----------
    pipeline_name : str
        Name for the pipeline
    description : str
        Pipeline description
    nodes : list[dict]
        List of node configurations, each with:
        - kind: Node type
        - name: Node name
        - spec: Node configuration
        - dependencies: List of dependency names
    ports : dict | None
        Optional port configurations (llm, memory, etc.)

    Returns
    -------
        Complete YAML pipeline as string
    """
    import yaml

    # Create base pipeline
    config = {
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

    if ports:
        config["spec"]["ports"] = ports

    # Add nodes
    for node_config in nodes:
        node = {
            "kind": node_config["kind"],
            "metadata": {"name": node_config["name"]},
            "spec": node_config.get("spec", {}),
            "dependencies": node_config.get("dependencies", []),
        }
        config["spec"]["nodes"].append(node)

    return yaml.dump(config, sort_keys=False, default_flow_style=False)


# Run the server
if __name__ == "__main__":
    mcp.run()
