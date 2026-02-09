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
          "args": ["run", "python", "-m", "hexdag.mcp_server"]
        }
      }
    }
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from hexdag import api

# Create MCP server
mcp = FastMCP(
    "hexDAG",
    dependencies=["pydantic>=2.0", "pyyaml>=6.0", "jinja2>=3.1.0"],
)


# ============================================================================
# Component Discovery Tools
# ============================================================================


@mcp.tool()  # type: ignore[misc]
def list_nodes() -> str:
    """List all available node types with auto-generated documentation.

    Returns detailed information about each node type including:
    - Node name, namespace, and module path
    - Description (from docstring)
    - Parameters summary (from _yaml_schema if available)
    - Required vs optional parameters

    Returns
    -------
        JSON string with available nodes grouped by namespace
    """
    nodes = api.components.list_nodes()
    # Group by namespace
    by_namespace: dict[str, list[dict[str, Any]]] = {"core": []}
    for node in nodes:
        ns = node.get("namespace", "core")
        if ns not in by_namespace:
            by_namespace[ns] = []
        by_namespace[ns].append(node)
    return json.dumps(by_namespace, indent=2)


@mcp.tool()  # type: ignore[misc]
def list_adapters(port_type: str | None = None) -> str:
    """List all available adapters in the hexDAG registry.

    Args
    ----
        port_type: Optional filter by port type (e.g., "llm", "memory", "database", "secret")

    Returns
    -------
        JSON string with available adapters grouped by port type
    """
    adapters = api.components.list_adapters(port_type)
    # Group by port type
    by_port: dict[str, list[dict[str, Any]]] = {}
    for adapter in adapters:
        pt = adapter.get("port_type", "unknown")
        if pt not in by_port:
            by_port[pt] = []
        by_port[pt].append(adapter)
    return json.dumps(by_port, indent=2)


@mcp.tool()  # type: ignore[misc]
def list_tools(namespace: str | None = None) -> str:
    """List all available tools in the hexDAG registry.

    Args
    ----
        namespace: Optional filter by namespace (e.g., "core", "user", "plugin")

    Returns
    -------
        JSON string with available tools and their schemas
    """
    tools = api.components.list_tools(namespace)
    # Group by namespace
    by_namespace: dict[str, list[dict[str, Any]]] = {"core": []}
    for tool in tools:
        ns = tool.get("namespace", "core")
        if ns not in by_namespace:
            by_namespace[ns] = []
        by_namespace[ns].append(tool)
    return json.dumps(by_namespace, indent=2)


@mcp.tool()  # type: ignore[misc]
def list_macros() -> str:
    """List all available macros in the hexDAG registry.

    Macros are reusable pipeline templates that expand into subgraphs.

    Returns
    -------
        JSON string with available macros and their descriptions
    """
    macros = api.components.list_macros()
    return json.dumps(macros, indent=2)


@mcp.tool()  # type: ignore[misc]
def list_tags() -> str:
    """List all available YAML custom tags.

    Returns detailed information about each tag including:
    - Tag name (e.g., "!py", "!include")
    - Description
    - Module path
    - Syntax examples
    - Security warnings (if applicable)

    Returns
    -------
        JSON string with available tags and their documentation
    """
    tags = api.components.list_tags()
    # Convert list to dict by name for backward compatibility
    result: dict[str, dict[str, Any]] = {}
    for tag in tags:
        result[tag["name"]] = tag
    return json.dumps(result, indent=2)


@mcp.tool()  # type: ignore[misc]
def get_component_schema(
    component_type: str,
    name: str,
    namespace: str = "core",
) -> str:
    """Get detailed schema for a specific component.

    Parameters
    ----------
    component_type : str
        Type of component: "node", "adapter", "tool", "macro", "tag"
    name : str
        Component name (e.g., "llm_node", "OpenAIAdapter", "!py")
    namespace : str
        Component namespace (default: "core")

    Returns
    -------
        JSON Schema for the component
    """
    schema = api.components.get_component_schema(component_type, name, namespace)
    return json.dumps(schema, indent=2)


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
    return json.dumps(result, indent=2)


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
    return json.dumps(result, indent=2)


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
    return json.dumps(result, indent=2)


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
    return json.dumps(result, indent=2)


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
    return json.dumps(result, indent=2)


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
        return json.dumps(
            {"success": False, "error": "node_config must have 'kind' field"}, indent=2
        )
    if "name" not in node_config:
        return json.dumps(
            {"success": False, "error": "node_config must have 'name' field"}, indent=2
        )

    result = api.pipeline.add_node(
        yaml_content,
        kind=node_config["kind"],
        name=node_config["name"],
        spec=node_config.get("spec", {}),
        dependencies=node_config.get("dependencies", []),
    )
    return json.dumps(result, indent=2)


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
    return json.dumps(result, indent=2)


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
    return json.dumps(result, indent=2)


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
    return json.dumps(result, indent=2)


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
        return json.dumps({"success": False, "error": result.get("error")}, indent=2)

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
