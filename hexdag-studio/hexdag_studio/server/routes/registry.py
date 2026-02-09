"""Registry API for hexdag studio.

Provides access to hexDAG's built-in node types from the registry.
Uses the unified hexdag.api layer for feature parity with MCP server.
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from hexdag import api

router = APIRouter(prefix="/registry", tags=["registry"])


class NodeTypeInfo(BaseModel):
    """Information about a node type."""

    kind: str
    name: str
    description: str
    namespace: str
    color: str
    icon: str
    default_spec: dict[str, Any]
    required_ports: list[str]
    config_schema: dict[str, Any]


class NodeTypesResponse(BaseModel):
    """List of available node types."""

    nodes: list[NodeTypeInfo]


# Color mapping for node types (Studio-specific UI concern)
NODE_COLORS: dict[str, str] = {
    "function_node": "#3b82f6",  # blue
    "llm_node": "#8b5cf6",  # violet
    "re_act_agent_node": "#ec4899",  # pink
    "composite_node": "#6366f1",  # indigo
    "expression_node": "#06b6d4",  # cyan
    "tool_call_node": "#f97316",  # orange
    "port_call_node": "#84cc16",  # lime
    # Deprecated but still supported
    "conditional_node": "#f59e0b",  # amber
    "loop_node": "#10b981",  # emerald
    "data_node": "#14b8a6",  # teal
}

# Icon mapping for node types (Studio-specific UI concern)
NODE_ICONS: dict[str, str] = {
    "function_node": "Code",
    "llm_node": "Brain",
    "re_act_agent_node": "Bot",
    "composite_node": "Layers",
    "expression_node": "Calculator",
    "tool_call_node": "Wrench",
    "port_call_node": "Plug",
    # Deprecated but still supported
    "conditional_node": "GitBranch",
    "loop_node": "Repeat",
    "data_node": "Database",
}

# Default specs for node types (Studio-specific UI concern)
DEFAULT_SPECS: dict[str, dict[str, Any]] = {
    "function_node": {"fn": ""},
    "llm_node": {"prompt_template": ""},
    "re_act_agent_node": {"main_prompt": "", "config": {"max_steps": 5}},
    "composite_node": {"mode": "for-each", "items": "", "body": []},
    "expression_node": {"expressions": {}},
    "tool_call_node": {"tool_name": "", "arguments": {}},
    "port_call_node": {"port": "", "method": "", "input_mapping": {}},
    # Deprecated
    "conditional_node": {"branches": [], "else_action": None},
    "loop_node": {"while_condition": "", "body": "", "max_iterations": 100},
    "data_node": {"output": {}},
}

# Required ports for node types
REQUIRED_PORTS: dict[str, list[str]] = {
    "llm_node": ["llm"],
    "re_act_agent_node": ["llm", "tool_router"],
}

# Deprecated nodes that shouldn't appear in the palette
DEPRECATED_NODES = {"conditional_node", "loop_node", "data_node"}


def _kind_to_label(kind: str) -> str:
    """Convert node kind to human-readable label."""
    # Remove _node suffix and namespace prefix
    label = kind.split(":")[-1].replace("_node", "").replace("_", " ")
    # Title case
    return label.title()


def get_builtin_nodes(include_deprecated: bool = False) -> list[NodeTypeInfo]:
    """Get built-in hexDAG node types using the unified API.

    Uses hexdag.api.components.list_nodes() for discovery,
    then adds Studio-specific UI metadata (colors, icons, etc.).
    """
    # Use unified API to get nodes
    api_nodes = api.components.list_nodes()

    nodes: list[NodeTypeInfo] = []

    for node_data in api_nodes:
        kind = node_data.get("kind", "")
        namespace = node_data.get("namespace", "core")

        # Skip deprecated nodes if not requested
        if kind in DEPRECATED_NODES and not include_deprecated:
            continue

        # Get namespace prefix for plugin nodes
        base_kind = kind.split(":")[-1] if ":" in kind else kind

        nodes.append(
            NodeTypeInfo(
                kind=kind,
                name=node_data.get("name", _kind_to_label(kind)),
                description=node_data.get("description", f"{_kind_to_label(kind)} node"),
                namespace=namespace,
                # Studio-specific UI metadata
                color=NODE_COLORS.get(base_kind, "#6b7280"),
                icon=NODE_ICONS.get(base_kind, "Box"),
                default_spec=DEFAULT_SPECS.get(base_kind, {}),
                required_ports=REQUIRED_PORTS.get(base_kind, []),
                config_schema=node_data.get("schema", {}),
            )
        )

    return nodes


@router.get("/nodes", response_model=NodeTypesResponse)
async def get_node_types(include_deprecated: bool = False) -> NodeTypesResponse:
    """Get all available built-in node types.

    Uses the unified hexdag.api.components module for feature parity with MCP.

    Args:
        include_deprecated: Whether to include deprecated node types
    """
    nodes = get_builtin_nodes(include_deprecated)
    return NodeTypesResponse(nodes=nodes)


@router.get("/nodes/{kind}")
async def get_node_type(kind: str) -> NodeTypeInfo | dict[str, str]:
    """Get information about a specific node type.

    Uses the unified hexdag.api.components module for feature parity with MCP.
    """
    nodes = get_builtin_nodes(include_deprecated=True)

    for node in nodes:
        if node.kind == kind:
            return node

    return {"error": f"Node type '{kind}' not found"}


# Additional endpoints for feature parity with MCP


@router.get("/adapters")
async def get_adapters(
    port_type: str | None = None, include_plugins: bool = True
) -> list[dict[str, Any]]:
    """Get all available adapters (built-in and from plugins).

    Uses the unified hexdag.api.components module for built-in adapters,
    plus discovers plugin adapters for Studio.

    Args:
        port_type: Optional filter by port type (llm, memory, database, secret)
        include_plugins: Whether to include plugin adapters (default: True)
    """
    # Get built-in adapters from unified API
    adapters = api.components.list_adapters(port_type)

    # Add plugin marker for built-in adapters
    for adapter in adapters:
        adapter["plugin"] = "builtin"

    # Include plugin adapters if requested
    if include_plugins:
        from hexdag_studio.server.routes.plugins import discover_plugins

        plugins = discover_plugins()
        for plugin in plugins:
            for adapter in plugin.adapters:
                # Filter by port type if specified
                if port_type and adapter.get("port_type") != port_type:
                    continue
                adapter_copy = dict(adapter)
                adapter_copy["plugin"] = plugin.name
                adapters.append(adapter_copy)

    return adapters


@router.get("/tools")
async def get_tools(
    namespace: str | None = None, include_plugins: bool = True
) -> list[dict[str, Any]]:
    """Get all available tools (built-in and from plugins).

    Uses the unified hexdag.api.components module for built-in tools,
    plus discovers plugin tools for Studio.

    Args:
        namespace: Optional filter by namespace
        include_plugins: Whether to include plugin tools (default: True)
    """
    # Get built-in tools from unified API
    tools = api.components.list_tools(namespace)

    # Add plugin marker for built-in tools
    for tool in tools:
        tool["plugin"] = "builtin"

    # Include plugin tools if requested (future support)
    # Plugins don't currently export tools, but this is ready for when they do
    if include_plugins:
        from hexdag_studio.server.routes.plugins import discover_plugins

        plugins = discover_plugins()
        for plugin in plugins:
            # Check if plugin has tools (future feature)
            plugin_tools = getattr(plugin, "tools", [])
            for tool in plugin_tools:
                if namespace and tool.get("namespace") != namespace:
                    continue
                tool_copy = dict(tool)
                tool_copy["plugin"] = plugin.name
                tools.append(tool_copy)

    return tools


@router.get("/macros")
async def get_macros() -> list[dict[str, Any]]:
    """Get all available macros.

    Uses the unified hexdag.api.components module for feature parity with MCP.
    """
    return api.components.list_macros()


@router.get("/tags")
async def get_tags() -> list[dict[str, Any]]:
    """Get all available YAML custom tags.

    Uses the unified hexdag.api.components module for feature parity with MCP.
    """
    return api.components.list_tags()


@router.get("/schema/{component_type}/{name}")
async def get_component_schema(
    component_type: str,
    name: str,
    namespace: str = "core",
) -> dict[str, Any]:
    """Get detailed schema for a specific component.

    Uses the unified hexdag.api.components module for feature parity with MCP.

    Args:
        component_type: Type of component (node, adapter, tool, macro, tag)
        name: Component name
        namespace: Component namespace (default: core)
    """
    return api.components.get_component_schema(component_type, name, namespace)
