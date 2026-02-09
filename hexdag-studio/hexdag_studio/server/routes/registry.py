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
    module_path: str
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


def _is_builtin(module_path: str) -> bool:
    """Check if a module_path refers to a built-in component."""
    return module_path.startswith("hexdag.builtin.")


def get_builtin_nodes(include_deprecated: bool = False) -> list[NodeTypeInfo]:
    """Get built-in (core) hexDAG node types using the unified API.

    Uses hexdag.api.components.list_nodes() for discovery,
    then adds Studio-specific UI metadata (colors, icons, etc.).

    Only returns nodes from hexdag.builtin.* to avoid duplicates with
    plugin nodes which are served by /plugins endpoints.
    """
    # Use unified API to get nodes
    api_nodes = api.components.list_nodes()

    nodes: list[NodeTypeInfo] = []

    for node_data in api_nodes:
        kind = node_data.get("kind", "")
        module_path = node_data.get("module_path", "")

        # Only include built-in nodes - plugins are served by /plugins
        if not _is_builtin(module_path):
            continue

        # Skip deprecated nodes if not requested
        if kind in DEPRECATED_NODES and not include_deprecated:
            continue

        # Get base kind for UI metadata lookup
        base_kind = kind.split(":")[-1] if ":" in kind else kind

        nodes.append(
            NodeTypeInfo(
                kind=kind,
                name=node_data.get("name", _kind_to_label(kind)),
                description=node_data.get("description", f"{_kind_to_label(kind)} node"),
                module_path=module_path,
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


def _module_path_to_plugin(module_path: str) -> str:
    """Convert module_path to plugin name, or 'builtin' for built-in components."""
    if module_path.startswith("hexdag.builtin."):
        return "builtin"
    if module_path.startswith("hexdag_plugins."):
        parts = module_path.split(".")
        if len(parts) >= 2:
            return parts[1]
    # User plugin - use top-level package
    parts = module_path.split(".")
    return parts[0] if parts else "unknown"


@router.get("/adapters")
async def get_adapters(port_type: str | None = None) -> list[dict[str, Any]]:
    """Get all available adapters from unified API.

    Uses the unified hexdag.api.components module for all adapters.
    Adds plugin name based on module_path.

    Args:
        port_type: Optional filter by port type (llm, memory, database, secret)
    """
    # Get all adapters from unified API (already deduplicated)
    adapters = api.components.list_adapters(port_type)

    # Add plugin marker based on module_path
    return [
        {**adapter, "plugin": _module_path_to_plugin(adapter.get("module_path", ""))}
        for adapter in adapters
    ]


@router.get("/tools")
async def get_tools() -> list[dict[str, Any]]:
    """Get all available tools from unified API.

    Uses the unified hexdag.api.components module for all tools.
    """
    # Get all tools from unified API
    tools = api.components.list_tools()

    # Add plugin marker based on module_path
    return [
        {**tool, "plugin": _module_path_to_plugin(tool.get("module_path", ""))} for tool in tools
    ]


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
) -> dict[str, Any]:
    """Get detailed schema for a specific component.

    Uses the unified hexdag.api.components module for feature parity with MCP.

    Args:
        component_type: Type of component (node, adapter, tool, macro, tag)
        name: Component name or module_path
    """
    return api.components.get_component_schema(component_type, name)
