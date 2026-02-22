"""Component discovery API.

Provides unified functions for listing and inspecting hexDAG components:
- Nodes
- Adapters
- Tools
- Macros
- Tags (YAML custom tags)

Uses core SchemaGenerator for schema extraction - thin wrapper over core functionality.
"""

from __future__ import annotations

import logging
from typing import Any

from hexdag.kernel.pipeline_builder.tag_discovery import discover_tags, get_tag_schema
from hexdag.kernel.resolver import get_builtin_aliases, resolve
from hexdag.kernel.schema import SchemaGenerator

logger = logging.getLogger(__name__)


def is_builtin(module_path: str) -> bool:
    """Check if a module_path refers to a built-in component.

    Parameters
    ----------
    module_path : str
        Full import path (e.g., "hexdag.stdlib.nodes.LLMNode")

    Returns
    -------
    bool
        True if this is a built-in hexDAG component
    """
    return module_path.startswith("hexdag.stdlib.")


def list_nodes(include_deprecated: bool = False) -> list[dict[str, Any]]:
    """List all available node types with schemas.

    Discovers nodes from:
    1. Builtin nodes (hexdag.stdlib.nodes.*)
    2. User plugin paths (HEXDAG_PLUGIN_PATHS env var or set_user_plugin_paths())

    Parameters
    ----------
    include_deprecated : bool
        If True, include deprecated node types (default: False)

    Returns
    -------
    list[dict]
        List of node info dicts with keys:
        - kind: Node type alias (e.g., "llm_node")
        - name: Class name (e.g., "LLMNode")
        - module_path: Full import path (unique identifier)
        - description: Short description from docstring
        - parameters: Optional dict with "required" and "optional" lists
        - schema: Optional JSON Schema dict

    Examples
    --------
    >>> nodes = list_nodes()
    >>> any(n["kind"] == "llm_node" for n in nodes)
    True
    """
    from hexdag.kernel.discovery import discover_user_plugins

    aliases = get_builtin_aliases()
    seen_classes: set[str] = set()
    nodes: list[dict[str, Any]] = []

    # 1. Discover builtin nodes
    for alias, full_path in aliases.items():
        # Only include primary aliases (snake_case without namespace prefix)
        if ":" in alias:
            continue

        # Only include node types (ending with _node)
        if not alias.endswith("_node"):
            continue

        # Avoid duplicates (same class can have multiple aliases)
        if full_path in seen_classes:
            continue
        seen_classes.add(full_path)

        try:
            cls = resolve(full_path)
        except Exception:
            continue

        # Check if node is deprecated (docstring contains "deprecated" marker)
        is_deprecated = False
        if cls.__doc__:
            doc_lower = cls.__doc__.lower()
            is_deprecated = "deprecated" in doc_lower or ".. deprecated::" in doc_lower

        # Skip deprecated nodes unless explicitly requested
        if is_deprecated and not include_deprecated:
            continue

        # Extract description from _yaml_schema or docstring
        yaml_schema = getattr(cls, "_yaml_schema", None)
        schema: dict[str, Any] | None = None
        params_info: dict[str, Any] | None = None

        if yaml_schema and isinstance(yaml_schema, dict):
            description = yaml_schema.get(
                "description", (cls.__doc__ or "No description").split("\n")[0].strip()
            )
            # Extract parameter info from schema
            properties = yaml_schema.get("properties", {})
            required = yaml_schema.get("required", [])
            optional = [k for k in properties if k not in required]
            params_info = {"required": required, "optional": optional}
            schema = yaml_schema
        else:
            description = (cls.__doc__ or "No description available").split("\n")[0].strip()
            # Try to generate schema from __call__ (node factories define config there)
            try:
                # Node classes are factories - config params are in __call__, not __init__
                # Check if __call__ is overridden (not just inherited from object)
                for klass in cls.__mro__:
                    if klass is object:
                        continue
                    if "__call__" in klass.__dict__:
                        result = SchemaGenerator.from_callable(cls.__call__)
                        if isinstance(result, dict) and result.get("properties"):
                            schema = result
                            break
                # Fall back to class itself if no __call__ override
                if not schema or not schema.get("properties"):
                    result = SchemaGenerator.from_callable(cls)
                    if isinstance(result, dict):
                        schema = result
            except Exception:
                schema = None

        node_info: dict[str, Any] = {
            "kind": alias,
            "name": cls.__name__,
            "module_path": full_path,
            "description": description,
        }

        if params_info:
            node_info["parameters"] = params_info
        if schema:
            node_info["schema"] = schema

        nodes.append(node_info)

    # 2. Discover user plugin nodes (HEXDAG_PLUGIN_PATHS env var)
    for plugin_info in discover_user_plugins():
        for node in plugin_info.get("nodes", []):
            # Avoid duplicates
            module_path = node.get("module_path", "")
            if module_path in seen_classes:
                continue
            seen_classes.add(module_path)
            # Remove namespace if present (we use module_path for identification)
            node_copy = {k: v for k, v in node.items() if k != "namespace"}
            nodes.append(node_copy)

    return sorted(nodes, key=lambda x: x["kind"])


def list_adapters(port_type: str | None = None) -> list[dict[str, Any]]:
    """List all available adapters.

    Discovers adapters dynamically from four sources:
    1. hexdag.stdlib.adapters.* (builtin adapters)
    2. hexdag_plugins.* (installed plugin packages)
    3. User plugin paths (HEXDAG_PLUGIN_PATHS env var or set_user_plugin_paths())
    4. User-configured modules from hexdag.toml/pyproject.toml

    Parameters
    ----------
    port_type : str | None
        Optional filter by port type (e.g., "llm", "memory", "database", "secret")

    Returns
    -------
    list[dict]
        List of adapter info dicts with keys:
        - name: Adapter class name
        - port_type: Port type the adapter implements
        - module_path: Full import path (unique identifier)
        - description: Short description from docstring

    Examples
    --------
    >>> adapters = list_adapters(port_type="llm")
    >>> all(a["port_type"] == "llm" for a in adapters)
    True
    """
    from hexdag.kernel.discovery import (
        discover_adapters_in_package,
        discover_plugins,
        discover_user_modules,
        discover_user_plugins,
    )

    adapters: list[dict[str, Any]] = []

    # 1. Discover builtin adapters dynamically
    adapters.extend(discover_adapters_in_package("hexdag.stdlib.adapters", detect_port_type))

    # 2. Discover plugin adapters (hexdag_plugins namespace)
    for plugin_name in discover_plugins():
        # Try structured layout: hexdag_plugins/<plugin>/adapters/
        adapters.extend(
            discover_adapters_in_package(
                f"hexdag_plugins.{plugin_name}.adapters",
                detect_port_type,
            )
        )
        # Also try flat layout: hexdag_plugins/<plugin>/
        adapters.extend(
            discover_adapters_in_package(
                f"hexdag_plugins.{plugin_name}",
                detect_port_type,
            )
        )

    # 3. Discover user plugin paths (HEXDAG_PLUGIN_PATHS env var)
    for plugin_info in discover_user_plugins():
        for adapter in plugin_info.get("adapters", []):
            # Remove namespace if present
            adapter_copy = {k: v for k, v in adapter.items() if k != "namespace"}
            adapters.append(adapter_copy)

    # 4. Discover user-configured modules from hexdag.toml
    for user_module in discover_user_modules():
        # Try as adapters subpackage
        adapters.extend(
            discover_adapters_in_package(
                f"{user_module}.adapters" if not user_module.endswith(".adapters") else user_module,
                detect_port_type,
            )
        )
        # Also try the module directly (flat layout)
        adapters.extend(
            discover_adapters_in_package(
                user_module,
                detect_port_type,
            )
        )

    # Filter by port type if specified
    if port_type:
        adapters = [a for a in adapters if a["port_type"] == port_type]

    # Remove duplicates (same class discovered via multiple paths)
    seen: set[str] = set()
    unique_adapters: list[dict[str, Any]] = []
    for adapter in adapters:
        key = adapter["module_path"]
        if key not in seen:
            seen.add(key)
            unique_adapters.append(adapter)

    return sorted(unique_adapters, key=lambda x: (x["port_type"], x["name"]))


def detect_port_type(adapter_class: type) -> str:
    """Detect port type from adapter class using protocol inspection.

    Adapters MUST inherit from their port protocol to be properly detected.
    This is a convention enforced by hexDAG - no name-based guessing.

    Port Protocol Convention
    ------------------------
    All adapters must inherit from their corresponding port protocol:

    - LLM adapters: inherit from `LLM`, `SupportsGeneration`, `SupportsFunctionCalling`, etc.
    - Memory adapters: inherit from `Memory`
    - Database adapters: inherit from `Database` or `SQLAdapter`
    - Secret adapters: inherit from `SecretStore`
    - Storage adapters: inherit from `FileStorage` or `VectorStorePort`
    - Tool adapters: inherit from `ToolRouter`

    Example::

        from hexdag.kernel.ports.llm import LLM

        class MyCustomLLMAdapter(LLM):
            async def aresponse(self, messages):
                ...

    Parameters
    ----------
    adapter_class : type
        The adapter class to inspect

    Returns
    -------
    str
        Port type: "llm", "memory", "database", "secret", "storage", "tool_router", or "unknown"

    Examples
    --------
    >>> from hexdag.stdlib.adapters.openai import OpenAIAdapter
    >>> detect_port_type(OpenAIAdapter)
    'llm'
    """
    # Check explicit decorator metadata first (future @adapter decorator)
    explicit_port = getattr(adapter_class, "_hexdag_implements_port", None)
    if explicit_port:
        return str(explicit_port)

    # Check protocol inheritance (required convention)
    mro_names = [c.__name__ for c in adapter_class.__mro__]

    # LLM adapters implement LLM, SupportsGeneration, SupportsFunctionCalling, etc.
    llm_protocols = {"LLM", "SupportsGeneration", "SupportsFunctionCalling", "SupportsVision"}
    if any(name in mro_names for name in llm_protocols):
        return "llm"

    # Memory adapters implement Memory protocol
    if "Memory" in mro_names:
        return "memory"

    # Database adapters implement Database or SQLAdapter
    if "Database" in mro_names or "DatabasePort" in mro_names or "SQLAdapter" in mro_names:
        return "database"

    # Secret adapters implement SecretStore
    if "SecretStore" in mro_names or "SecretPort" in mro_names:
        return "secret"

    # Storage adapters implement FileStorage or VectorStorePort
    storage_ports = ("FileStorage", "FileStoragePort", "VectorStorePort")
    if any(name in mro_names for name in storage_ports):
        return "storage"

    # DataStore adapters implement DataStore
    if "DataStore" in mro_names:
        return "data_store"

    # Tool router adapters implement ToolRouter
    if "ToolRouter" in mro_names:
        return "tool_router"

    # No fallback - adapters must follow the convention
    return "unknown"


def list_tools() -> list[dict[str, Any]]:
    """List all available tools.

    Discovers tools dynamically from three levels:
    1. hexdag.kernel.domain.agent_tools (builtin tools)
    2. hexdag_plugins.*/tools (plugin tools)
    3. User-configured modules from hexdag.toml/pyproject.toml

    Returns
    -------
    list[dict]
        List of tool info dicts with keys:
        - name: Tool function name
        - module_path: Full import path (unique identifier)
        - description: Short description from docstring

    Examples
    --------
    >>> tools = list_tools()
    >>> any(t["name"] == "tool_end" for t in tools)
    True
    """
    from hexdag.kernel.discovery import (
        discover_plugins,
        discover_tools_in_module,
        discover_user_modules,
    )

    tools: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_tools(tool_list: list[dict[str, Any]]) -> None:
        for tool in tool_list:
            module_path = tool.get("module_path", "")
            if module_path not in seen:
                seen.add(module_path)
                tools.append(tool)

    # 1. Discover builtin tools
    add_tools(discover_tools_in_module("hexdag.kernel.domain.agent_tools"))

    # 2. Discover plugin tools
    for plugin_name in discover_plugins():
        add_tools(discover_tools_in_module(f"hexdag_plugins.{plugin_name}.tools"))

    # 3. Discover user-configured tools
    for user_module in discover_user_modules():
        # Try as tools submodule
        add_tools(
            discover_tools_in_module(
                f"{user_module}.tools" if not user_module.endswith(".tools") else user_module
            )
        )

    return sorted(tools, key=lambda x: x["name"])


def list_macros() -> list[dict[str, Any]]:
    """List all available macros.

    Discovers macros dynamically from:
    1. hexdag.stdlib.macros (builtin macros)
    2. hexdag_plugins.*/macros (plugin macros)

    Macros are reusable pipeline templates that expand into subgraphs.

    Returns
    -------
    list[dict]
        List of macro info dicts with keys:
        - name: Macro class name
        - module_path: Full import path (unique identifier)
        - description: Short description from docstring

    Examples
    --------
    >>> macros = list_macros()
    >>> any(m["name"] == "ReasoningAgentMacro" for m in macros)
    True
    """
    from hexdag.kernel.discovery import discover_macros_in_module, discover_plugins

    macros: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_macros(macro_list: list[dict[str, Any]]) -> None:
        for macro in macro_list:
            module_path = macro.get("module_path", "")
            if module_path not in seen:
                seen.add(module_path)
                macros.append(macro)

    # Discover builtin macros
    add_macros(discover_macros_in_module("hexdag.stdlib.macros"))

    # Discover plugin macros
    for plugin_name in discover_plugins():
        add_macros(discover_macros_in_module(f"hexdag_plugins.{plugin_name}.macros"))

    return sorted(macros, key=lambda x: x["name"])


def list_tags() -> list[dict[str, Any]]:
    """List all available YAML custom tags.

    Returns
    -------
    list[dict]
        List of tag info dicts with keys:
        - name: Tag name (e.g., "!py", "!include")
        - description: Tag description
        - module: Module where tag handler is defined
        - syntax: List of syntax examples
        - is_registered: Whether tag is registered with YAML loader
        - security_warning: Optional security warning for dangerous tags

    Examples
    --------
    >>> tags = list_tags()
    >>> any(t["name"] == "!py" for t in tags)
    True
    """
    tag_dict = discover_tags()
    tags: list[dict[str, Any]] = []

    for tag_info in tag_dict.values():
        tag_data: dict[str, Any] = {
            "name": tag_info["name"],
            "description": tag_info["description"],
            "module": tag_info["module"],
            "syntax": tag_info.get("syntax", []),
            "is_registered": tag_info.get("is_registered", False),
        }

        # Include security warning if present
        if "security_warning" in tag_info:
            tag_data["security_warning"] = tag_info["security_warning"]

        tags.append(tag_data)

    return sorted(tags, key=lambda x: x["name"])


def get_component_schema(component_type: str, name: str) -> dict[str, Any]:
    """Get detailed schema for a specific component.

    Parameters
    ----------
    component_type : str
        Type of component: "node", "adapter", "tool", "macro", "tag"
    name : str
        Component name or module_path (e.g., "llm_node", "OpenAIAdapter", "!py")

    Returns
    -------
    dict
        JSON Schema for the component, or error dict if not found

    Examples
    --------
    >>> schema = get_component_schema("node", "llm_node")
    >>> "properties" in schema or "error" in schema
    True
    """
    if component_type == "node":
        return _get_node_schema(name)
    if component_type == "adapter":
        return _get_adapter_schema(name)
    if component_type == "tool":
        return _get_tool_schema(name)
    if component_type == "macro":
        return _get_macro_schema(name)
    if component_type == "tag":
        return _get_tag_schema(name)
    return {"error": f"Unknown component type: {component_type}"}


def _get_node_schema(name: str) -> dict[str, Any]:
    """Get schema for a node type."""
    try:
        cls = resolve(name)
    except Exception as e:
        return {"error": f"Cannot resolve node '{name}': {e}"}

    # Check for explicit _yaml_schema
    yaml_schema = getattr(cls, "_yaml_schema", None)
    if yaml_schema and isinstance(yaml_schema, dict):
        return dict(yaml_schema)

    # Fall back to SchemaGenerator
    try:
        result = SchemaGenerator.from_callable(cls)
        if isinstance(result, dict):
            return result
        return {"error": f"Schema generator returned non-dict for '{name}'"}
    except Exception as e:
        return {"error": f"Cannot generate schema for '{name}': {e}"}


def _get_adapter_schema(name: str) -> dict[str, Any]:
    """Get schema for an adapter."""
    try:
        from hexdag.stdlib import adapters as builtin_adapters

        cls = getattr(builtin_adapters, name, None)
        if cls is None:
            return {"error": f"Adapter '{name}' not found"}

        result = SchemaGenerator.from_callable(cls)
        if isinstance(result, dict):
            return result
        return {"error": f"Schema generator returned non-dict for adapter '{name}'"}
    except Exception as e:
        return {"error": f"Cannot generate schema for adapter '{name}': {e}"}


def _get_tool_schema(name: str) -> dict[str, Any]:
    """Get schema for a tool."""
    try:
        from hexdag.kernel.domain import agent_tools

        fn = getattr(agent_tools, name, None)
        if fn is None:
            return {"error": f"Tool '{name}' not found"}

        result = SchemaGenerator.from_callable(fn)
        if isinstance(result, dict):
            return result
        return {"error": f"Schema generator returned non-dict for tool '{name}'"}
    except Exception as e:
        return {"error": f"Cannot generate schema for tool '{name}': {e}"}


def _get_macro_schema(name: str) -> dict[str, Any]:
    """Get schema for a macro."""
    try:
        from hexdag.stdlib import macros as builtin_macros

        cls = getattr(builtin_macros, name, None)
        if cls is None:
            return {"error": f"Macro '{name}' not found"}

        result = SchemaGenerator.from_callable(cls)
        if isinstance(result, dict):
            return result
        return {"error": f"Schema generator returned non-dict for macro '{name}'"}
    except Exception as e:
        return {"error": f"Cannot generate schema for macro '{name}': {e}"}


def _get_tag_schema(name: str) -> dict[str, Any]:
    """Get schema for a YAML tag."""
    try:
        schema = get_tag_schema(name)
        if schema:
            return schema
        return {"error": f"Tag '{name}' not found or has no schema"}
    except ValueError as e:
        return {"error": str(e)}
