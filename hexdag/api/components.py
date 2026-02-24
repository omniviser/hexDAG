"""Component discovery engine (internal).

This module is the authoritative source for component discovery.
VFS ``LibProvider`` delegates to these functions. External consumers
(MCP server, Studio) should use VFS paths instead of calling
these functions directly.

Discovers:
- Nodes, Adapters, Tools, Macros, Tags (YAML custom tags)
"""

from __future__ import annotations

import logging
from typing import Any

from hexdag.compiler.tag_discovery import discover_tags, get_tag_schema
from hexdag.kernel import (
    SchemaGenerator,
    detect_port_type,
    discover_adapters_in_package,
    discover_macros_in_module,
    discover_plugins,
    discover_tools_in_module,
    discover_user_modules,
    discover_user_plugins,
    get_builtin_aliases,
    resolve,
)

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


def _discover_entities(
    discover_fn: Any,
    builtin_modules: list[str],
    plugin_submodule: str,
    user_submodule: str | None = None,
    sort_key: str = "name",
) -> list[dict[str, Any]]:
    """Generic multi-source entity discovery with deduplication.

    Discovers from builtins, plugins, and optionally user modules.
    Deduplicates by ``module_path`` and sorts by *sort_key*.
    """
    entities: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(items: list[dict[str, Any]]) -> None:
        for item in items:
            mp = item.get("module_path", "")
            if mp not in seen:
                seen.add(mp)
                entities.append(item)

    # 1. Builtins
    for mod in builtin_modules:
        _add(discover_fn(mod))

    # 2. Plugins
    for plugin_name in discover_plugins():
        _add(discover_fn(f"hexdag_plugins.{plugin_name}.{plugin_submodule}"))

    # 3. User modules
    if user_submodule is not None:
        for user_module in discover_user_modules():
            suffix = user_submodule
            if not user_module.endswith(f".{suffix}"):
                _add(discover_fn(f"{user_module}.{suffix}"))
            else:
                _add(discover_fn(user_module))

    return sorted(entities, key=lambda x: x[sort_key])


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
    return _discover_entities(
        discover_tools_in_module,
        builtin_modules=["hexdag.kernel.domain.agent_tools"],
        plugin_submodule="tools",
        user_submodule="tools",
    )


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
    return _discover_entities(
        discover_macros_in_module,
        builtin_modules=["hexdag.stdlib.macros"],
        plugin_submodule="macros",
    )


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

    Uses :func:`resolve` to find components by alias or module path,
    then extracts schema via ``_yaml_schema`` attribute or
    :class:`SchemaGenerator`. Works for builtins and plugins alike.

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
    if component_type == "tag":
        return _get_tag_schema(name)
    if component_type not in ("node", "adapter", "tool", "macro"):
        return {"error": f"Unknown component type: {component_type}"}
    return _resolve_schema(name)


def _resolve_schema(name: str) -> dict[str, Any]:
    """Resolve a component and extract its schema."""
    try:
        cls_or_fn = resolve(name)
    except Exception as e:
        return {"error": f"Cannot resolve '{name}': {e}"}

    # Check for explicit _yaml_schema
    yaml_schema = getattr(cls_or_fn, "_yaml_schema", None)
    if yaml_schema and isinstance(yaml_schema, dict):
        return dict(yaml_schema)

    # Fall back to SchemaGenerator
    try:
        result = SchemaGenerator.from_callable(cls_or_fn)
        if isinstance(result, dict):
            return result
        return {"error": f"Schema generator returned non-dict for '{name}'"}
    except Exception as e:
        return {"error": f"Cannot generate schema for '{name}': {e}"}


def _get_tag_schema(name: str) -> dict[str, Any]:
    """Get schema for a YAML tag."""
    try:
        schema = get_tag_schema(name)
        if schema:
            return schema
        return {"error": f"Tag '{name}' not found or has no schema"}
    except ValueError as e:
        return {"error": str(e)}
