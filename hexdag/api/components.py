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

import importlib
import logging
from typing import Any

from hexdag.core.pipeline_builder.tag_discovery import discover_tags, get_tag_schema
from hexdag.core.resolver import get_builtin_aliases, resolve
from hexdag.core.schema import SchemaGenerator

logger = logging.getLogger(__name__)


def list_nodes(include_deprecated: bool = False) -> list[dict[str, Any]]:
    """List all available node types with schemas.

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
        - namespace: Component namespace (e.g., "core")
        - module_path: Full import path
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
            "namespace": "core",
            "module_path": full_path,
            "description": description,
        }

        if params_info:
            node_info["parameters"] = params_info
        if schema:
            node_info["schema"] = schema

        nodes.append(node_info)

    return sorted(nodes, key=lambda x: x["kind"])


def list_adapters(port_type: str | None = None) -> list[dict[str, Any]]:
    """List all available adapters.

    Discovers adapters from:
    1. hexdag.builtin.adapters.* (builtin adapters)
    2. hexdag_plugins.* (plugin adapters)

    Parameters
    ----------
    port_type : str | None
        Optional filter by port type (e.g., "llm", "memory", "database", "secret")

    Returns
    -------
    list[dict]
        List of adapter info dicts with keys:
        - name: Adapter class name
        - namespace: Component namespace (e.g., "core" or "plugins")
        - port_type: Port type the adapter implements
        - module_path: Full import path
        - description: Short description from docstring

    Examples
    --------
    >>> adapters = list_adapters(port_type="llm")
    >>> all(a["port_type"] == "llm" for a in adapters)
    True
    """
    adapters: list[dict[str, Any]] = []
    seen_classes: set[str] = set()

    # Scan builtin adapters submodules
    adapter_modules = [
        ("hexdag.builtin.adapters.mock", "llm", "core"),
        ("hexdag.builtin.adapters.openai", "llm", "core"),
        ("hexdag.builtin.adapters.anthropic", "llm", "core"),
        ("hexdag.builtin.adapters.memory", "memory", "core"),
        ("hexdag.builtin.adapters.database.sqlite", "database", "core"),
        ("hexdag.builtin.adapters.secret", "secret", "core"),
    ]

    # Also scan hexdag_plugins for plugin adapters
    # Plugin convention: hexdag_plugins/<plugin_name>/adapters/ or hexdag_plugins/<plugin_name>/
    try:
        import pkgutil

        import hexdag_plugins

        for module_info in pkgutil.iter_modules(hexdag_plugins.__path__):
            plugin_name = module_info.name
            # Try structured layout first: hexdag_plugins/<plugin>/adapters/
            adapter_modules.append((
                f"hexdag_plugins.{plugin_name}.adapters",
                "unknown",
                f"plugins.{plugin_name}",
            ))
            # Also scan root of plugin (for flat layout like azure/)
            adapter_modules.append((
                f"hexdag_plugins.{plugin_name}",
                "unknown",
                f"plugins.{plugin_name}",
            ))
    except ImportError:
        pass

    for module_path, guessed_port, namespace in adapter_modules:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue

        for name in dir(module):
            if name.startswith("_"):
                continue

            obj = getattr(module, name, None)
            if obj is None or not isinstance(obj, type):
                continue

            # Check if it's an adapter class (ends with Adapter or adapter-like)
            if not (name.endswith("Adapter") or name.endswith("Memory") or name.endswith("LLM")):
                continue

            # Avoid duplicates
            full_path = f"{module_path}.{name}"
            if full_path in seen_classes:
                continue
            seen_classes.add(full_path)

            # Get actual port type using protocol inspection (preferred) or name guessing
            actual_port = detect_port_type(obj)
            # Fallback to module-based guess if still unknown
            if actual_port == "unknown" and guessed_port != "unknown":
                actual_port = guessed_port

            # Filter by port type if specified (after determining actual type)
            if port_type and actual_port != port_type:
                continue

            # Get secrets from decorator
            secrets_dict = getattr(obj, "_hexdag_secrets", {})
            secrets = list(secrets_dict.keys()) if secrets_dict else []

            # Use core SchemaGenerator for config schema
            try:
                config_schema = SchemaGenerator.from_callable(obj.__init__)
            except Exception:
                config_schema = {"type": "object", "properties": {}}

            adapter_info = {
                "name": name,
                "namespace": namespace,
                "module_path": full_path,
                "port_type": actual_port,
                "description": (obj.__doc__ or "No description available").split("\n")[0].strip(),
                "config_schema": config_schema,
                "secrets": secrets,
            }

            adapters.append(adapter_info)

    return sorted(adapters, key=lambda x: (x["port_type"], x["name"]))


def detect_port_type(adapter_class: type) -> str:
    """Detect port type from adapter class using protocol inspection.

    Adapters MUST inherit from their port protocol to be properly detected.
    This is a convention enforced by hexDAG - no name-based guessing.

    Port Protocol Convention
    ------------------------
    All adapters must inherit from their corresponding port protocol:

    - LLM adapters: inherit from `LLM`, `SupportsGeneration`, `SupportsFunctionCalling`, etc.
    - Memory adapters: inherit from `Memory`
    - Database adapters: inherit from `DatabasePort` or `SQLAdapter`
    - Secret adapters: inherit from `SecretPort`
    - Storage adapters: inherit from `FileStoragePort` or `VectorStorePort`
    - Tool adapters: inherit from `ToolRouter`

    Example::

        from hexdag.core.ports.llm import LLM

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
    >>> from hexdag.builtin.adapters.openai import OpenAIAdapter
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

    # Database adapters implement DatabasePort or SQLAdapter
    if "DatabasePort" in mro_names or "SQLAdapter" in mro_names:
        return "database"

    # Secret adapters implement SecretPort
    if "SecretPort" in mro_names:
        return "secret"

    # Storage adapters implement FileStoragePort or VectorStorePort
    if "FileStoragePort" in mro_names or "VectorStorePort" in mro_names:
        return "storage"

    # Tool router adapters implement ToolRouter
    if "ToolRouter" in mro_names:
        return "tool_router"

    # No fallback - adapters must follow the convention
    return "unknown"


def list_tools(namespace: str | None = None) -> list[dict[str, Any]]:
    """List all available tools.

    Parameters
    ----------
    namespace : str | None
        Optional filter by namespace (e.g., "core")

    Returns
    -------
    list[dict]
        List of tool info dicts with keys:
        - name: Tool function name
        - namespace: Component namespace (e.g., "core")
        - module_path: Full import path
        - description: Short description from docstring

    Examples
    --------
    >>> tools = list_tools()
    >>> any(t["name"] == "tool_end" for t in tools)
    True
    """
    from hexdag.builtin.tools import builtin_tools

    tools: list[dict[str, Any]] = []

    # Filter by namespace if specified (only core namespace for builtin)
    if namespace and namespace != "core":
        return tools

    # Scan builtin tools module
    for name in dir(builtin_tools):
        if name.startswith("_"):
            continue

        obj = getattr(builtin_tools, name, None)
        if obj is None:
            continue

        # Check if it's a callable (function or class)
        if not callable(obj):
            continue

        # Skip non-tool items
        if name in ("Any", "TypeVar", "TYPE_CHECKING"):
            continue

        tool_info = {
            "name": name,
            "namespace": "core",
            "module_path": f"hexdag.builtin.tools.{name}",
            "description": (obj.__doc__ or "No description available").split("\n")[0].strip(),
        }

        tools.append(tool_info)

    return sorted(tools, key=lambda x: x["name"])


def list_macros() -> list[dict[str, Any]]:
    """List all available macros.

    Macros are reusable pipeline templates that expand into subgraphs.

    Returns
    -------
    list[dict]
        List of macro info dicts with keys:
        - name: Macro class name
        - namespace: Component namespace (e.g., "core")
        - module_path: Full import path
        - description: Short description from docstring

    Examples
    --------
    >>> macros = list_macros()
    >>> any(m["name"] == "ReasoningAgentMacro" for m in macros)
    True
    """
    from hexdag.builtin import macros as builtin_macros

    macros: list[dict[str, Any]] = []

    # Scan builtin macros module
    for name in dir(builtin_macros):
        if name.startswith("_"):
            continue

        obj = getattr(builtin_macros, name, None)
        if obj is None or not isinstance(obj, type):
            continue

        # Check if it's a macro class (ends with Macro)
        if not name.endswith("Macro"):
            continue

        macro_info = {
            "name": name,
            "namespace": "core",
            "module_path": f"hexdag.builtin.macros.{name}",
            "description": (obj.__doc__ or "No description available").split("\n")[0].strip(),
        }
        macros.append(macro_info)

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


def get_component_schema(component_type: str, name: str, namespace: str = "core") -> dict[str, Any]:
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
    dict
        JSON Schema for the component, or error dict if not found

    Examples
    --------
    >>> schema = get_component_schema("node", "llm_node")
    >>> "properties" in schema or "error" in schema
    True
    """
    if component_type == "node":
        return _get_node_schema(name, namespace)
    if component_type == "adapter":
        return _get_adapter_schema(name, namespace)
    if component_type == "tool":
        return _get_tool_schema(name, namespace)
    if component_type == "macro":
        return _get_macro_schema(name, namespace)
    if component_type == "tag":
        return _get_tag_schema(name)
    return {"error": f"Unknown component type: {component_type}"}


def _get_node_schema(name: str, namespace: str) -> dict[str, Any]:
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


def _get_adapter_schema(name: str, namespace: str) -> dict[str, Any]:
    """Get schema for an adapter."""
    try:
        from hexdag.builtin import adapters as builtin_adapters

        cls = getattr(builtin_adapters, name, None)
        if cls is None:
            return {"error": f"Adapter '{name}' not found"}

        result = SchemaGenerator.from_callable(cls)
        if isinstance(result, dict):
            return result
        return {"error": f"Schema generator returned non-dict for adapter '{name}'"}
    except Exception as e:
        return {"error": f"Cannot generate schema for adapter '{name}': {e}"}


def _get_tool_schema(name: str, namespace: str) -> dict[str, Any]:
    """Get schema for a tool."""
    try:
        from hexdag.builtin.tools import builtin_tools

        fn = getattr(builtin_tools, name, None)
        if fn is None:
            return {"error": f"Tool '{name}' not found"}

        result = SchemaGenerator.from_callable(fn)
        if isinstance(result, dict):
            return result
        return {"error": f"Schema generator returned non-dict for tool '{name}'"}
    except Exception as e:
        return {"error": f"Cannot generate schema for tool '{name}': {e}"}


def _get_macro_schema(name: str, namespace: str) -> dict[str, Any]:
    """Get schema for a macro."""
    try:
        from hexdag.builtin import macros as builtin_macros

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
