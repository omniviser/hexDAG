"""Plugin management API for hexdag studio.

Discovers and manages hexdag plugins for use in pipelines.
"""

import importlib
import importlib.metadata
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/plugins", tags=["plugins"])


class PluginInfo(BaseModel):
    """Information about a plugin."""

    name: str
    version: str
    description: str
    module: str
    adapters: list[dict[str, Any]]
    nodes: list[dict[str, Any]]
    installed: bool
    enabled: bool


class PluginAdapter(BaseModel):
    """Information about a plugin adapter."""

    name: str
    port_type: str
    description: str
    config_schema: dict[str, Any]
    secrets: list[str]


class PluginNode(BaseModel):
    """Information about a plugin node."""

    kind: str
    name: str
    description: str
    config_schema: dict[str, Any]
    color: str


class PluginsResponse(BaseModel):
    """List of available plugins."""

    plugins: list[PluginInfo]


def discover_plugins() -> list[PluginInfo]:
    """Discover installed hexdag plugins.

    Looks for:
    1. Entry points with group 'hexdag.plugins'
    2. Packages matching 'hexdag-*' or 'hexdag_plugins.*'
    3. Local plugin directories in hexdag_plugins/
    """
    plugins = []

    # 1. Check for local plugins in hexdag_plugins directory
    plugins_dir = Path(__file__).parent.parent.parent.parent.parent / "hexdag_plugins"
    if plugins_dir.exists():
        for plugin_dir in plugins_dir.iterdir():
            if plugin_dir.is_dir() and not plugin_dir.name.startswith("_"):
                plugin_info = _load_local_plugin(plugin_dir)
                if plugin_info:
                    plugins.append(plugin_info)

    # 2. Check for installed packages via entry points
    try:
        eps = importlib.metadata.entry_points()
        if hasattr(eps, "select"):
            # Python 3.10+
            hexdag_eps = eps.select(group="hexdag.plugins")
        else:
            # Python 3.9
            hexdag_eps = eps.get("hexdag.plugins", [])

        for ep in hexdag_eps:
            try:
                plugin_module = ep.load()
                plugin_info = _create_plugin_info_from_module(ep.name, plugin_module)
                # Avoid duplicates
                if not any(p.name == plugin_info.name for p in plugins):
                    plugins.append(plugin_info)
            except Exception as e:
                print(f"Failed to load plugin {ep.name}: {e}")
    except Exception:
        pass

    return plugins


def _load_local_plugin(plugin_dir: Path) -> PluginInfo | None:
    """Load a plugin from a local directory."""
    plugin_name = plugin_dir.name

    # Check for __init__.py
    init_file = plugin_dir / "__init__.py"
    if not init_file.exists():
        return None

    # Try to import the module
    try:
        module_name = f"hexdag_plugins.{plugin_name}"

        # Add parent to path if needed
        parent_path = str(plugin_dir.parent)
        if parent_path not in sys.path:
            sys.path.insert(0, parent_path)

        module = importlib.import_module(module_name)
        return _create_plugin_info_from_module(plugin_name, module)

    except Exception as e:
        print(f"Failed to load local plugin {plugin_name}: {e}")
        return None


def _create_plugin_info_from_module(name: str, module: Any) -> PluginInfo:
    """Create PluginInfo from a loaded module."""
    # Get version
    version = getattr(module, "__version__", "0.0.0")

    # Get description from docstring
    description = (module.__doc__ or "").strip().split("\n")[0]

    # Discover adapters
    adapters = _discover_adapters(module)

    # Discover nodes
    nodes = _discover_nodes(module)

    return PluginInfo(
        name=name,
        version=version,
        description=description,
        module=module.__name__,
        adapters=adapters,
        nodes=nodes,
        installed=True,
        enabled=True,
    )


def _discover_adapters(module: Any) -> list[dict[str, Any]]:
    """Discover adapter classes in a module."""
    adapters = []

    # Check __all__ for exported names
    exported = getattr(module, "__all__", [])

    for name in exported:
        obj = getattr(module, name, None)
        if obj is None:
            continue

        # Check if it's an adapter (using hexdag decorator attributes)
        hexdag_type = getattr(obj, "_hexdag_type", None)
        if (
            hexdag_type is not None
            and str(hexdag_type.value if hasattr(hexdag_type, "value") else hexdag_type)
            == "adapter"
        ):
            port_type = getattr(obj, "_hexdag_implements_port", "unknown")
            secrets_dict = getattr(obj, "_hexdag_secrets", {})
            adapters.append({
                "name": getattr(obj, "_hexdag_name", name),
                "port_type": port_type,
                "description": getattr(
                    obj, "_hexdag_description", (obj.__doc__ or "").strip().split("\n")[0]
                ),
                "config_schema": _extract_config_schema(obj),
                "secrets": list(secrets_dict.keys()) if secrets_dict else [],
            })
        # Legacy check for _hexdag_adapter_metadata (backward compatibility)
        elif metadata := getattr(obj, "_hexdag_adapter_metadata", None):
            adapters.append({
                "name": metadata.get("name", name),
                "port_type": metadata.get("port_type", "unknown"),
                "description": (obj.__doc__ or "").strip().split("\n")[0],
                "config_schema": _extract_config_schema(obj),
                "secrets": list(metadata.get("secrets", {}).keys()),
            })
        elif name.endswith("Adapter"):
            # Fallback: treat classes ending in Adapter as adapters
            port_type = _guess_port_type(name)
            adapters.append({
                "name": name,
                "port_type": port_type,
                "description": (obj.__doc__ or "").strip().split("\n")[0],
                "config_schema": _extract_config_schema(obj),
                "secrets": [],
            })

    return adapters


def _discover_nodes(module: Any) -> list[dict[str, Any]]:
    """Discover node classes in a module."""
    nodes = []

    # Check __all__ for exported names
    exported = getattr(module, "__all__", [])

    for name in exported:
        obj = getattr(module, name, None)
        if obj is None:
            continue

        # Check if it's a node (using hexdag decorator attributes)
        hexdag_type = getattr(obj, "_hexdag_type", None)
        if (
            hexdag_type is not None
            and str(hexdag_type.value if hasattr(hexdag_type, "value") else hexdag_type) == "node"
        ):
            # Get namespace to construct full kind
            namespace = getattr(obj, "_hexdag_namespace", "core")
            node_name = getattr(obj, "_hexdag_name", _to_snake_case(name))
            # Use namespace:name format for plugin nodes (not core)
            kind = f"{namespace}:{node_name}" if namespace != "core" else node_name
            nodes.append({
                "kind": kind,
                "name": name,
                "namespace": namespace,
                "description": getattr(
                    obj, "_hexdag_description", (obj.__doc__ or "").strip().split("\n")[0]
                ),
                "config_schema": _extract_config_schema(obj),
                "color": "#6b7280",  # Default color - can be extended later
            })
        # Legacy check for _hexdag_node_metadata (backward compatibility)
        elif metadata := getattr(obj, "_hexdag_node_metadata", None):
            nodes.append({
                "kind": metadata.get("kind", name),
                "name": metadata.get("name", name),
                "description": (obj.__doc__ or "").strip().split("\n")[0],
                "config_schema": _extract_config_schema(obj),
                "color": metadata.get("color", "#6b7280"),
            })
        elif name.endswith("Node"):
            # Fallback: treat classes ending in Node as nodes
            nodes.append({
                "kind": _to_snake_case(name),
                "name": name,
                "description": (obj.__doc__ or "").strip().split("\n")[0],
                "config_schema": _extract_config_schema(obj),
                "color": "#6b7280",
            })

    return nodes


def _extract_config_schema(cls: type) -> dict[str, Any]:
    """Extract configuration schema from a class __init__ signature."""
    import inspect

    schema = {"type": "object", "properties": {}, "required": []}

    try:
        sig = inspect.signature(cls.__init__)
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "args", "kwargs"):
                continue

            prop: dict[str, Any] = {}

            # Get type annotation
            if param.annotation != inspect.Parameter.empty:
                prop["type"] = _python_type_to_json_type(param.annotation)

            # Get default value
            if param.default != inspect.Parameter.empty:
                prop["default"] = param.default
            else:
                schema["required"].append(param_name)

            schema["properties"][param_name] = prop

    except Exception:
        pass

    return schema


def _python_type_to_json_type(py_type: Any) -> str:
    """Convert Python type annotation to JSON schema type."""
    type_str = str(py_type)

    if "str" in type_str:
        return "string"
    if "int" in type_str:
        return "integer"
    if "float" in type_str:
        return "number"
    if "bool" in type_str:
        return "boolean"
    if "list" in type_str or "List" in type_str:
        return "array"
    if "dict" in type_str or "Dict" in type_str:
        return "object"
    return "string"


def _guess_port_type(adapter_name: str) -> str:
    """Guess port type from adapter name."""
    name_lower = adapter_name.lower()
    if "openai" in name_lower or "llm" in name_lower or "anthropic" in name_lower:
        return "llm"
    if "memory" in name_lower or "cosmos" in name_lower:
        return "memory"
    if "storage" in name_lower or "blob" in name_lower or "s3" in name_lower:
        return "storage"
    if "keyvault" in name_lower or "secret" in name_lower:
        return "secret"
    if "database" in name_lower or "sql" in name_lower:
        return "database"
    return "unknown"


def _to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case."""
    import re

    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


# ===== API Endpoints =====
# NOTE: Specific routes MUST come before parameterized routes to avoid matching issues


@router.get("", response_model=PluginsResponse)
async def list_plugins() -> PluginsResponse:
    """List all available plugins."""
    plugins = discover_plugins()
    return PluginsResponse(plugins=plugins)


@router.get("/adapters/all")
async def get_all_adapters() -> list[dict[str, Any]]:
    """Get all adapters from all plugins."""
    plugins = discover_plugins()
    adapters = []

    for plugin in plugins:
        for adapter in plugin.adapters:
            adapter_copy = dict(adapter)
            adapter_copy["plugin"] = plugin.name
            adapters.append(adapter_copy)

    return adapters


@router.get("/nodes/all")
async def get_all_nodes() -> list[dict[str, Any]]:
    """Get all nodes from all plugins."""
    plugins = discover_plugins()
    nodes = []

    for plugin in plugins:
        for node in plugin.nodes:
            node_copy = dict(node)
            node_copy["plugin"] = plugin.name
            nodes.append(node_copy)

    return nodes


@router.get("/{plugin_name}")
async def get_plugin(plugin_name: str) -> PluginInfo:
    """Get details about a specific plugin."""
    plugins = discover_plugins()

    for plugin in plugins:
        if plugin.name == plugin_name:
            return plugin

    return PluginInfo(
        name=plugin_name,
        version="0.0.0",
        description="Plugin not found",
        module="",
        adapters=[],
        nodes=[],
        installed=False,
        enabled=False,
    )


@router.get("/{plugin_name}/adapters")
async def get_plugin_adapters(plugin_name: str) -> list[dict[str, Any]]:
    """Get adapters provided by a plugin."""
    plugins = discover_plugins()

    for plugin in plugins:
        if plugin.name == plugin_name:
            return plugin.adapters

    return []


@router.get("/{plugin_name}/nodes")
async def get_plugin_nodes(plugin_name: str) -> list[dict[str, Any]]:
    """Get nodes provided by a plugin."""
    plugins = discover_plugins()

    for plugin in plugins:
        if plugin.name == plugin_name:
            return plugin.nodes

    return []
