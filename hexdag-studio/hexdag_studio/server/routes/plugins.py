"""Plugin management API for hexdag studio."""

import importlib
import importlib.metadata
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from hexdag import api
from hexdag.core.schema import SchemaGenerator

router = APIRouter(prefix="/plugins", tags=["plugins"])

# Module-level plugin configuration
_plugin_paths: list[Path] = []
_with_subdirs: bool = False


def set_plugin_paths(paths: list[Path], with_subdirs: bool = False) -> None:
    """Set plugin paths from CLI."""
    global _plugin_paths, _with_subdirs
    _plugin_paths = [p.resolve() for p in paths]
    _with_subdirs = with_subdirs


def get_plugin_paths() -> list[Path]:
    """Get current plugin paths."""
    return _plugin_paths.copy()


# ===== Dependency Installation =====


def _detect_package_manager() -> str:
    """Detect uv or pip."""
    if not shutil.which("uv"):
        return "pip"
    try:
        result = subprocess.run(
            ["uv", "pip", "list", "-q"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return "uv"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "pip"


def install_plugin_dependencies(paths: list[Path], with_subdirs: bool = False) -> None:
    """Install plugin dependencies from pyproject.toml files."""
    plugin_dirs = _collect_plugin_dirs(paths, with_subdirs)
    if not plugin_dirs:
        return

    pkg_manager = _detect_package_manager()
    print(f"Using {pkg_manager} for dependency installation...")

    for plugin_dir in plugin_dirs:
        if not (plugin_dir / "pyproject.toml").exists():
            print(f"  Skipping '{plugin_dir.name}' (no pyproject.toml)")
            continue

        print(f"  Installing '{plugin_dir.name}'...")
        try:
            cmd = (
                ["uv", "pip", "install", "-e", str(plugin_dir), "-q"]
                if pkg_manager == "uv"
                else [sys.executable, "-m", "pip", "install", "-e", str(plugin_dir), "-q"]
            )
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"    Warning: {result.stderr.strip()}")
            else:
                print("    Done.")
        except Exception as e:
            print(f"    Warning: Failed to install: {e}")


def _collect_plugin_dirs(paths: list[Path], with_subdirs: bool) -> list[Path]:
    """Collect plugin directories from paths."""
    dirs: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        if with_subdirs:
            dirs.extend(d for d in path.iterdir() if d.is_dir() and not d.name.startswith("_"))
        else:
            dirs.append(path)
    return dirs


# ===== Models =====


class PluginInfo(BaseModel):
    """Plugin information."""

    name: str
    version: str
    description: str
    module: str
    adapters: list[dict[str, Any]]
    nodes: list[dict[str, Any]]
    installed: bool = True
    enabled: bool = True


class PluginsResponse(BaseModel):
    """List of plugins."""

    plugins: list[PluginInfo]


# ===== Plugin Discovery =====


def discover_plugins() -> list[PluginInfo]:
    """Discover plugins from paths or default directory."""
    plugins: list[PluginInfo] = []
    seen: set[str] = set()

    def add_plugin(info: PluginInfo | None) -> None:
        if info and info.name not in seen:
            plugins.append(info)
            seen.add(info.name)

    # Load from configured paths or default directory
    if _plugin_paths:
        for path in _plugin_paths:
            if not path.exists():
                continue
            if _with_subdirs:
                for subdir in path.iterdir():
                    if subdir.is_dir() and not subdir.name.startswith("_"):
                        add_plugin(_load_plugin(subdir, path))
            else:
                add_plugin(_load_plugin(path, path.parent))
    else:
        # Default: hexdag_plugins directory
        default_dir = Path(__file__).parent.parent.parent.parent.parent / "hexdag_plugins"
        if default_dir.exists():
            for subdir in default_dir.iterdir():
                if subdir.is_dir() and not subdir.name.startswith("_"):
                    add_plugin(_load_plugin(subdir, default_dir))

    # Entry points
    try:
        eps = importlib.metadata.entry_points()
        hexdag_eps = (
            eps.select(group="hexdag.plugins")
            if hasattr(eps, "select")
            else eps.get("hexdag.plugins", [])
        )
        for ep in hexdag_eps:
            try:
                add_plugin(_module_to_plugin_info(ep.name, ep.load()))
            except Exception as e:
                print(f"Failed to load plugin {ep.name}: {e}")
    except Exception:
        pass

    return plugins


def _load_plugin(plugin_dir: Path, parent_path: Path) -> PluginInfo | None:
    """Load a plugin from a directory."""
    name = plugin_dir.name
    if not (plugin_dir / "__init__.py").exists():
        return None

    try:
        module_name = name
        path_to_add = str(parent_path)
        if path_to_add not in sys.path:
            sys.path.insert(0, path_to_add)
        module = importlib.import_module(module_name)
        return _module_to_plugin_info(name, module)
    except ModuleNotFoundError as e:
        missing = str(e).replace("No module named ", "").strip("'")
        print(f"Plugin '{name}' requires missing dependency: {missing}")
        print(f"  Install with: pip install {missing}")
        return None
    except Exception as e:
        print(f"Failed to load plugin {name}: {e}")
        return None


def _module_to_plugin_info(name: str, module: Any) -> PluginInfo:
    """Create PluginInfo from a module."""
    return PluginInfo(
        name=name,
        version=getattr(module, "__version__", "0.0.0"),
        description=(module.__doc__ or "").strip().split("\n")[0],
        module=module.__name__,
        adapters=_discover_adapters(module),
        nodes=_discover_nodes(module),
    )


def _discover_adapters(module: Any) -> list[dict[str, Any]]:
    """Find adapters in a module."""
    adapters = []
    for name in getattr(module, "__all__", []):
        obj = getattr(module, name, None)
        if obj is None:
            continue

        hexdag_type = getattr(obj, "_hexdag_type", None)
        type_str = (
            str(hexdag_type.value if hasattr(hexdag_type, "value") else hexdag_type)
            if hexdag_type
            else None
        )

        if type_str == "adapter":
            adapters.append({
                "name": getattr(obj, "_hexdag_name", name),
                "port_type": getattr(obj, "_hexdag_implements_port", "unknown"),
                "description": getattr(
                    obj, "_hexdag_description", (obj.__doc__ or "").split("\n")[0].strip()
                ),
                "config_schema": _get_schema(obj),
                "secrets": list(getattr(obj, "_hexdag_secrets", {}).keys()),
            })
        elif name.endswith("Adapter") and isinstance(obj, type):
            adapters.append({
                "name": name,
                "port_type": _detect_port_type(obj),
                "description": (obj.__doc__ or "").split("\n")[0].strip(),
                "config_schema": _get_schema(obj),
                "secrets": [],
            })
    return adapters


def _discover_nodes(module: Any) -> list[dict[str, Any]]:
    """Find nodes in a module."""
    nodes = []
    for name in getattr(module, "__all__", []):
        obj = getattr(module, name, None)
        if obj is None:
            continue

        hexdag_type = getattr(obj, "_hexdag_type", None)
        type_str = (
            str(hexdag_type.value if hasattr(hexdag_type, "value") else hexdag_type)
            if hexdag_type
            else None
        )

        if type_str == "node":
            namespace = getattr(obj, "_hexdag_namespace", "core")
            node_name = getattr(obj, "_hexdag_name", _to_snake_case(name))
            kind = f"{namespace}:{node_name}" if namespace != "core" else node_name
            nodes.append({
                "kind": kind,
                "name": name,
                "namespace": namespace,
                "description": getattr(
                    obj, "_hexdag_description", (obj.__doc__ or "").split("\n")[0].strip()
                ),
                "config_schema": _get_node_schema(obj),
                "color": _get_node_color(obj),
                "icon": _get_node_icon(obj),
            })
        elif name.endswith("Node") and isinstance(obj, type):
            kind = _to_snake_case(name)
            nodes.append({
                "kind": kind,
                "name": name,
                "description": (obj.__doc__ or "").split("\n")[0].strip(),
                "config_schema": _get_node_schema(obj),
                "color": _get_node_color(obj),
                "icon": _get_node_icon(obj),
            })
    return nodes


def _get_schema(cls: type) -> dict[str, Any]:
    """Extract config schema from class __init__."""
    try:
        return SchemaGenerator.from_callable(cls.__init__)
    except Exception:
        return {"type": "object", "properties": {}}


def _detect_port_type(adapter_class: type) -> str:
    """Detect port type via API or name heuristics."""
    try:
        return api.components.detect_port_type(adapter_class)
    except Exception:
        name = adapter_class.__name__.lower()
        patterns = {
            "llm": ["llm", "openai", "anthropic", "claude", "gpt"],
            "memory": ["memory", "redis", "cosmos"],
            "database": ["sql", "mysql", "postgres", "database", "db"],
            "secret": ["secret", "vault", "keyvault"],
            "storage": ["storage", "blob", "file", "s3"],
            "vector_store": ["vector", "embedding", "chroma", "pgvector"],
            "tool_router": ["tool", "router"],
        }
        for port_type, keywords in patterns.items():
            if any(kw in name for kw in keywords):
                return port_type
        return "unknown"


def _to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _get_node_icon(cls: type) -> str:
    """Get icon for a node class.

    Reads _hexdag_icon attribute from the class. Returns "Package" as default.

    Plugins/nodes can define icons on their classes:
        class FileReaderNode:
            _hexdag_icon = "FileInput"  # Lucide icon name
    """
    return getattr(cls, "_hexdag_icon", "Package")


def _get_node_color(cls: type) -> str:
    """Get color for a node class.

    Reads _hexdag_color attribute from the class. Returns gray as default.

    Plugins/nodes can define colors on their classes:
        class FileReaderNode:
            _hexdag_color = "#10b981"  # Hex color
    """
    return getattr(cls, "_hexdag_color", "#6b7280")


def _get_node_schema(cls: type) -> dict[str, Any]:
    """Extract config schema from node class.

    For node factories, uses __call__ method (where config params are defined).
    Falls back to __init__ for other node types.
    """
    try:
        # Check if __call__ is overridden on this class (not just inherited from object)
        for klass in cls.__mro__:
            if klass is object:
                continue
            if "__call__" in klass.__dict__:
                schema = SchemaGenerator.from_callable(cls.__call__)
                if schema.get("properties"):
                    return schema
                break
        # Fall back to __init__
        return SchemaGenerator.from_callable(cls.__init__)
    except Exception:
        return {"type": "object", "properties": {}}


# ===== API Endpoints =====


@router.get("", response_model=PluginsResponse)
async def list_plugins() -> PluginsResponse:
    """List all plugins."""
    return PluginsResponse(plugins=discover_plugins())


@router.get("/adapters/all")
async def get_all_adapters() -> list[dict[str, Any]]:
    """Get all adapters from plugins and built-ins."""
    adapters = [{**a, "plugin": "builtin"} for a in api.components.list_adapters()]
    for plugin in discover_plugins():
        adapters.extend({**a, "plugin": plugin.name} for a in plugin.adapters)
    return adapters


@router.get("/nodes/all")
async def get_all_nodes() -> list[dict[str, Any]]:
    """Get all nodes from plugins."""
    nodes = []
    for plugin in discover_plugins():
        nodes.extend({**n, "plugin": plugin.name} for n in plugin.nodes)
    return nodes


@router.get("/{plugin_name}")
async def get_plugin(plugin_name: str) -> PluginInfo:
    """Get a specific plugin."""
    for plugin in discover_plugins():
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
    """Get adapters from a plugin."""
    for plugin in discover_plugins():
        if plugin.name == plugin_name:
            return plugin.adapters
    return []


@router.get("/{plugin_name}/nodes")
async def get_plugin_nodes(plugin_name: str) -> list[dict[str, Any]]:
    """Get nodes from a plugin."""
    for plugin in discover_plugins():
        if plugin.name == plugin_name:
            return plugin.nodes
    return []
