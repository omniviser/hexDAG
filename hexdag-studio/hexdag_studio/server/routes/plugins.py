"""Plugin management API for hexdag studio.

Uses hexdag.api.components as the single source of truth for component discovery.
This ensures no duplicate components between plugins and registry endpoints.
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from hexdag import api
from hexdag.kernel.discovery import (
    get_user_plugin_paths as core_get_user_plugin_paths,
)
from hexdag.kernel.discovery import (
    set_user_plugin_paths as core_set_user_plugin_paths,
)

router = APIRouter(prefix="/plugins", tags=["plugins"])

# Module-level plugin configuration (for subdirs expansion)
_with_subdirs: bool = False


def set_plugin_paths(paths: list[Path], with_subdirs: bool = False) -> None:
    """Set plugin paths from CLI - delegates to core discovery.

    Parameters
    ----------
    paths : list[Path]
        List of plugin directory paths
    with_subdirs : bool
        If True, expand paths to include subdirectories
    """
    global _with_subdirs
    _with_subdirs = with_subdirs

    # Expand paths if with_subdirs is True
    expanded: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        if with_subdirs and path.is_dir():
            expanded.extend(d for d in path.iterdir() if d.is_dir() and not d.name.startswith("_"))
        else:
            expanded.append(path)

    # Delegate to core discovery
    core_set_user_plugin_paths(expanded)


def get_plugin_paths() -> list[Path]:
    """Get current plugin paths from core discovery."""
    return core_get_user_plugin_paths()


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


# ===== Plugin Discovery (Single Source of Truth) =====


def _module_path_to_plugin_name(module_path: str) -> str | None:
    """Extract plugin name from module_path.

    Returns None for built-in components (hexdag.stdlib.*).

    Examples:
        "hexdag.stdlib.nodes.LLMNode" -> None (builtin)
        "hexdag_plugins.mysql_adapter.MySQLAdapter" -> "mysql_adapter"
        "acme.adapters.CustomAdapter" -> "acme"
    """
    if module_path.startswith("hexdag.stdlib."):
        return None  # Built-in, not a plugin

    if module_path.startswith("hexdag_plugins."):
        # Extract plugin name: hexdag_plugins.<plugin_name>.*
        parts = module_path.split(".")
        if len(parts) >= 2:
            return parts[1]
        return None

    # User plugin - use top-level package name
    parts = module_path.split(".")
    if parts:
        return parts[0]
    return None


def _create_empty_plugin(name: str) -> PluginInfo:
    """Create an empty PluginInfo for a plugin name."""
    return PluginInfo(
        name=name,
        version="0.0.0",
        description="",
        module="",
        adapters=[],
        nodes=[],
    )


def discover_plugins() -> list[PluginInfo]:
    """Discover plugins using api.components as the single source of truth.

    Groups components by their module_path to identify plugins.
    All discovery is delegated to hexdag.api.components which handles:
    - Built-in components (hexdag.stdlib.*) - excluded from plugins
    - Plugin components (hexdag_plugins.*)
    - User plugin paths (HEXDAG_PLUGIN_PATHS env var)
    - User-configured modules (hexdag.toml/pyproject.toml)

    Returns
    -------
    list[PluginInfo]
        List of discovered plugins (excludes core/built-in components)
    """
    plugins: dict[str, PluginInfo] = {}

    # Get all adapters and nodes from the unified API (already deduplicated)
    all_adapters = api.components.list_adapters()
    all_nodes = api.components.list_nodes()

    # Group adapters by plugin (determined from module_path)
    for adapter in all_adapters:
        module_path = adapter.get("module_path", "")
        plugin_name = _module_path_to_plugin_name(module_path)
        if plugin_name is None:
            continue  # Skip built-in

        if plugin_name not in plugins:
            plugins[plugin_name] = _create_empty_plugin(plugin_name)
        plugins[plugin_name].adapters.append(adapter)

    # Group nodes by plugin (determined from module_path)
    for node in all_nodes:
        module_path = node.get("module_path", "")
        plugin_name = _module_path_to_plugin_name(module_path)
        if plugin_name is None:
            continue  # Skip built-in

        if plugin_name not in plugins:
            plugins[plugin_name] = _create_empty_plugin(plugin_name)
        plugins[plugin_name].nodes.append(node)

    return list(plugins.values())


# ===== API Endpoints =====


@router.get("", response_model=PluginsResponse)
async def list_plugins() -> PluginsResponse:
    """List all plugins.

    Uses api.components as the single source of truth for discovery.
    """
    return PluginsResponse(plugins=discover_plugins())


@router.get("/adapters/all")
async def get_all_adapters() -> list[dict[str, Any]]:
    """Get all adapters (built-in + plugins) from unified API.

    Uses api.components.list_adapters() as the single source of truth.
    No duplicate discovery - just adds plugin name metadata.
    """
    all_adapters = api.components.list_adapters()
    return [
        {
            **adapter,
            "plugin": _module_path_to_plugin_name(adapter.get("module_path", "")) or "builtin",
        }
        for adapter in all_adapters
    ]


@router.get("/nodes/all")
async def get_all_nodes() -> list[dict[str, Any]]:
    """Get all plugin nodes (excludes core built-in nodes).

    Uses api.components.list_nodes() as the single source of truth.
    Returns only non-core nodes to avoid duplicates with /registry/nodes.
    """
    all_nodes = api.components.list_nodes()
    result = []
    for node in all_nodes:
        module_path = node.get("module_path", "")
        plugin_name = _module_path_to_plugin_name(module_path)
        if plugin_name is not None:  # Not built-in
            result.append({**node, "plugin": plugin_name})
    return result


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
    """Get adapters from a specific plugin."""
    for plugin in discover_plugins():
        if plugin.name == plugin_name:
            return plugin.adapters
    return []


@router.get("/{plugin_name}/nodes")
async def get_plugin_nodes(plugin_name: str) -> list[dict[str, Any]]:
    """Get nodes from a specific plugin."""
    for plugin in discover_plugins():
        if plugin.name == plugin_name:
            return plugin.nodes
    return []
