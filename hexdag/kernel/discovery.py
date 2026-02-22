"""Dynamic component discovery utilities.

This module provides functions to discover components (adapters, tools, macros)
dynamically using Python's pkgutil and importlib.metadata, eliminating the need
for hardcoded component lists.

The discovery system supports three levels:
1. Builtin components from hexdag.stdlib.*
2. Plugin components from hexdag_plugins.*
3. User-authored components from:
   - Paths configured via set_user_plugin_paths()
   - Environment variable HEXDAG_PLUGIN_PATHS
   - Modules configured in hexdag.toml/pyproject.toml

Examples
--------
>>> from hexdag.kernel.discovery import discover_plugins, discover_modules
>>> plugins = discover_plugins()
>>> modules = discover_modules("hexdag.stdlib.adapters")

>>> # User plugins via environment variable
>>> import os
>>> os.environ["HEXDAG_PLUGIN_PATHS"] = "./my_adapters:./my_nodes"
>>> user_plugins = discover_user_plugins()
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

# ============================================================================
# User Plugin Path Configuration
# ============================================================================

# Module-level storage for user-configured plugin paths
_user_plugin_paths: list[Path] = []


def set_user_plugin_paths(paths: list[Path]) -> None:
    """Configure user plugin paths for discovery.

    Called by Studio CLI, MCP server, or user code on startup.
    Clears the discovery cache to force re-discovery.

    Parameters
    ----------
    paths : list[Path]
        List of paths to plugin directories

    Examples
    --------
    >>> from pathlib import Path
    >>> set_user_plugin_paths([Path("./my_adapters"), Path("./my_nodes")])
    """
    global _user_plugin_paths
    _user_plugin_paths = [p.resolve() for p in paths if p.exists()]
    clear_discovery_cache()


def get_user_plugin_paths() -> list[Path]:
    """Get configured user plugin paths.

    Combines programmatically set paths with HEXDAG_PLUGIN_PATHS env var.

    Returns
    -------
    list[Path]
        List of plugin paths (deduplicated)

    Examples
    --------
    >>> import os
    >>> os.environ["HEXDAG_PLUGIN_PATHS"] = "/path/to/plugins"
    >>> paths = get_user_plugin_paths()
    """
    paths: list[Path] = list(_user_plugin_paths)

    # Also check environment variable
    env_paths = os.environ.get("HEXDAG_PLUGIN_PATHS", "")
    if env_paths:
        for p in env_paths.split(os.pathsep):
            path = Path(p).resolve()
            if path.exists() and path not in paths:
                paths.append(path)

    return paths


@lru_cache(maxsize=1)
def discover_modules(package_path: str) -> list[str]:
    """Discover all modules in a package recursively.

    Parameters
    ----------
    package_path : str
        Dotted path to the package (e.g., "hexdag.stdlib.adapters")

    Returns
    -------
    list[str]
        List of fully qualified module paths

    Examples
    --------
    >>> modules = discover_modules("hexdag.stdlib.adapters")
    >>> isinstance(modules, list)
    True
    """
    try:
        package = importlib.import_module(package_path)
        if not hasattr(package, "__path__"):
            return [package_path]  # Single module, not a package

        modules = []
        for _finder, name, _ispkg in pkgutil.walk_packages(
            package.__path__, prefix=f"{package_path}."
        ):
            modules.append(name)
        return modules
    except ImportError:
        return []


@lru_cache(maxsize=1)
def discover_plugins() -> list[str]:
    """Discover installed hexdag plugins.

    Scans the hexdag_plugins namespace package for available plugins.

    Returns
    -------
    list[str]
        List of plugin names (e.g., ["azure", "hexdag_etl", "storage"])

    Examples
    --------
    >>> plugins = discover_plugins()
    >>> isinstance(plugins, list)
    True
    """
    plugins = []
    try:
        import hexdag_plugins

        for _finder, name, ispkg in pkgutil.iter_modules(hexdag_plugins.__path__):
            if ispkg:  # Only include packages, not loose modules
                plugins.append(name)
    except ImportError:
        pass
    return plugins


def discover_classes_in_module(
    module_path: str,
    base_class: type | None = None,
    suffix: str | None = None,
) -> list[tuple[str, type]]:
    """Discover classes in a module matching criteria.

    Parameters
    ----------
    module_path : str
        Dotted path to the module
    base_class : type | None
        If provided, only return subclasses of this type
    suffix : str | None
        If provided, only return classes whose names end with this suffix

    Returns
    -------
    list[tuple[str, type]]
        List of (class_name, class_object) tuples

    Examples
    --------
    >>> classes = discover_classes_in_module(
    ...     "hexdag.stdlib.adapters.mock.mock_llm",
    ...     suffix="LLM"
    ... )
    >>> isinstance(classes, list)
    True
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        return []

    classes = []
    for name in dir(module):
        if name.startswith("_"):
            continue

        obj = getattr(module, name, None)
        if obj is None or not isinstance(obj, type):
            continue

        # Check suffix filter
        if suffix and not name.endswith(suffix):
            continue

        # Check base class filter
        if base_class:
            try:
                if not issubclass(obj, base_class):
                    continue
            except TypeError:
                # issubclass raised TypeError (e.g., for non-class objects)
                continue

        classes.append((name, obj))

    return classes


def discover_adapters_in_package(
    package_path: str,
    detect_port_type_fn: Any = None,
) -> list[dict[str, Any]]:
    """Discover all adapter classes in a package.

    Parameters
    ----------
    package_path : str
        Dotted path to the adapters package
    detect_port_type_fn : callable | None
        Function to detect port type from adapter class.
        If None, uses hexdag.api.components.detect_port_type

    Returns
    -------
    list[dict[str, Any]]
        List of adapter info dicts with keys:
        - name: Adapter class name
        - module_path: Full import path (unique identifier)
        - port_type: Detected port type
        - description: First line of docstring
        - config_schema: JSON Schema for __init__
        - secrets: List of secret parameter names
    """
    # Lazy import to avoid circular dependency
    if detect_port_type_fn is None:
        from hexdag.api.components import detect_port_type

        detect_port_type_fn = detect_port_type

    from hexdag.kernel.schema import SchemaGenerator

    adapters = []
    # Use class identity (id) to deduplicate - same class may be imported in multiple modules
    seen_classes: set[int] = set()

    for module_path in discover_modules(package_path):
        # Discover all classes, then filter by suffix
        for name, cls in discover_classes_in_module(module_path):
            # Check for adapter-like class names
            if not name.endswith(("Adapter", "Memory", "LLM")):
                continue

            # Deduplicate by class identity (same class imported in multiple places)
            class_id = id(cls)
            if class_id in seen_classes:
                continue
            seen_classes.add(class_id)

            # Use the class's actual module, not where we found it
            actual_module = getattr(cls, "__module__", module_path)
            full_path = f"{actual_module}.{name}"

            # Detect port type
            try:
                port_type = detect_port_type_fn(cls)
            except Exception:
                port_type = "unknown"

            # Get secrets from decorator
            secrets_dict = getattr(cls, "_hexdag_secrets", {})
            secrets = list(secrets_dict.keys()) if secrets_dict else []

            # Generate config schema
            try:
                schema = SchemaGenerator.from_callable(cls.__init__)  # type: ignore[misc]
                config_schema = (
                    schema if isinstance(schema, dict) else {"type": "object", "properties": {}}
                )
            except Exception:
                config_schema = {"type": "object", "properties": {}}

            adapters.append({
                "name": name,
                "module_path": full_path,
                "port_type": port_type,
                "description": (cls.__doc__ or "No description").split("\n")[0].strip(),
                "config_schema": config_schema,
                "secrets": secrets,
            })

    return adapters


def discover_tools_in_module(module_path: str) -> list[dict[str, Any]]:
    """Discover tool functions in a module.

    Parameters
    ----------
    module_path : str
        Dotted path to the tools module

    Returns
    -------
    list[dict[str, Any]]
        List of tool info dicts with keys:
        - name: Tool function name
        - module_path: Full import path (unique identifier)
        - description: First line of docstring
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        return []

    tools = []
    for name in dir(module):
        if name.startswith("_"):
            continue

        obj = getattr(module, name, None)
        if obj is None or not callable(obj):
            continue

        # Skip common non-tool items
        if name in ("Any", "TypeVar", "TYPE_CHECKING"):
            continue

        # Skip classes (tools are functions)
        if isinstance(obj, type):
            continue

        tools.append({
            "name": name,
            "module_path": f"{module_path}.{name}",
            "description": (obj.__doc__ or "No description").split("\n")[0].strip(),
        })

    return tools


def discover_macros_in_module(module_path: str) -> list[dict[str, Any]]:
    """Discover macro classes in a module.

    Parameters
    ----------
    module_path : str
        Dotted path to the macros module

    Returns
    -------
    list[dict[str, Any]]
        List of macro info dicts with keys:
        - name: Macro class name
        - module_path: Full import path (unique identifier)
        - description: First line of docstring
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        return []

    macros = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        if not name.endswith("Macro"):
            continue

        obj = getattr(module, name, None)
        if obj is None or not isinstance(obj, type):
            continue

        macros.append({
            "name": name,
            "module_path": f"{module_path}.{name}",
            "description": (obj.__doc__ or "No description").split("\n")[0].strip(),
        })

    return macros


@lru_cache(maxsize=1)
def discover_user_modules() -> list[str]:
    """Discover user-configured modules from hexdag.toml or pyproject.toml.

    Reads the `modules` configuration from the project's hexdag configuration
    and returns modules that are not builtin or plugins (user-authored).

    Returns
    -------
    list[str]
        List of user module paths (e.g., ["myapp.adapters", "myapp.nodes"])

    Examples
    --------
    >>> # With hexdag.toml containing:
    >>> # modules = ["myapp.adapters", "myapp.nodes"]
    >>> user_modules = discover_user_modules()
    >>> isinstance(user_modules, list)
    True
    """
    try:
        from hexdag.kernel.config.loader import load_config
    except ImportError:
        return []

    try:
        config = load_config()
    except FileNotFoundError:
        return []

    user_modules = []
    for module in config.modules:
        # Skip builtin and plugin modules - only return user-authored
        if module.startswith("hexdag."):
            continue
        if module.startswith("hexdag_plugins."):
            continue
        user_modules.append(module)

    return user_modules


def get_user_namespace(module_path: str) -> str:
    """Get the namespace for a user module.

    Extracts a meaningful namespace from the module path.
    For example, "myapp.adapters.custom" -> "myapp"

    Parameters
    ----------
    module_path : str
        The full module path

    Returns
    -------
    str
        The namespace (top-level package name or "user")
    """
    parts = module_path.split(".")
    if parts:
        return parts[0]
    return "user"


def discover_user_plugins() -> list[dict[str, Any]]:
    """Discover plugins from user-configured paths.

    Scans paths from:
    - set_user_plugin_paths() (programmatic)
    - HEXDAG_PLUGIN_PATHS environment variable

    Each path is treated as a plugin directory. The directory name becomes
    the plugin name. Adapters are discovered from classes in __all__ or
    by naming convention (ends with Adapter, Memory, LLM).

    Returns
    -------
    list[dict[str, Any]]
        List of plugin info dicts with keys:
        - name: Plugin name (directory name)
        - module: Module name
        - adapters: List of adapter dicts
        - nodes: List of node dicts

    Examples
    --------
    >>> import os
    >>> os.environ["HEXDAG_PLUGIN_PATHS"] = "./acme_adapters"
    >>> plugins = discover_user_plugins()  # doctest: +SKIP
    >>> [p["name"] for p in plugins]  # doctest: +SKIP
    ['acme_adapters']
    """
    plugins: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for path in get_user_plugin_paths():
        if not path.is_dir():
            continue

        plugin_name = path.name

        # Skip if we've seen this plugin name already
        if plugin_name in seen_names:
            continue
        seen_names.add(plugin_name)

        # Skip if no __init__.py (not a package)
        if not (path / "__init__.py").exists():
            continue

        # Add parent to sys.path temporarily for import
        parent_path = str(path.parent)
        path_added = parent_path not in sys.path
        if path_added:
            sys.path.insert(0, parent_path)

        try:
            module = importlib.import_module(plugin_name)
            plugin_info = _load_user_plugin(plugin_name, module)
            if plugin_info:
                plugins.append(plugin_info)
        except ImportError as e:
            # Log but don't fail - plugin may have missing dependencies
            print(f"Warning: Failed to import plugin '{plugin_name}': {e}")
        finally:
            # Clean up sys.path
            if path_added and parent_path in sys.path:
                sys.path.remove(parent_path)

    return plugins


def _load_user_plugin(name: str, module: Any) -> dict[str, Any]:
    """Load plugin info from a module.

    Parameters
    ----------
    name : str
        Plugin name
    module : Any
        Imported module object

    Returns
    -------
    dict[str, Any]
        Plugin info dict with name, module, adapters, nodes
    """
    return {
        "name": name,
        "module": module.__name__,
        "version": getattr(module, "__version__", "0.0.0"),
        "description": (module.__doc__ or "").strip().split("\n")[0],
        "adapters": _discover_adapters_in_module(module),
        "nodes": _discover_nodes_in_module(module),
    }


def _discover_adapters_in_module(module: Any) -> list[dict[str, Any]]:
    """Discover adapter classes in a module.

    Looks for adapters in __all__ or by class name convention.
    """
    from hexdag.kernel.schema import SchemaGenerator

    adapters: list[dict[str, Any]] = []

    # Get items from __all__ or fall back to dir()
    items = getattr(module, "__all__", None) or [n for n in dir(module) if not n.startswith("_")]

    for name in items:
        obj = getattr(module, name, None)
        if obj is None or not isinstance(obj, type):
            continue

        # Check if it's an adapter by decorator or naming convention
        hexdag_type = getattr(obj, "_hexdag_type", None)
        is_adapter_by_type = str(hexdag_type) == "adapter" if hexdag_type else False
        is_adapter_by_name = name.endswith(("Adapter", "Memory", "LLM"))

        if not (is_adapter_by_type or is_adapter_by_name):
            continue

        # Detect port type
        port_type = _detect_adapter_port_type(obj)

        # Get secrets from decorator
        secrets_dict = getattr(obj, "_hexdag_secrets", {})
        secrets = list(secrets_dict.keys()) if secrets_dict else []

        # Generate config schema
        try:
            config_schema = SchemaGenerator.from_callable(obj.__init__)
        except Exception:
            config_schema = {"type": "object", "properties": {}}

        adapters.append({
            "name": getattr(obj, "_hexdag_name", name),
            "module_path": f"{module.__name__}.{name}",
            "port_type": port_type,
            "description": getattr(
                obj,
                "_hexdag_description",
                (obj.__doc__ or "No description").split("\n")[0].strip(),
            ),
            "config_schema": config_schema,
            "secrets": secrets,
            # UI metadata (optional, for Studio)
            "icon": getattr(obj, "_hexdag_icon", "Package"),
            "color": getattr(obj, "_hexdag_color", "#6b7280"),
        })

    return adapters


def _discover_nodes_in_module(module: Any) -> list[dict[str, Any]]:
    """Discover node classes in a module.

    Looks for nodes in __all__ or by class name convention.
    """

    nodes: list[dict[str, Any]] = []

    # Get items from __all__ or fall back to dir()
    items = getattr(module, "__all__", None) or [n for n in dir(module) if not n.startswith("_")]

    for name in items:
        obj = getattr(module, name, None)
        if obj is None or not isinstance(obj, type):
            continue

        # Check if it's a node by decorator or naming convention
        hexdag_type = getattr(obj, "_hexdag_type", None)
        is_node_by_type = str(hexdag_type) == "node" if hexdag_type else False
        is_node_by_name = name.endswith("Node")

        if not (is_node_by_type or is_node_by_name):
            continue

        # Convert CamelCase to snake_case for kind
        kind = _to_snake_case(getattr(obj, "_hexdag_name", name))

        # Get schema from __call__ (node factories) or __init__
        try:
            config_schema = _get_node_schema(obj)
        except Exception:
            config_schema = {"type": "object", "properties": {}}

        nodes.append({
            "kind": kind,
            "name": name,
            "module_path": f"{module.__name__}.{name}",
            "description": getattr(
                obj,
                "_hexdag_description",
                (obj.__doc__ or "No description").split("\n")[0].strip(),
            ),
            "config_schema": config_schema,
            # UI metadata (optional, for Studio)
            "icon": getattr(obj, "_hexdag_icon", "Box"),
            "color": getattr(obj, "_hexdag_color", "#6b7280"),
        })

    return nodes


def _detect_adapter_port_type(adapter_class: type) -> str:
    """Detect port type from adapter class using protocol inspection.

    Falls back to name-based heuristics if no protocol inheritance found.
    """
    # Check explicit decorator metadata first
    explicit_port = getattr(adapter_class, "_hexdag_implements_port", None)
    if explicit_port:
        return str(explicit_port)

    # Check protocol inheritance (MRO)
    mro_names = [c.__name__ for c in adapter_class.__mro__]

    # LLM protocols
    if any(
        name in mro_names
        for name in ("LLM", "SupportsGeneration", "SupportsFunctionCalling", "SupportsVision")
    ):
        return "llm"

    if "Memory" in mro_names:
        return "memory"

    if "DatabasePort" in mro_names or "SQLAdapter" in mro_names:
        return "database"

    if "SecretPort" in mro_names:
        return "secret"

    if "FileStoragePort" in mro_names or "VectorStorePort" in mro_names:
        return "storage"

    if "ToolRouter" in mro_names:
        return "tool_router"

    # Fall back to name-based heuristics
    class_name = adapter_class.__name__.lower()
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
        if any(kw in class_name for kw in keywords):
            return port_type

    return "unknown"


def _to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\\1_\\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\\1_\\2", s1).lower()


def _get_node_schema(cls: type) -> dict[str, Any]:
    """Extract config schema from node class.

    For node factories, uses __call__ method (where config params are defined).
    Falls back to __init__ for other node types.
    """
    from hexdag.kernel.schema import SchemaGenerator

    # Check if __call__ is overridden on this class
    for klass in cls.__mro__:
        if klass is object:
            continue
        if "__call__" in klass.__dict__:
            schema = SchemaGenerator.from_callable(cls.__call__)
            if isinstance(schema, dict) and schema.get("properties"):
                return schema
            break

    # Fall back to __init__
    init_schema = SchemaGenerator.from_callable(cls.__init__)  # type: ignore[misc]
    return init_schema if isinstance(init_schema, dict) else {"type": "object", "properties": {}}


def clear_discovery_cache() -> None:
    """Clear all discovery caches.

    Useful for testing or when plugins are dynamically loaded/unloaded.
    """
    discover_modules.cache_clear()
    discover_plugins.cache_clear()
    discover_user_modules.cache_clear()
