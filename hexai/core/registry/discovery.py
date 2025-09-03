"""Entry points discovery functionality for the registry system."""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Any

logger = logging.getLogger(__name__)


def discover_entry_points(group: str) -> dict[str, Any]:
    """Discover components via entry points.

    Scans the specified entry point group for registered components
    and returns a mapping of names to their loaded implementations.

    Parameters
    ----------
    group : str
        Entry point group name to scan (e.g., 'hexdag.plugins').

    Returns
    -------
    dict[str, Any]
        Mapping of entry point names to loaded components.

    Examples
    --------
    >>> # Discover plugins
    >>> plugins = discover_entry_points('hexdag.plugins')
    >>> for name, register_func in plugins.items():
    ...     register_func(registry, name)

    Raises
    ------
    ImportError
        If entry points module is not available.
    """
    discovered: dict[str, Any] = {}

    try:
        eps = entry_points(group=group)
        for ep in eps:
            try:
                component = ep.load()
                discovered[ep.name] = component
                logger.debug(f"Discovered component '{ep.name}' from entry point")
            except Exception as e:
                logger.warning(f"Failed to load entry point '{ep.name}' in group '{group}': {e}")
    except Exception as e:
        logger.warning(f"Failed to discover entry points for group '{group}': {e}")

    if discovered:
        logger.info(f"Discovered {len(discovered)} components from entry point group '{group}'")

    return discovered


def discover_plugins() -> int:
    """Discover and register all installed plugins via entry points.

    This function is the main plugin discovery mechanism. Plugins declare
    themselves via entry points in their pyproject.toml or setup.py:

    ```toml
    [project.entry-points."hexai.plugins"]
    my_plugin = "my_plugin:register"
    ```

    The register function should import modules with decorated components,
    which will auto-register with the registry.

    Returns
    -------
    int
        Number of plugins successfully loaded.

    Examples
    --------
    >>> # Called during registry initialization
    >>> count = discover_plugins()
    >>> print(f"Loaded {count} plugins")
    """
    plugins = discover_entry_points("hexai.plugins")
    loaded_count = 0

    for plugin_name, register_func in plugins.items():
        try:
            logger.info(f"Loading plugin: {plugin_name}")
            # Call the plugin's register function
            # This should import the plugin's modules, triggering decorator registrations
            register_func()
            loaded_count += 1
            logger.info(f"Successfully loaded plugin: {plugin_name}")
        except Exception as e:
            logger.error(f"Failed to load plugin '{plugin_name}': {e}")

    return loaded_count
