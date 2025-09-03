"""Plugin discovery functionality (simplified - pluggy handles most of this)."""

from __future__ import annotations

import logging
from typing import Any

import pluggy

logger = logging.getLogger(__name__)

# Re-export pluggy markers for plugin authors
hookimpl = pluggy.HookimplMarker("hexdag")
hookspec = pluggy.HookspecMarker("hexdag")


def create_plugin(namespace: str) -> type:
    """Create a simple plugin class.

    This is a convenience function for plugin authors who just want
    to register components without implementing the full plugin interface.

    Parameters
    ----------
    namespace : str
        Plugin namespace.

    Returns
    -------
    type
        Plugin class with hexdag_initialize hook.

    Examples
    --------
    >>> # In plugin's __init__.py:
    >>> from hexai.core.registry.discovery import create_plugin
    >>>
    >>> Plugin = create_plugin('my_plugin')
    >>>
    >>> def register():
    ...     '''Entry point function.'''
    ...     from . import nodes  # Import triggers decorators
    ...     return Plugin()
    """

    class SimplePlugin:
        """Auto-generated plugin class."""

        def __init__(self) -> None:
            self.namespace = namespace

        @hookimpl  # type: ignore[misc]
        def hexdag_initialize(self) -> None:
            """Plugin is already initialized by importing modules."""
            logger.debug(f"Plugin '{namespace}' initialized")

    SimplePlugin.__name__ = f"{namespace.title()}Plugin"
    return SimplePlugin


# For backward compatibility
def discover_entry_points(group: str) -> dict[str, Any]:
    """Legacy function for entry point discovery.

    Note: Pluggy now handles this internally via load_setuptools_entrypoints().
    This is kept for backward compatibility only.

    Parameters
    ----------
    group : str
        Entry point group name.

    Returns
    -------
    dict[str, Any]
        Empty dict (pluggy handles discovery).
    """
    logger.warning(
        f"discover_entry_points('{group}') is deprecated. "
        "Pluggy handles entry point discovery automatically."
    )
    return {}


def discover_plugins() -> int:
    """Legacy function for plugin discovery.

    Note: The registry now handles this automatically.
    This is kept for backward compatibility only.

    Returns
    -------
    int
        0 (registry handles discovery).
    """
    logger.warning(
        "discover_plugins() is deprecated. The registry discovers plugins automatically."
    )
    return 0
