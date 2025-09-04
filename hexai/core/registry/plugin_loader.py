"""Simplified plugin loading functionality."""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hexai.core.registry.registry import ComponentRegistry

logger = logging.getLogger(__name__)

# Constants
# Note: This is the Python entry point group name for plugin discovery.
# It's not related to registry namespaces - plugins can register their
# components in any namespace (e.g., "my_plugin", "custom_namespace", etc.)
PLUGIN_ENTRY_GROUP = "hexdag.plugins"


class PluginLoader:
    """Simple plugin discovery and loading via Python entry points."""

    def __init__(self, registry: ComponentRegistry | None = None):
        """Initialize plugin loader."""
        self.registry = registry
        self._loaded_plugins: set[str] = set()

    def set_registry(self, registry: ComponentRegistry) -> None:
        """Set the registry to load plugins into."""
        self.registry = registry

    def load_plugins(self) -> int:
        """Discover and load all plugins.

        Returns
        -------
        int
            Number of plugins successfully loaded
        """
        if not self.registry:
            raise RuntimeError("Registry not set. Call set_registry() first.")

        entry_points = self._discover_entry_points()
        loaded_count = 0

        for entry_point in entry_points:
            if self._load_entry_point(entry_point):
                loaded_count += 1

        if loaded_count > 0:
            logger.info(f"Loaded {loaded_count} plugins")

        return loaded_count

    def load_specific_plugin(self, plugin_name: str) -> bool:
        """Load a specific plugin by name."""
        if plugin_name in self._loaded_plugins:
            logger.debug(f"Plugin '{plugin_name}' already loaded")
            return True

        entry_points = self._discover_entry_points()

        for entry_point in entry_points:
            if entry_point.name == plugin_name:
                return self._load_entry_point(entry_point)

        logger.warning(f"Plugin '{plugin_name}' not found")
        return False

    def list_available_plugins(self) -> list[str]:
        """List all available plugins."""
        entry_points = self._discover_entry_points()
        return [ep.name for ep in entry_points]

    def list_loaded_plugins(self) -> list[str]:
        """List all loaded plugins."""
        return list(self._loaded_plugins)

    def _discover_entry_points(self) -> list[Any]:
        """Discover plugin entry points."""
        try:
            entry_points = importlib.metadata.entry_points()

            # Handle different Python versions
            if hasattr(entry_points, "select"):
                # Python 3.10+
                return list(entry_points.select(group=PLUGIN_ENTRY_GROUP))
            else:
                # Python 3.9 - entry_points returns dict
                eps = entry_points()  # type: ignore
                return list(eps.get(PLUGIN_ENTRY_GROUP, []))
        except Exception as e:
            logger.warning(f"Plugin discovery failed: {e}")
            return []

    def _load_entry_point(self, entry_point: Any) -> bool:
        """Load a single entry point."""
        try:
            # Skip if already loaded
            if entry_point.name in self._loaded_plugins:
                logger.debug(f"Plugin '{entry_point.name}' already loaded")
                return False

            # Load the entry point - this imports the module
            # which triggers decorator registrations
            plugin = entry_point.load()

            # If it's a callable registration function, call it
            if callable(plugin):
                plugin(self.registry)

            self._loaded_plugins.add(entry_point.name)
            logger.info(f"Loaded plugin: {entry_point.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load plugin '{entry_point.name}': {e}")
            return False
