"""Simplified component registry using pluggy and decorators.

This module provides a centralized registry for all hexDAG components,
supporting plugins while maintaining control over component registration.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import pluggy

from hexai.core.registry.metadata import ComponentMetadata
from hexai.core.registry.types import ComponentType

logger = logging.getLogger(__name__)

# Plugin system markers
hookspec = pluggy.HookspecMarker("hexdag")
hookimpl = pluggy.HookimplMarker("hexdag")


class PluginSpec:
    """Plugin hook specifications for hexDAG."""

    @hookspec  # type: ignore[misc]
    def hexdag_initialize(self) -> None:
        """Initialize plugin by importing modules with decorated components."""
        pass

    @hookspec  # type: ignore[misc]
    def hexdag_configure(self, config: dict) -> None:
        """Configure plugin with settings."""
        pass


class ComponentRegistry:
    """Simplified registry with decorator-based registration for all components.

    This registry uses a unified decorator approach for both core and plugin
    components, with automatic protection for core components.

    Examples
    --------
    >>> from hexai.core.registry import registry
    >>>
    >>> # Get a core component
    >>> node = registry.get('passthrough')
    >>>
    >>> # Get a plugin component
    >>> analyzer = registry.get('sentiment_analyzer', namespace='nlp_plugin')
    >>>
    >>> # List all components
    >>> components = registry.list_components()
    """

    _instance: Optional[ComponentRegistry] = None
    _pending_components: List[Tuple[Type, Dict[str, Any]]] = []

    def __new__(cls) -> ComponentRegistry:
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the registry with pluggy support."""
        if hasattr(self, "_initialized"):
            return

        # Component storage: namespace -> name -> metadata
        self._components: Dict[str, Dict[str, ComponentMetadata]] = {}

        # Track protected core components
        self._protected_components: Set[str] = set()

        # Component instances cache (lazy instantiation)
        self._instances: Dict[str, Any] = {}

        # Plugin manager
        self.pm = pluggy.PluginManager("hexdag")
        self.pm.add_hookspecs(PluginSpec)

        # Load core components first
        self._load_core_components()

        # Process pending core components
        self._process_pending_components()

        # Discover and load plugins
        self._load_plugins()

        # Process any pending plugin components
        self._process_pending_components()

        self._initialized = True
        logger.info("Component registry initialized")

    @classmethod
    def add_pending(cls, component_cls: Type, metadata: Dict[str, Any]) -> None:
        """Add a component to the pending registration queue."""
        cls._pending_components.append((component_cls, metadata))

    def register(
        self,
        name: str,
        component: Any,
        component_type: ComponentType | str,
        namespace: str = "user",
        replace: bool = False,
        **kwargs: Any,
    ) -> None:
        """Register a component in the registry.

        Parameters
        ----------
        name : str
            Component name.
        component : Any
            Component instance or class.
        component_type : ComponentType | str
            Type of component.
        namespace : str
            Component namespace (default: 'user').
        replace : bool
            Allow replacement of existing component.
        **kwargs : Any
            Additional metadata.

        Raises
        ------
        ValueError
            If component exists and replace=False.
        """
        full_name = f"{namespace}:{name}"

        # Check for existing component
        existing = self._find_component(name)

        if existing:
            if existing.is_core and namespace != "core":
                # Plugin trying to override core component
                warnings.warn(
                    f"⚠️  Component '{name}' shadows CORE component!\n"
                    f"    Core version remains at 'core:{name}'\n"
                    f"    Plugin version will be at '{full_name}'\n"
                    f"    Use registry.get('{name}', namespace='{namespace}') for plugin version",
                    UserWarning,
                    stacklevel=2,
                )
                logger.warning(
                    f"Plugin component '{full_name}' shadows core component 'core:{name}'"
                )
            elif existing.namespace == namespace:
                if not existing.replaceable and not replace:
                    raise ValueError(
                        f"Component '{full_name}' already registered and not replaceable"
                    )
                logger.info(f"Replacing component '{full_name}'")
            elif existing.namespace != namespace:
                logger.debug(
                    f"Component '{name}' exists in '{existing.namespace}', "
                    f"registering in '{namespace}'"
                )

        # Create namespace if needed
        if namespace not in self._components:
            self._components[namespace] = {}

        # Create metadata
        metadata = ComponentMetadata(
            name=name,
            component_type=component_type,
            namespace=namespace,
            is_core=(namespace == "core"),
            replaceable=kwargs.pop("replaceable", False),
            version=kwargs.pop("version", "1.0.0"),
            author=kwargs.pop("author", namespace),
            description=kwargs.pop("description", ""),
            tags=frozenset(kwargs.pop("tags", set())),
            dependencies=frozenset(kwargs.pop("dependencies", set())),
        )

        # Store component
        self._components[namespace][name] = metadata

        # Store instance if provided
        instance_key = f"{namespace}:{name}"
        if not isinstance(component, type):
            self._instances[instance_key] = component
        else:
            # Store class for lazy instantiation
            metadata.component_class = component

        # Track core components
        if namespace == "core":
            self._protected_components.add(name)

        logger.debug(f"Registered {component_type} '{full_name}'")

    def get(
        self,
        name: str,
        component_type: ComponentType | str | None = None,
        namespace: str | None = None,
    ) -> Any:
        """Get a component by name.

        Parameters
        ----------
        name : str
            Component name. Can include namespace as 'namespace:name'.
        component_type : ComponentType | str | None
            Component type for validation (optional).
        namespace : str | None
            Namespace to search in. If None, searches core first.

        Returns
        -------
        Any
            The component instance.

        Raises
        ------
        KeyError
            If component not found.
        """
        # Parse namespace from name if provided
        if ":" in name and namespace is None:
            namespace, name = name.split(":", 1)

        # Direct lookup if namespace specified
        if namespace:
            if namespace in self._components and name in self._components[namespace]:
                return self._get_instance(namespace, name)
            raise KeyError(f"Component '{namespace}:{name}' not found")

        # Search order: core -> other namespaces
        # Core always wins for unqualified names
        if "core" in self._components and name in self._components["core"]:
            return self._get_instance("core", name)

        # Search other namespaces
        for ns in self._components:
            if ns != "core" and name in self._components[ns]:
                if name in self._protected_components:
                    logger.debug(
                        f"Using '{ns}:{name}' (shadows core component). "
                        f"Use registry.get('{name}', namespace='core') for core version"
                    )
                return self._get_instance(ns, name)

        available = self._list_available_components()
        raise KeyError(f"Component '{name}' not found. Available components: {available}")

    def get_metadata(
        self, name: str, component_type: ComponentType | str | None = None, namespace: str = "core"
    ) -> ComponentMetadata:
        """Get metadata for a component.

        Parameters
        ----------
        name : str
            Component name.
        component_type : ComponentType | str | None
            Component type (optional).
        namespace : str
            Namespace (default: 'core').

        Returns
        -------
        ComponentMetadata
            Component metadata.
        """
        # Parse namespace from name
        if ":" in name:
            namespace, name = name.split(":", 1)

        if namespace in self._components and name in self._components[namespace]:
            return self._components[namespace][name]

        raise KeyError(f"Component '{namespace}:{name}' not found")

    def list_components(
        self,
        component_type: ComponentType | str | None = None,
        namespace: str | None = None,
        include_metadata: bool = False,
    ) -> Union[list[str], list[tuple[str, ComponentMetadata]]]:
        """List registered components.

        Parameters
        ----------
        component_type : ComponentType | str | None
            Filter by type.
        namespace : str | None
            Filter by namespace.
        include_metadata : bool
            Include metadata in results.

        Returns
        -------
        list[str] | list[tuple[str, ComponentMetadata]]
            Component names or (name, metadata) tuples.
        """
        results: List[Any] = []

        namespaces = [namespace] if namespace else self._components.keys()

        for ns in namespaces:
            if ns not in self._components:
                continue

            for name, metadata in self._components[ns].items():
                # Filter by type if specified
                if component_type and metadata.component_type != component_type:
                    continue

                if include_metadata:
                    results.append((f"{ns}:{name}", metadata))
                else:
                    results.append(f"{ns}:{name}")

        return results

    def list_namespaces(self) -> list[str]:
        """List all registered namespaces.

        Returns
        -------
        list[str]
            Namespace names.
        """
        return sorted(self._components.keys())

    def _get_instance(self, namespace: str, name: str) -> Any:
        """Get or create component instance (lazy instantiation)."""
        instance_key = f"{namespace}:{name}"

        # Return cached instance if available
        if instance_key in self._instances:
            return self._instances[instance_key]

        # Create instance from class
        metadata = self._components[namespace][name]
        if hasattr(metadata, "component_class") and metadata.component_class is not None:
            instance = metadata.component_class()
            self._instances[instance_key] = instance
            return instance

        raise RuntimeError(f"No instance or class for component '{instance_key}'")

    def _find_component(self, name: str) -> Optional[ComponentMetadata]:
        """Find component in any namespace."""
        for namespace_components in self._components.values():
            if name in namespace_components:
                return namespace_components[name]
        return None

    def _list_available_components(self) -> list[str]:
        """List all available component names."""
        components = []
        for namespace, namespace_components in self._components.items():
            for name in namespace_components:
                components.append(f"{namespace}:{name}")
        return components

    def _load_core_components(self) -> None:
        """Load core hexDAG components."""
        try:
            # Import core modules to trigger decorators
            import hexai.core.nodes  # noqa: F401

            logger.debug("Core nodes module loaded")
        except ImportError as e:
            logger.debug(f"Core nodes module not found: {e}")

        # Add imports for other core modules as they're created
        # import hexai.core.adapters
        # import hexai.core.tools
        # etc.

    def _load_plugins(self) -> None:
        """Discover and load plugins via pluggy."""
        # Load from setuptools entry points
        self.pm.load_setuptools_entrypoints("hexdag.plugins")

        # Call plugin initialization hooks
        self.pm.hook.hexdag_initialize()

        plugin_count = len(self.pm.get_plugins())
        if plugin_count > 0:
            logger.info(f"Loaded {plugin_count} plugins")

    def _process_pending_components(self) -> None:
        """Process all pending component registrations from decorators."""
        processed = 0

        while self._pending_components:
            component_cls, metadata = self._pending_components.pop(0)

            try:
                self.register(
                    name=metadata["name"],
                    component=component_cls,
                    component_type=metadata["component_type"],
                    namespace=metadata.get("namespace", "core"),
                    **{
                        k: v
                        for k, v in metadata.items()
                        if k not in ["name", "component_type", "namespace"]
                    },
                )
                processed += 1
            except Exception as e:
                logger.error(f"Failed to register component {metadata.get('name', 'unknown')}: {e}")

        if processed > 0:
            logger.debug(f"Processed {processed} pending components")

    def _clear_for_testing(self) -> None:
        """Clear registry for testing. Only works in test mode."""
        import os

        if not os.environ.get("PYTEST_CURRENT_TEST"):
            raise RuntimeError("clear_for_testing() only available in tests")

        self._components.clear()
        self._protected_components.clear()
        self._instances.clear()
        self._pending_components.clear()
        logger.warning("Registry cleared for testing")


# Global registry instance
registry = ComponentRegistry()
