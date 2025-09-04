"""Simplified component registry for hexDAG.

This module provides a centralized registry for all hexDAG components,
supporting plugins while maintaining control over component registration.
"""

from __future__ import annotations

import inspect
import logging
import threading
import warnings
from typing import Any

from hexai.core.registry.metadata import ComponentMetadata, InstanceFactory
from hexai.core.registry.plugin_loader import PluginLoader
from hexai.core.registry.types import ComponentType, Namespace, NodeSubtype

logger = logging.getLogger(__name__)

# Constants
NAMESPACE_SEPARATOR = ":"
ERROR_COMPONENT_NONE = "Component cannot be None"
ERROR_CORE_PROTECTED = (
    "Cannot register '{name}' in protected 'core' namespace. "
    "Use a different namespace for plugin components."
)
ERROR_ALREADY_REGISTERED = "Component '{name}' already registered. Use replace=True to override."
ERROR_NOT_FOUND = "Component '{name}' not found"


class ComponentRegistry:
    """Simplified registry with decorator-based registration.

    Combines storage, validation, and permission checking in one place
    for simplicity and clarity.
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        # Simple storage: namespace -> name -> metadata
        self._components: dict[Namespace | str, dict[str, ComponentMetadata]] = {}

        # Track protected components (core components that shouldn't be overridden)
        self._protected_components: set[str] = set()

        # Plugin loader (kept separate as it's a distinct concern)
        self._plugin_loader = PluginLoader(self)

        # Thread lock for all operations
        self._lock = threading.RLock()

        # Ready state
        self._ready = False

    def register(
        self,
        name: str,
        component: Any,
        component_type: ComponentType | str,
        namespace: str | Namespace = Namespace.USER,
        replace: bool = False,
        privileged: bool = False,
        **kwargs: Any,
    ) -> None:
        """Register a component in the registry.

        Parameters
        ----------
        name : str
            Component name.
        component : Any
            Component (class, function, or instance).
        component_type : ComponentType | str
            Type of component.
        namespace : str | Namespace
            Component namespace (default: 'user').
        replace : bool
            Allow replacement of existing component.
        privileged : bool
            Whether the registration has elevated privileges (for core components).
        **kwargs : Any
            Additional metadata (subtype, description, etc.).
        """
        with self._lock:
            # Validate inputs
            self._validate_registration(
                name, component, component_type, namespace, privileged, replace
            )

            # Convert types to enums
            ns_result = self._to_namespace(namespace)
            if ns_result is not None:
                namespace = ns_result
            if isinstance(component_type, str):
                component_type = ComponentType(component_type)

            # Check for existing component
            qualified_name = f"{namespace}{NAMESPACE_SEPARATOR}{name}"
            existing = self._find_component(name)

            if existing:
                self._handle_existing_component(existing, name, namespace, qualified_name, replace)

            # Create metadata
            metadata = ComponentMetadata(
                name=name,
                component_type=component_type,
                component=component,
                namespace=namespace,
                subtype=kwargs.get("subtype"),
                description=kwargs.get("description", ""),
            )

            # Store component
            if namespace not in self._components:
                self._components[namespace] = {}
            self._components[namespace][name] = metadata

            # Mark core components as protected
            if namespace == Namespace.CORE:
                self._protected_components.add(name)

            logger.debug(f"Registered {component_type} '{qualified_name}'")

    def get(
        self,
        name: str,
        namespace: str | Namespace | None = None,
        **kwargs: Any,
    ) -> Any:
        """Get a component by name.

        Parameters
        ----------
        name : str
            Component name. Can include namespace as 'namespace:name'.
        namespace : str | None
            Namespace to search in. If None, searches core first.
        **kwargs : Any
            Arguments to pass to component instantiation.

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
        if NAMESPACE_SEPARATOR in name and namespace is None:
            namespace, name = name.split(NAMESPACE_SEPARATOR, 1)

        # Convert to namespace enum
        namespace = self._to_namespace(namespace) if namespace else None

        # Direct lookup if namespace specified
        if namespace:
            metadata = self._components.get(namespace, {}).get(name)
            if metadata:
                # Resolve lazy component if needed
                component = (
                    metadata.resolve_lazy_component() if metadata.is_lazy else metadata.component
                )
                return InstanceFactory.create_instance(component, **kwargs)
            raise KeyError(f"{ERROR_NOT_FOUND.format(name=f'{namespace}:{name}')}")

        # Search namespaces, prioritizing core
        for ns in [Namespace.CORE] + [ns for ns in self._components if ns != Namespace.CORE]:
            metadata = self._components.get(ns, {}).get(name)
            if metadata:
                if ns != Namespace.CORE and name in self._protected_components:
                    logger.debug(
                        f"Using '{ns}:{name}' (shadows core component). "
                        f"Use registry.get('{name}', namespace='core') for core version"
                    )
                # Resolve lazy component if needed
                component = (
                    metadata.resolve_lazy_component() if metadata.is_lazy else metadata.component
                )
                return InstanceFactory.create_instance(component, **kwargs)

        # Component not found
        available = [f"{ns}:{n}" for ns, comps in self._components.items() for n in comps]
        raise KeyError(f"{ERROR_NOT_FOUND.format(name=name)}. Available: {available}")

    def get_metadata(
        self, name: str, namespace: str | Namespace = Namespace.CORE
    ) -> ComponentMetadata:
        """Get metadata for a component.

        Parameters
        ----------
        name : str
            Component name.
        namespace : str | Namespace
            Namespace (default: 'core').

        Returns
        -------
        ComponentMetadata
            Component metadata.

        Raises
        ------
        KeyError
            If component not found.
        """
        ns_result = self._to_namespace(namespace)
        if ns_result is not None:
            namespace = ns_result
        metadata = self._components.get(namespace, {}).get(name)
        if metadata:
            return metadata
        raise KeyError(f"Component '{namespace}:{name}' not found")

    def list_components(
        self,
        component_type: ComponentType | str | None = None,
        namespace: str | None = None,
        subtype: NodeSubtype | str | None = None,
    ) -> list[tuple[str, ComponentMetadata]]:
        """List registered components with their metadata.

        Parameters
        ----------
        component_type : ComponentType | str | None
            Filter by type.
        namespace : str | None
            Filter by namespace.
        subtype : NodeSubtype | str | None
            Filter by subtype (for nodes).

        Returns
        -------
        list[tuple[str, ComponentMetadata]]
            List of (qualified_name, metadata) tuples.
        """
        with self._lock:
            results = []
            namespace_enum = self._to_namespace(namespace) if namespace else None

            for ns, components in self._components.items():
                # Filter by namespace if specified
                if namespace_enum and ns != namespace_enum:
                    continue

                for name, metadata in components.items():
                    # Filter by type if specified
                    if component_type and metadata.component_type != component_type:
                        continue

                    # Filter by subtype if specified
                    if subtype and metadata.subtype != subtype:
                        continue

                    results.append((f"{ns}{NAMESPACE_SEPARATOR}{name}", metadata))

            return results

    def list_namespaces(self) -> list[str]:
        """List all registered namespaces."""
        with self._lock:
            return sorted(str(ns) for ns in self._components.keys())

    def load_plugins(self) -> int:
        """Load plugins via Python entry points."""
        return self._plugin_loader.load_plugins()

    def set_ready(self, ready: bool = True) -> None:
        """Set registry ready state."""
        with self._lock:
            self._ready = ready

    def is_ready(self) -> bool:
        """Check if registry is ready."""
        return self._ready

    # Private helper methods

    def _validate_registration(
        self,
        name: str,
        component: Any,
        component_type: ComponentType | str,
        namespace: str | Namespace,
        privileged: bool,
        replace: bool,
    ) -> None:
        """Validate a registration request."""
        # Validate name
        if not name or not isinstance(name, str):
            raise ValueError("Component name must be a non-empty string")

        # Validate component
        if component is None:
            raise TypeError(ERROR_COMPONENT_NONE)

        # Basic validation - components should be callable or classes
        if not (inspect.isclass(component) or callable(component)):
            raise TypeError(f"Components must be classes or callable, got {type(component)}")

        # Check namespace permissions
        namespace_enum = self._to_namespace(namespace)
        if namespace_enum == Namespace.CORE and not privileged:
            # Allow replacement of existing core components with privilege
            if not (name in self._protected_components and replace):
                raise PermissionError(ERROR_CORE_PROTECTED.format(name=name))

    def _handle_existing_component(
        self,
        existing: ComponentMetadata,
        name: str,
        namespace: Namespace | str,
        full_name: str,
        replace: bool,
    ) -> None:
        """Handle registration when component already exists."""
        if existing.is_core and namespace != Namespace.CORE:
            # Plugin trying to shadow core component
            warnings.warn(
                f"⚠️  Component '{name}' shadows HEXDAG CORE component!\n"
                f"    Core version remains at 'core:{name}'\n"
                f"    Plugin version will be at '{full_name}'\n"
                f"    This may cause unexpected behavior!",
                UserWarning,
                stacklevel=3,
            )
        elif existing.namespace == namespace:
            if not replace:
                raise ValueError(ERROR_ALREADY_REGISTERED.format(name=full_name))
            logger.info(f"Replaced component '{full_name}'")
        else:
            logger.debug(
                f"Component '{name}' exists in '{existing.namespace}', registering in '{namespace}'"
            )

    def _to_namespace(self, namespace: str | Namespace | None) -> Namespace | str | None:
        """Convert to Namespace enum for known system namespaces.

        Known system namespaces: 'core', 'user', 'plugin'
        These get special handling. All others are treated as custom plugin namespaces.
        """
        if namespace is None:
            return None
        if isinstance(namespace, Namespace):
            return namespace

        # Check for known system namespaces (case-insensitive for convenience)
        if isinstance(namespace, str):
            lower = namespace.lower()
            if lower == "core":
                return Namespace.CORE
            elif lower == "user":
                return Namespace.USER
            elif lower == "plugin":
                return Namespace.PLUGIN

        # Keep as string for custom plugin namespaces
        return namespace

    def _find_component(self, name: str) -> ComponentMetadata | None:
        """Find component in any namespace."""
        for components in self._components.values():
            if name in components:
                return components[name]
        return None


# Global registry instance
registry = ComponentRegistry()
