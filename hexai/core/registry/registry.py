"""Simplified component registry for hexDAG.

This module provides a centralized registry for all hexDAG components,
supporting plugins while maintaining control over component registration.
"""

from __future__ import annotations

import inspect
import logging
import re
import threading
import warnings
from typing import Any

from hexai.core.registry.exceptions import (
    ComponentAlreadyRegisteredError,
    ComponentNotFoundError,
    InvalidComponentError,
    NamespacePermissionError,
)
from hexai.core.registry.metadata import ComponentMetadata, InstanceFactory
from hexai.core.registry.plugin_loader import PluginLoader
from hexai.core.registry.types import ComponentType, Namespace, NodeSubtype

logger = logging.getLogger(__name__)

# Constants
NAMESPACE_SEPARATOR = ":"


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
        component_type: str,
        namespace: str = "user",
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
        component_type : str
            Type of component ('node', 'tool', 'adapter', etc.).
        namespace : str
            Component namespace (default: 'user').
        replace : bool
            Allow replacement of existing component.
        privileged : bool
            Whether the registration has elevated privileges (for core components).
        **kwargs : Any
            Additional metadata (subtype, description, etc.).
        """
        with self._lock:
            # Step 1: Convert external string API to internal enums
            namespace_enum = self._to_namespace(namespace)
            component_type_enum = ComponentType(component_type)

            # Step 2: Validate registration request
            # This checks: name format, component validity, namespace permissions
            self._validate_registration(name, component, namespace_enum, privileged, replace)

            # Step 3: Check if component already exists (in ANY namespace)
            qualified_name = f"{namespace_enum}{NAMESPACE_SEPARATOR}{name}"
            existing = self._find_component(name)

            if existing:
                # Handle conflicts: shadowing, replacement, or error
                self._handle_existing_component(
                    existing, name, namespace_enum, qualified_name, replace
                )

            # Step 4: Create metadata for the component
            metadata = ComponentMetadata(
                name=name,
                component_type=component_type_enum,
                component=component,
                namespace=namespace_enum,
                subtype=kwargs.get("subtype"),
                description=kwargs.get("description", ""),
            )

            # Store component
            if namespace_enum not in self._components:
                self._components[namespace_enum] = {}
            self._components[namespace_enum][name] = metadata

            # Step 5: Mark core components as protected
            # This prevents non-privileged code from registering in core namespace
            if namespace_enum == Namespace.CORE:
                self._protected_components.add(name)

            logger.debug(f"Registered {component_type_enum} '{qualified_name}'")

    def get(
        self,
        name: str,
        namespace: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Get a component by name and instantiate it.

        Three search strategies (in priority order):
        1. Qualified name (e.g., 'core:my_component') - searches only that namespace
        2. Explicit namespace parameter - searches only that namespace
        3. Fallback search - searches all namespaces (core first for compatibility)

        Parameters
        ----------
        name : str
            Component name. Can include namespace as 'namespace:name'.
        namespace : str | None
            Explicit namespace to search in. Ignored if name is qualified.
        **kwargs : Any
            Arguments to pass to component instantiation.

        Returns
        -------
        Any
            The component instance.

        Raises
        ------
        ComponentNotFoundError
            If component not found.

        Examples
        --------
        >>> registry.get('my_node')  # Searches all namespaces, core first
        >>> registry.get('my_node', namespace='plugin')  # Only searches plugin namespace
        >>> registry.get('plugin:my_node')  # Only searches plugin namespace
        """
        with self._lock:
            # Strategy 1: Qualified name takes precedence (e.g., 'namespace:component')
            if NAMESPACE_SEPARATOR in name:
                return self._get_by_qualified_name(name, **kwargs)

            # Strategy 2: Explicit namespace parameter
            elif namespace is not None:
                return self._get_from_specific_namespace(name, namespace, **kwargs)

            # Strategy 3: Search all namespaces with fallback
            else:
                return self._get_with_fallback_search(name, **kwargs)

    def _get_by_qualified_name(self, qualified_name: str, **kwargs: Any) -> Any:
        """Get component using a qualified name (e.g., 'namespace:component').

        This is the most explicit way to get a component - no ambiguity.

        Parameters
        ----------
        qualified_name : str
            Name in format 'namespace:component'
        **kwargs : Any
            Arguments for component instantiation

        Returns
        -------
        Any
            The instantiated component

        Raises
        ------
        ComponentNotFoundError
            If component not found in the specified namespace
        """
        namespace_str, component_name = qualified_name.split(NAMESPACE_SEPARATOR, 1)
        namespace = self._to_namespace(namespace_str)

        return self._get_and_instantiate(component_name, namespace, namespace_str, **kwargs)

    def _get_from_specific_namespace(self, name: str, namespace: str, **kwargs: Any) -> Any:
        """Get component from a specific namespace only.

        No fallback, no searching - just looks in the specified namespace.

        Parameters
        ----------
        name : str
            Component name (without namespace prefix)
        namespace : str
            The namespace to search in
        **kwargs : Any
            Arguments for component instantiation

        Returns
        -------
        Any
            The instantiated component

        Raises
        ------
        ComponentNotFoundError
            If component not found in the specified namespace
        """
        namespace_enum = self._to_namespace(namespace)
        return self._get_and_instantiate(name, namespace_enum, namespace, **kwargs)

    def _get_with_fallback_search(self, name: str, **kwargs: Any) -> Any:
        """Search for component across all namespaces with intelligent fallback.

        Search order:
        1. Core namespace (highest priority for backward compatibility)
        2. All other namespaces in registration order

        This method logs warnings when non-core components shadow core ones.

        Parameters
        ----------
        name : str
            Component name to search for
        **kwargs : Any
            Arguments for component instantiation

        Returns
        -------
        Any
            The instantiated component

        Raises
        ------
        ComponentNotFoundError
            If component not found in any namespace
        """
        # First, try core namespace (backward compatibility)
        metadata = self._get_from_namespace(name, Namespace.CORE)
        if metadata:
            return self._instantiate_component(metadata, **kwargs)

        # Then try other namespaces
        for ns in self._components:
            if ns != Namespace.CORE:
                metadata = self._get_from_namespace(name, ns)
                if metadata:
                    # Warn if this shadows a protected component name
                    if name in self._protected_components:
                        logger.debug(
                            f"Using '{ns}:{name}' which shadows a core component name. "
                            f"Use get('{name}', namespace='core') for the core version."
                        )
                    return self._instantiate_component(metadata, **kwargs)

        # Not found anywhere - provide helpful error
        available = self._list_all_components()
        raise ComponentNotFoundError(name, None, available)

    def get_metadata(self, name: str, namespace: str = Namespace.CORE.value) -> ComponentMetadata:
        """Get metadata for a component.

        Parameters
        ----------
        name : str
            Component name.
        namespace : str
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
        with self._lock:
            namespace_enum = self._to_namespace(namespace)
            metadata = self._components.get(namespace_enum, {}).get(name)
            if metadata:
                return metadata
            raise ComponentNotFoundError(name, namespace)

    def list_components(
        self,
        component_type: str | None = None,
        namespace: str | None = None,
        subtype: str | None = None,
    ) -> list[tuple[str, ComponentMetadata]]:
        """List registered components with their metadata.

        Parameters
        ----------
        component_type : str | None
            Filter by type ('node', 'tool', etc.).
        namespace : str | None
            Filter by namespace.
        subtype : str | None
            Filter by subtype (for nodes).

        Returns
        -------
        list[tuple[str, ComponentMetadata]]
            List of (qualified_name, metadata) tuples.
        """
        with self._lock:
            results = []
            namespace_enum = self._to_namespace(namespace) if namespace else None
            type_enum = ComponentType(component_type) if component_type else None
            subtype_enum = NodeSubtype(subtype) if subtype else None

            for ns, components in self._components.items():
                # Filter by namespace if specified
                if namespace_enum and ns != namespace_enum:
                    continue

                for name, metadata in components.items():
                    # Filter by type if specified
                    if type_enum and metadata.component_type != type_enum:
                        continue

                    # Filter by subtype if specified
                    if subtype_enum and metadata.subtype != subtype_enum:
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
        namespace_enum: Namespace | str,
        privileged: bool,
        replace: bool,
    ) -> None:
        """Validate a registration request."""
        # Validate name
        if not name or not isinstance(name, str):
            raise InvalidComponentError(
                name or "<empty>", "Component name must be a non-empty string"
            )

        # Validate component name - must be alphanumeric (plus underscores for snake_case)
        if not re.match(r"^[a-zA-Z0-9_]+$", name):
            raise InvalidComponentError(
                name,
                f"Component name must only contain letters, numbers, and underscores, got '{name}'",
            )

        # Validate component
        if component is None:
            raise InvalidComponentError(name, "Component cannot be None")

        # Basic validation - components should be callable or classes
        if not (inspect.isclass(component) or callable(component)):
            raise InvalidComponentError(
                name, f"Components must be classes or callable, got {type(component)}"
            )

        # Check namespace permissions
        if namespace_enum == Namespace.CORE and not privileged:
            # Allow replacement of existing core components with privilege
            if not (name in self._protected_components and replace):
                raise NamespacePermissionError(name, Namespace.CORE.value)

    def _handle_existing_component(
        self,
        existing: ComponentMetadata,
        name: str,
        namespace: Namespace | str,
        full_name: str,
        replace: bool,
    ) -> None:
        """Handle registration when component already exists.

        Three cases:
        1. Plugin shadowing core component - WARNING but allowed
        2. Same namespace collision - ERROR unless replace=True
        3. Different namespace - OK (multiple components with same name)
        """
        if existing.is_core and namespace != Namespace.CORE:
            # Case 1: Plugin/user component shadows core - this is ALLOWED but warned
            # The core component remains accessible via 'core:name'
            warnings.warn(
                f"⚠️  Component '{name}' shadows HEXDAG CORE component!\n"
                f"    Core version remains at 'core:{name}'\n"
                f"    Plugin version will be at '{full_name}'\n"
                f"    This may cause unexpected behavior!",
                UserWarning,
                stacklevel=3,
            )
        elif existing.namespace == namespace:
            # Case 2: Same namespace - this is a real conflict
            if not replace:
                raise ComponentAlreadyRegisteredError(name, str(namespace))
            logger.info(f"Replaced component '{full_name}'")
        else:
            # Case 3: Different namespaces - this is fine, components can share names
            logger.debug(
                f"Component '{name}' exists in '{existing.namespace}', registering in '{namespace}'"
            )

    def _to_namespace(self, namespace: str | None) -> Namespace | str:
        """Convert string to Namespace enum for known system namespaces.

        This is an internal method that converts external string API to internal enums.
        Validates that namespace names only contain alphanumeric characters.

        Known system namespaces: 'core', 'user', 'plugin', 'test'
        These get converted to enums. All others remain as strings for custom namespaces.

        Parameters
        ----------
        namespace : str | None
            Namespace string or None

        Returns
        -------
        Namespace | str
            Enum for system namespaces, string for custom ones

        Raises
        ------
        ValueError
            If namespace is None or contains invalid characters
        """
        if namespace is None:
            raise InvalidComponentError("<none>", "Namespace cannot be None")

        # Validate namespace name - must be alphanumeric (plus underscores)
        if not re.match(r"^[a-zA-Z0-9_]+$", namespace):
            raise InvalidComponentError(
                namespace,
                "Namespace name must only contain letters, numbers, and underscores, "
                f"got '{namespace}'",
            )

        # Check for known system namespaces (case-insensitive for convenience)
        lower = namespace.lower()
        if lower == "core":
            return Namespace.CORE
        elif lower == "user":
            return Namespace.USER
        elif lower == "plugin":
            return Namespace.PLUGIN
        elif lower == "test":
            return Namespace.TEST

        # Keep as string for custom plugin namespaces
        return namespace

    def _find_component(self, name: str) -> ComponentMetadata | None:
        """Find component in any namespace."""
        for components in self._components.values():
            if name in components:
                return components[name]
        return None

    def _get_from_namespace(
        self, name: str, namespace: Namespace | str
    ) -> ComponentMetadata | None:
        """Get component metadata from a specific namespace.

        Parameters
        ----------
        name : str
            Component name
        namespace : Namespace | str
            Namespace to search in

        Returns
        -------
        ComponentMetadata | None
            Component metadata if found, None otherwise
        """
        return self._components.get(namespace, {}).get(name)

    def _get_and_instantiate(
        self,
        name: str,
        namespace: Namespace | str,
        original_namespace: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Get component from namespace and instantiate it.

        Consolidates the common pattern of getting metadata and instantiating.

        Parameters
        ----------
        name : str
            Component name
        namespace : Namespace | str
            Namespace to search in (enum or string)
        original_namespace : str | None
            Original namespace string for error messages (if different from namespace)
        **kwargs : Any
            Arguments for component instantiation

        Returns
        -------
        Any
            The instantiated component

        Raises
        ------
        ComponentNotFoundError
            If component not found in the specified namespace
        """
        metadata = self._get_from_namespace(name, namespace)
        if not metadata:
            # Use original_namespace for error if provided, else convert namespace to string
            error_namespace = original_namespace if original_namespace else str(namespace)
            raise ComponentNotFoundError(name, error_namespace)

        return self._instantiate_component(metadata, **kwargs)

    def _instantiate_component(self, metadata: ComponentMetadata, **kwargs: Any) -> Any:
        """Resolve and instantiate a component from its metadata.

        Parameters
        ----------
        metadata : ComponentMetadata
            Component metadata
        **kwargs : Any
            Arguments to pass to component instantiation

        Returns
        -------
        Any
            The instantiated component
        """
        # Resolve lazy component if needed
        component = metadata.resolve_lazy_component() if metadata.is_lazy else metadata.component
        return InstanceFactory.create_instance(component, **kwargs)

    def _list_all_components(self) -> list[str]:
        """List all available component names with namespaces.

        Returns
        -------
        list[str]
            List of 'namespace:component' strings
        """
        return [f"{ns}:{name}" for ns, comps in self._components.items() for name in comps]


# Global registry instance
registry = ComponentRegistry()
