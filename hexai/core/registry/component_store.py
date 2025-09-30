"""Component storage and retrieval."""

from __future__ import annotations

import logging

from hexai.core.registry.exceptions import (
    ComponentAlreadyRegisteredError,
    ComponentNotFoundError,
)
from hexai.core.registry.models import (
    NAMESPACE_SEPARATOR,
    ComponentInfo,
    ComponentMetadata,
    ComponentType,
    NodeSubtype,
)
from hexai.core.registry.validation import RegistryValidator

logger = logging.getLogger(__name__)


class ComponentStore:
    """Handles component storage, search, and retrieval.

    This class is responsible for:
    - Storing components in a namespace-based structure
    - Searching for components with priority ordering
    - Retrieving component metadata
    - Listing components by various criteria
    """

    # Default search priority (immutable)
    DEFAULT_SEARCH_PRIORITY = ("core", "user", "plugin")

    def __init__(self, search_priority: tuple[str, ...] | None = None) -> None:
        """Initialize empty component store.

        Args
        ----
        search_priority : tuple[str, ...] | None
            Order to search namespaces when no namespace is specified.
        """
        self._components: dict[str, dict[str, ComponentMetadata]] = {}
        self._search_priority = search_priority or self.DEFAULT_SEARCH_PRIORITY
        self._protected_components: set[str] = set()

    def register(
        self,
        metadata: ComponentMetadata,
        namespace: str,
        is_protected: bool = False,
    ) -> None:
        """Store a component in the registry.

        Args
        ----
        metadata : ComponentMetadata
            Component metadata to store.
        namespace : str
            Normalized namespace to store in.
        is_protected : bool
            Whether to mark as protected.

        Raises
        ------
        ComponentAlreadyRegisteredError
            If component already exists in namespace.
        """
        # Check for duplicates
        if namespace in self._components and metadata.name in self._components[namespace]:
            raise ComponentAlreadyRegisteredError(metadata.name, namespace)

        # Store component
        if namespace not in self._components:
            self._components[namespace] = {}
        self._components[namespace][metadata.name] = metadata

        if is_protected:
            self._protected_components.add(f"{namespace}:{metadata.name}")

        logger.debug("Stored component %s:%s", namespace, metadata.name)

    def get_metadata(
        self,
        name: str,
        namespace: str | None = None,
        component_type: ComponentType | None = None,
    ) -> ComponentMetadata:
        """Get component metadata.

        Args
        ----
        name : str
            Component name (can be qualified like "core:my_component").
        namespace : str | None
            Explicit namespace to search in.
        component_type : ComponentType | None
            Filter by component type.

        Returns
        -------
        ComponentMetadata
            The component metadata.

        Raises
        ------
        ComponentNotFoundError
            If component is not found.
        """
        # Parse the component name and namespace
        component_name, resolved_namespace = self._resolve_component_location(name, namespace)

        # Get metadata
        if resolved_namespace:
            metadata = self._get_from_namespace(component_name, resolved_namespace)
        else:
            metadata = self._search_component(component_name)

        if not metadata:
            raise ComponentNotFoundError(name, namespace, self._get_available_components())

        # Check component type if specified
        if component_type and metadata.component_type != component_type:
            raise ComponentNotFoundError(
                f"Component '{name}' exists but "
                f"is type {metadata.component_type}, not {component_type}",
                namespace,
                self._get_available_components(),
            )

        return metadata

    def get_info(self, name: str, namespace: str | None = None) -> ComponentInfo:
        """Get detailed component information.

        Args
        ----
        name : str
            Component name (can be qualified).
        namespace : str | None
            Explicit namespace to search in.

        Returns
        -------
        ComponentInfo
            Component information.

        Raises
        ------
        ComponentNotFoundError
            If component is not found.
        """
        if NAMESPACE_SEPARATOR in name:
            namespace_str, component_name = name.split(NAMESPACE_SEPARATOR, 1)
        else:
            if namespace:
                namespace_str = namespace
            else:
                found_namespace = self._find_namespace(name)
                if found_namespace is None:
                    raise ComponentNotFoundError(name, None, self._get_available_components())
                namespace_str = found_namespace
            component_name = name

        metadata = self._get_from_namespace(component_name, namespace_str)
        if not metadata:
            raise ComponentNotFoundError(name, namespace)

        qualified_name = f"{namespace_str}.{component_name}" if namespace_str else component_name
        return ComponentInfo(
            name=component_name,
            namespace=namespace_str,
            qualified_name=qualified_name,
            component_type=metadata.component_type,
            metadata=metadata,
            is_protected=namespace_str == "core",
        )

    def list_components(
        self,
        component_type: ComponentType | None = None,
        namespace: str | None = None,
        subtype: NodeSubtype | None = None,
    ) -> list[ComponentInfo]:
        """List components matching criteria.

        Args
        ----
        component_type : ComponentType | None
            Filter by component type.
        namespace : str | None
            Filter by namespace.
        subtype : NodeSubtype | None
            Filter by node subtype.

        Returns
        -------
        list[ComponentInfo]
            List of matching components.
        """
        results = []

        for ns_str, components in self._components.items():
            if namespace and ns_str != namespace:
                continue

            for name, metadata in components.items():
                if component_type and metadata.component_type != component_type:
                    continue
                if subtype and metadata.subtype != subtype:
                    continue

                qualified_name = f"{ns_str}.{name}" if ns_str else name
                results.append(
                    ComponentInfo(
                        name=name,
                        namespace=ns_str,
                        qualified_name=qualified_name,
                        component_type=metadata.component_type,
                        metadata=metadata,
                        is_protected=ns_str == "core",
                    )
                )

        return results

    def list_namespaces(self) -> list[str]:
        """List all registered namespaces.

        Returns
        -------
        list[str]
            Sorted list of namespace names.
        """
        return sorted(self._components.keys())

    def is_namespace_empty(self, namespace: str) -> bool:
        """Check if a namespace has no components.

        Args
        ----
        namespace : str
            Namespace to check.

        Returns
        -------
        bool
            True if namespace is empty or doesn't exist.
        """
        return namespace not in self._components or len(self._components[namespace]) == 0

    def clear(self) -> None:
        """Clear all stored components (for bootstrap cleanup)."""
        self._components.clear()
        self._protected_components.clear()

    def get_adapters_for_port(self, port_name: str) -> list[ComponentMetadata]:
        """Get all adapters implementing a specific port.

        Args
        ----
        port_name : str
            Name of the port.

        Returns
        -------
        list[ComponentMetadata]
            List of adapter components implementing the port.
        """
        adapters = []

        for components in self._components.values():
            for metadata in components.values():
                # Check if it's an adapter
                if metadata.component_type != ComponentType.ADAPTER:
                    continue

                # Check if it implements the requested port
                implements = metadata.implements_port or RegistryValidator.get_implements_port(
                    metadata.component
                )

                if implements and self._port_names_match(port_name, implements):
                    adapters.append(metadata)

        return adapters

    def _port_names_match(self, requested: str, implemented: str) -> bool:
        """Check if port names match (handling qualified names).

        Args
        ----
        requested : str
            Requested port name (may be qualified).
        implemented : str
            Implemented port name (may be qualified).

        Returns
        -------
        bool
            True if names match.
        """
        # Exact match
        if implemented == requested:
            return True

        # Match base names (ignoring namespace)
        requested_base = requested.split(":")[-1] if ":" in requested else requested
        implemented_base = implemented.split(":")[-1] if ":" in implemented else implemented

        return requested_base == implemented_base

    def get_available_components(self) -> list[str]:
        """Get list of all available component names.

        Returns
        -------
        list[str]
            List of qualified component names.
        """
        return self._get_available_components()

    def get_available_ports(self) -> list[str]:
        """Get list of all available port names.

        Returns
        -------
        list[str]
            List of qualified port names.
        """
        available: list[str] = []
        for ns, components in self._components.items():
            for name, metadata in components.items():
                if metadata.component_type == ComponentType.PORT:
                    available.append(f"{ns}:{name}")
        return available

    # Private helper methods

    def _resolve_component_location(
        self, name: str, namespace: str | None = None
    ) -> tuple[str, str | None]:
        """Resolve component name and namespace from input.

        Returns
        -------
        tuple[str, str | None]
            Tuple of (component_name, resolved_namespace).
        """
        if NAMESPACE_SEPARATOR in name:
            # Qualified name like "core:my_component"
            namespace_str, component_name = name.split(NAMESPACE_SEPARATOR, 1)
            return component_name, namespace_str
        if namespace:
            # Explicit namespace provided
            return name, namespace
        # Search needed
        return name, None

    def _get_from_namespace(self, name: str, namespace: str) -> ComponentMetadata | None:
        """Get metadata from specific namespace.

        Returns
        -------
        ComponentMetadata | None
            Metadata if found, None otherwise.
        """
        return self._components.get(namespace, {}).get(name)

    def _search_component(self, name: str) -> ComponentMetadata | None:
        """Search for component with priority order.

        Returns
        -------
        ComponentMetadata | None
            Metadata if found, None otherwise.
        """
        namespace = self._find_namespace(name)
        if namespace:
            return self._components[namespace][name]
        return None

    def _find_namespace(self, name: str) -> str | None:
        """Find namespace containing a component.

        Returns
        -------
        str | None
            Namespace string if found, None otherwise.
        """
        # First check priority namespaces
        for ns in self._search_priority:
            if ns in self._components and name in self._components[ns]:
                return ns

        # Then check other namespaces
        for ns, components in self._components.items():
            if ns not in self._search_priority and name in components:
                return ns
        return None

    def _get_available_components(self) -> list[str]:
        """Get list of all available component names.

        Returns
        -------
        list[str]
            List of qualified component names.
        """
        available: list[str] = []
        for ns, components in self._components.items():
            available.extend(f"{ns}:{name}" for name in components)
        return available
