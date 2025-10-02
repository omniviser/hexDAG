"""Refactored component registry with improved design patterns."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

# Removed AdapterValidator - validation moved to runtime/type-checkers
from hexai.core.registry.bootstrap_manager import BootstrapManager
from hexai.core.registry.component_store import ComponentStore
from hexai.core.registry.discovery import register_components as default_register_components
from hexai.core.registry.exceptions import (
    NamespacePermissionError,
    RegistryImmutableError,
)
from hexai.core.registry.locks import ReadWriteLock
from hexai.core.registry.models import (
    ComponentInfo,
    ComponentMetadata,
    ComponentType,
    InstanceFactory,
    NodeSubtype,
)
from hexai.core.registry.validation import RegistryValidator

if TYPE_CHECKING:
    from hexai.core.config import ManifestEntry

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """Central registry with bootstrap-based initialization.

    The registry follows a strict lifecycle:
    1. Empty at creation
    2. Populated via bootstrap(manifest)
    3. Read-only after bootstrap (in production)

    This is similar to Django's app registry pattern.

    The registry now delegates to specialized collaborators:
    - ComponentStore: Storage and retrieval
    - BootstrapManager: Lifecycle management
    - RegistryValidator: Name and namespace validation
    """

    def __init__(self, _search_priority: tuple[str, ...] | None = None) -> None:
        """Initialize an empty registry.

        Args
        ----
        _search_priority : tuple[str, ...] | None
            Internal parameter for testing. Users should not set this.
        """
        # Collaborators
        self._store = ComponentStore(search_priority=_search_priority)
        self._bootstrap = BootstrapManager(self._store)
        # Removed AdapterValidator - Pydantic Protocols handle validation at runtime

        # Track configurable components
        self._configurable_components: dict[str, dict[str, Any]] = {}

        self._lock = ReadWriteLock()

    def get_configurable_components(self) -> dict[str, dict[str, Any]]:
        """Get all registered configurable components.

        Returns
        -------
        dict[str, dict[str, Any]]
            Dictionary mapping config namespaces to component info
        """
        with self._lock.read():
            return self._configurable_components.copy()

    def bootstrap(
        self,
        manifest: list[ManifestEntry],
        dev_mode: bool = False,
    ) -> None:
        """Bootstrap the registry from a manifest.

        This method:
        1. Validates the manifest
        2. Imports modules
        3. Calls register_components() for each module
        4. Marks the registry as ready (immutable in production)

        Parameters
        ----------
        manifest : list[ManifestEntry]
            The component manifest declaring what to load.
        dev_mode : bool
            If True, allows post-bootstrap registration (for development).
        """
        with self._lock.write():
            # Delegate to bootstrap manager
            self._bootstrap.bootstrap(
                manifest=manifest,
                dev_mode=dev_mode,
                register_components_fn=lambda _, ns, mod: default_register_components(
                    registry=self, namespace=ns, module_path=mod
                ),
            )

    @property
    def ready(self) -> bool:
        """Check if the registry has been bootstrapped."""
        return self._bootstrap.ready

    @property
    def manifest(self) -> list[ManifestEntry] | None:
        """Get the current manifest."""
        return self._bootstrap.manifest

    @property
    def dev_mode(self) -> bool:
        """Check if registry is in development mode."""
        return self._bootstrap.dev_mode

    def is_namespace_empty(self, namespace: str) -> bool:
        """Check if a namespace has no components (for testing).

        Returns
        -------
            True if namespace has no components or doesn't exist, False otherwise.
        """
        return self._store.is_namespace_empty(namespace)

    def register(
        self,
        name: str,
        component: object,
        component_type: str,
        namespace: str = "user",
        privileged: bool = False,
        subtype: NodeSubtype | str | None = None,
        description: str = "",
    ) -> ComponentMetadata:
        """Register a component in the registry.

        After bootstrap, registration is only allowed in dev mode or
        during the bootstrap process itself.

        Returns
        -------
            ComponentInfo with details about the registration.

        Raises
        ------
        NamespacePermissionError
            If namespace permission is denied.
        RegistryImmutableError
            If registry is read-only and not in dev mode.
        """
        with self._lock.write():
            # Check if we can register
            if not self._bootstrap.can_register():
                raise RegistryImmutableError(
                    f"Cannot register component '{name}' after bootstrap. "
                    f"Registry is read-only in production mode. "
                    f"Use dev_mode=True in bootstrap() for development."
                )

            # Validate inputs
            namespace_str = RegistryValidator.validate_namespace(namespace)
            component_type_enum = RegistryValidator.validate_component_type(component_type)
            wrapped_component = RegistryValidator.wrap_component(component)
            RegistryValidator.validate_component_name(name)

            # Check namespace permissions
            if RegistryValidator.is_protected_namespace(namespace_str) and not privileged:
                raise NamespacePermissionError(name, namespace_str)

            # Extract metadata from component attributes
            implements_port_str = (
                RegistryValidator.get_implements_port(component)
                if component_type_enum == ComponentType.ADAPTER
                else None
            )
            port_requirements_list = RegistryValidator.get_required_ports(component)

            # Create metadata
            metadata = ComponentMetadata(
                name=name,
                component_type=component_type_enum,
                component=wrapped_component,
                namespace=namespace_str,
                subtype=subtype,
                description=description,
                implements_port=implements_port_str,
                port_requirements=port_requirements_list,
            )

            # Store component
            is_protected = RegistryValidator.is_protected_namespace(namespace_str)
            self._store.register(metadata, namespace_str, is_protected)

            # Track configurable components
            self._track_configurable_component(
                name, component, component_type_enum, namespace_str, implements_port_str
            )

            logger.debug("Registered %s", metadata.qualified_name)
            return metadata

    def _track_configurable_component(
        self,
        name: str,
        component: object,
        component_type: ComponentType,
        namespace: str,
        implements_port: str | None,
    ) -> None:
        """Track components that implement ConfigurableComponent protocol."""
        if hasattr(component, "get_config_class"):
            try:
                get_config_method = getattr(component, "get_config_class")  # noqa: B009
                config_class = get_config_method()
                self._configurable_components[name] = {
                    "component_class": component,
                    "config_class": config_class,
                    "namespace": namespace,
                    "name": name,
                    "type": component_type,
                    "port": implements_port,
                }
                logger.debug("Registered configurable component: %s", name)
            except Exception as e:
                logger.debug("Component %s does not implement ConfigurableComponent: %s", name, e)

    def get_metadata(
        self, name: str, namespace: str | None = None, component_type: ComponentType | None = None
    ) -> ComponentMetadata:
        """Get component metadata without instantiation.

        This is useful when you want to inspect a component
        before deciding whether/how to instantiate it.


        Returns
        -------
            ComponentMetadata for the requested component.

        Args
        ----
        name : str
            Component name
        namespace : str | None
            Namespace to search in
        component_type : ComponentType | None
            Filter by component type
        """
        # No lock needed after bootstrap (immutable)
        if not self._bootstrap.ready:
            with self._lock.read():
                return self._store.get_metadata(name, namespace, component_type)
        return self._store.get_metadata(name, namespace, component_type)

    def get(
        self, name: str, namespace: str | None = None, init_params: dict[str, object] | None = None
    ) -> object:
        """Get and instantiate a component.

        This is a convenience wrapper around get_metadata() + instantiation.

        Returns
        -------
            An instantiated component.
        """
        metadata = self.get_metadata(name, namespace)
        return InstanceFactory.create_instance(metadata.component, init_params)

    def get_info(self, name: str, namespace: str | None = None) -> ComponentInfo:
        """Get detailed information about a component.

        Returns
        -------
            ComponentInfo with component details.
        """
        # No lock needed after bootstrap (immutable)
        if not self._bootstrap.ready:
            with self._lock.read():
                return self._store.get_info(name, namespace)
        return self._store.get_info(name, namespace)

    def list_components(
        self,
        component_type: ComponentType | None = None,
        namespace: str | None = None,
        subtype: NodeSubtype | None = None,
    ) -> list[ComponentInfo]:
        """List components with rich information.

        Returns
        -------
            List of ComponentInfo objects.
        """
        # No lock needed after bootstrap (immutable)
        if not self._bootstrap.ready:
            with self._lock.read():
                return self._store.list_components(component_type, namespace, subtype)
        return self._store.list_components(component_type, namespace, subtype)

    def list_namespaces(self) -> list[str]:
        """List all registered namespaces.

        Returns
        -------
            Sorted list of namespace names.
        """
        # No lock needed after bootstrap (immutable)
        if not self._bootstrap.ready:
            with self._lock.read():
                return self._store.list_namespaces()
        return self._store.list_namespaces()

    def get_adapters_for_port(self, port_name: str) -> list[ComponentMetadata]:
        """Get all adapters that implement a specific port.

        Parameters
        ----------
        port_name : str
            Name of the port

        Returns
        -------
        list[ComponentMetadata]
            List of adapter components that implement the port
        """
        with self._lock.read():
            return self._store.get_adapters_for_port(port_name)

    # Testing support methods

    def _reset_for_testing(self) -> None:
        """Reset registry state (for testing only)."""
        self._store.clear()
        self._bootstrap.reset()
        self._configurable_components.clear()


# Global registry instance
registry = ComponentRegistry()
