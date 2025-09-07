"""Refactored component registry with improved design patterns."""

from __future__ import annotations

import inspect
import logging
import re
from typing import Any

from hexai.core.registry.discovery import register_components as default_register_components
from hexai.core.registry.exceptions import (
    ComponentAlreadyRegisteredError,
    ComponentNotFoundError,
    InvalidComponentError,
    NamespacePermissionError,
    RegistryAlreadyBootstrappedError,
    RegistryImmutableError,
)
from hexai.core.registry.locks import ReadWriteLock
from hexai.core.registry.manifest import ComponentManifest
from hexai.core.registry.models import (
    ComponentInfo,
    ComponentMetadata,
    ComponentType,
    InstanceFactory,
    NodeSubtype,
)

logger = logging.getLogger(__name__)

# Constants
NAMESPACE_SEPARATOR = ":"


class ComponentRegistry:
    """Central registry with bootstrap-based initialization.

    The registry follows a strict lifecycle:
    1. Empty at creation
    2. Populated via bootstrap(manifest)
    3. Read-only after bootstrap (in production)

    This is similar to Django's app registry pattern.
    """

    # Known system namespaces
    SYSTEM_NAMESPACES = {"core", "user", "plugin", "test"}
    PROTECTED_NAMESPACES = {"core"}  # Require privilege to register

    # Default search priority (immutable)
    DEFAULT_SEARCH_PRIORITY = ("core", "user", "plugin")

    def __init__(self, _search_priority: tuple[str, ...] | None = None) -> None:
        """Initialize an empty registry.

        Args
        ----
        _search_priority : tuple[str, ...] | None
            Internal parameter for testing. Users should not set this.
        """
        self._components: dict[str, dict[str, ComponentMetadata]] = {}
        self._protected_components: set[str] = set()

        # Search priority is an implementation detail
        self._search_priority = _search_priority or self.DEFAULT_SEARCH_PRIORITY

        self._lock = ReadWriteLock()

        # Bootstrap state
        self._ready = False
        self._manifest: ComponentManifest | None = None
        self._dev_mode = False  # If True, allows post-bootstrap registration
        self._bootstrap_context = False  # True during bootstrap process

    def _cleanup_state(self) -> None:
        """Clean up registry state on bootstrap failure."""
        self._components.clear()
        self._protected_components.clear()
        self._ready = False
        self._manifest = None
        self._bootstrap_context = False

    def bootstrap(
        self,
        manifest: ComponentManifest | list[dict[str, str]],
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
        manifest : ComponentManifest | list[dict[str, str]]
            The component manifest declaring what to load.
        dev_mode : bool
            If True, allows post-bootstrap registration (for development).

        Raises
        ------
        RegistryAlreadyBootstrappedError
            If the registry has already been bootstrapped.
        ImportError
            If a module in the manifest cannot be imported.
        """
        with self._lock.write():
            # Prepare for bootstrap
            manifest = self._prepare_bootstrap(manifest, dev_mode)

            # Load all modules from manifest
            self._bootstrap_context = True
            try:
                total_registered = self._load_manifest_modules(manifest)
                self._finalize_bootstrap(total_registered, dev_mode)
            finally:
                self._bootstrap_context = False

    def _prepare_bootstrap(
        self, manifest: ComponentManifest | list[dict[str, str]], dev_mode: bool
    ) -> ComponentManifest:
        """Prepare registry for bootstrap.

        Returns validated ComponentManifest.
        """
        if self._ready:
            raise RegistryAlreadyBootstrappedError(
                "Registry has already been bootstrapped. Use reset() if you need to re-bootstrap."
            )

        # Convert list to ComponentManifest if needed
        if isinstance(manifest, list):
            manifest = ComponentManifest(manifest)

        # Validate manifest
        manifest.validate()

        # Store configuration
        self._manifest = manifest
        self._dev_mode = dev_mode

        logger.info(f"Bootstrapping registry with {len(manifest.entries)} entries")
        return manifest

    def _load_manifest_modules(self, manifest: ComponentManifest) -> int:
        """Load and register components from all manifest modules.

        Returns total number of components registered.
        """
        total_registered = 0

        for entry in manifest.entries:
            try:
                count = default_register_components(
                    registry=self,
                    namespace=entry.namespace,
                    module_path=entry.module,
                )
                total_registered += count
                logger.info(
                    f"Registered {count} components from {entry.module} "
                    f"into namespace '{entry.namespace}'"
                )
            except ImportError as e:
                logger.error(f"Failed to import module {entry.module}: {e}")
                self._cleanup_state()
                raise
            except (
                ComponentAlreadyRegisteredError,
                InvalidComponentError,
                NamespacePermissionError,
            ) as e:
                logger.error(f"Failed to register components from {entry.module}: {e}")
                self._cleanup_state()
                raise

        return total_registered

    def _finalize_bootstrap(self, total_registered: int, dev_mode: bool) -> None:
        """Mark registry as ready and log completion."""
        self._ready = True
        logger.info(
            f"Bootstrap complete: {total_registered} components registered. "
            f"Registry is {'mutable (dev mode)' if dev_mode else 'read-only'}."
        )

    @property
    def ready(self) -> bool:
        """Check if the registry has been bootstrapped."""
        return self._ready

    @property
    def manifest(self) -> ComponentManifest | None:
        """Get the current manifest."""
        return self._manifest

    @property
    def dev_mode(self) -> bool:
        """Check if registry is in development mode."""
        return self._dev_mode

    def is_namespace_empty(self, namespace: str) -> bool:
        """Check if a namespace has no components (for testing)."""
        return namespace not in self._components or len(self._components[namespace]) == 0

    def register(
        self,
        name: str,
        component: Any,
        component_type: str,
        namespace: str = "user",
        privileged: bool = False,
        **kwargs: Any,
    ) -> ComponentInfo:
        """Register a component in the registry.

        After bootstrap, registration is only allowed in dev mode or
        during the bootstrap process itself.

        Raises ComponentAlreadyRegisteredError if component exists.

        Returns ComponentInfo with details about the registration.
        """
        with self._lock.write():
            # Check if we can register
            if self._ready and not self._dev_mode and not self._bootstrap_context:
                raise RegistryImmutableError(
                    f"Cannot register component '{name}' after bootstrap. "
                    f"Registry is read-only in production mode. "
                    f"Use dev_mode=True in bootstrap() for development."
                )
            namespace_str = self._normalize_namespace(namespace)
            component_type_enum = self._validate_component_type(component_type)
            self._validate_component(name, component)

            if namespace_str in self.PROTECTED_NAMESPACES and not privileged:
                raise NamespacePermissionError(name, namespace_str)

            # Check for duplicates - always error
            if namespace_str in self._components and name in self._components[namespace_str]:
                raise ComponentAlreadyRegisteredError(name, namespace_str)

            # Create metadata
            metadata = ComponentMetadata(
                name=name,
                component_type=component_type_enum,
                component=component,
                namespace=namespace_str,
                subtype=kwargs.get("subtype"),
                description=kwargs.get("description", ""),
            )

            # Store component
            if namespace_str not in self._components:
                self._components[namespace_str] = {}
            self._components[namespace_str][name] = metadata

            # Create registration info
            registration_info = ComponentInfo(
                name=name,
                namespace=namespace_str,
                qualified_name=f"{namespace_str}{NAMESPACE_SEPARATOR}{name}",
                component_type=component_type_enum,
                metadata=metadata,
            )

            if namespace_str in self.PROTECTED_NAMESPACES:
                self._protected_components.add(f"{namespace_str}:{name}")
                registration_info.is_protected = True

            logger.debug(f"Registered {registration_info.qualified_name}")
            return registration_info

    def get_metadata(self, name: str, namespace: str | None = None) -> ComponentMetadata:
        """Get component metadata without instantiation.

        This is useful when you want to inspect a component
        before deciding whether/how to instantiate it.
        """
        # No lock needed after bootstrap (immutable)
        if not self._ready:
            with self._lock.read():
                return self._get_metadata_unlocked(name, namespace)
        return self._get_metadata_unlocked(name, namespace)

    def _get_metadata_unlocked(self, name: str, namespace: str | None = None) -> ComponentMetadata:
        """Get metadata without holding lock (assumes lock already held or not needed)."""
        # Parse the component name and namespace
        component_name, resolved_namespace = self._resolve_component_location(name, namespace)

        # Get metadata
        if resolved_namespace:
            metadata = self._get_metadata_internal(component_name, resolved_namespace)
        else:
            metadata = self._search_component(component_name)

        if not metadata:
            raise ComponentNotFoundError(name, namespace, self._get_available_components())

        return metadata

    def _resolve_component_location(
        self, name: str, namespace: str | None = None
    ) -> tuple[str, str | None]:
        """Resolve component name and namespace from various input formats."""
        if NAMESPACE_SEPARATOR in name:
            # Qualified name like "core:my_component"
            namespace_str, component_name = name.split(NAMESPACE_SEPARATOR, 1)
            return component_name, namespace_str
        elif namespace:
            # Explicit namespace provided
            return name, self._normalize_namespace(namespace)
        else:
            # Search needed
            return name, None

    def get(self, name: str, namespace: str | None = None, **kwargs: Any) -> Any:
        """Get and instantiate a component.

        This is a convenience wrapper around get_metadata() + instantiation.
        """
        metadata = self.get_metadata(name, namespace)
        return InstanceFactory.create_instance(metadata.component, **kwargs)

    def get_info(self, name: str, namespace: str | None = None) -> ComponentInfo:
        """Get detailed information about a component."""
        # No lock needed after bootstrap (immutable)
        if not self._ready:
            with self._lock.read():
                return self._get_info_unlocked(name, namespace)
        return self._get_info_unlocked(name, namespace)

    def _get_info_unlocked(self, name: str, namespace: str | None = None) -> ComponentInfo:
        """Get info without holding lock (assumes lock already held or not needed)."""
        if NAMESPACE_SEPARATOR in name:
            namespace_str, component_name = name.split(NAMESPACE_SEPARATOR, 1)
            namespace_str = self._normalize_namespace(namespace_str)
        else:
            if namespace:
                namespace_str = self._normalize_namespace(namespace)
            else:
                found_namespace = self._find_namespace(name)
                if found_namespace is None:
                    raise ComponentNotFoundError(name, None, self._get_available_components())
                namespace_str = found_namespace
            component_name = name

        metadata = self._get_metadata_internal(component_name, namespace_str)
        if not metadata:
            raise ComponentNotFoundError(name, namespace)

        return ComponentInfo(
            name=component_name,
            namespace=namespace_str,
            qualified_name=f"{namespace_str}{NAMESPACE_SEPARATOR}{component_name}",
            component_type=metadata.component_type,
            metadata=metadata,
            is_protected=f"{namespace_str}:{component_name}" in self._protected_components,
        )

    def list_components(
        self,
        component_type: ComponentType | None = None,
        namespace: str | None = None,
        subtype: NodeSubtype | None = None,
    ) -> list[ComponentInfo]:
        """List components with rich information."""
        # No lock needed after bootstrap (immutable)
        if not self._ready:
            with self._lock.read():
                return self._list_components_unlocked(component_type, namespace, subtype)
        return self._list_components_unlocked(component_type, namespace, subtype)

    def _list_components_unlocked(
        self,
        component_type: ComponentType | None = None,
        namespace: str | None = None,
        subtype: NodeSubtype | None = None,
    ) -> list[ComponentInfo]:
        """List components without holding lock (assumes lock already held or not needed)."""
        results = []

        # Normalize namespace if provided
        normalized_namespace = self._normalize_namespace(namespace) if namespace else None

        for ns_str, components in self._components.items():
            if normalized_namespace and ns_str != normalized_namespace:
                continue

            for name, metadata in components.items():
                if component_type and metadata.component_type != component_type:
                    continue
                if subtype and metadata.subtype != subtype:
                    continue

                results.append(
                    ComponentInfo(
                        name=name,
                        namespace=ns_str,
                        qualified_name=f"{ns_str}{NAMESPACE_SEPARATOR}{name}",
                        component_type=metadata.component_type,
                        metadata=metadata,
                        is_protected=f"{ns_str}:{name}" in self._protected_components,
                    )
                )

        return results

    def list_namespaces(self) -> list[str]:
        """List all registered namespaces."""
        # No lock needed after bootstrap (immutable)
        if not self._ready:
            with self._lock.read():
                return sorted(self._components.keys())
        return sorted(self._components.keys())

    # Private helper methods

    def _normalize_namespace(self, namespace: str | None) -> str:
        """Normalize namespace to lowercase string."""
        if namespace is None or namespace == "":
            return "user"  # Default namespace

        # At this point, namespace must be a non-empty string based on type hints
        if not re.match(r"^[a-zA-Z0-9_]+$", namespace):
            raise InvalidComponentError(
                namespace, f"Namespace must be alphanumeric, got '{namespace}'"
            )

        return namespace.lower()  # Normalize to lowercase

    def _validate_component_type(self, component_type: str) -> ComponentType:
        """Validate component type."""
        try:
            return ComponentType(component_type)
        except ValueError:
            valid = ", ".join(t.value for t in ComponentType)
            raise InvalidComponentError(
                component_type, f"Invalid component type. Must be one of: {valid}"
            )

    def _validate_component(self, name: str, component: Any) -> None:
        """Validate component name and value."""
        if not name or not isinstance(name, str):
            raise InvalidComponentError(
                name or "<empty>", "Component name must be a non-empty string"
            )

        if not re.match(r"^[a-zA-Z0-9_]+$", name):
            raise InvalidComponentError(name, f"Component name must be alphanumeric, got '{name}'")

        # No need to check for None - inspect.isclass and callable handle it
        if not (inspect.isclass(component) or callable(component)):
            raise InvalidComponentError(
                name, f"Component must be class or callable, got {type(component)}"
            )

    def _get_metadata_internal(self, name: str, namespace: str) -> ComponentMetadata | None:
        """Get metadata for a specific component from namespace."""
        return self._components.get(namespace, {}).get(name)

    def _search_component(self, name: str) -> ComponentMetadata | None:
        """Search for component with priority order."""
        namespace = self._find_namespace(name)
        if namespace:
            return self._components[namespace][name]
        return None

    def _find_namespace(self, name: str) -> str | None:
        """Find the namespace containing a component, respecting search priority."""
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
        """Get list of all available component names."""
        available = []
        for ns, components in self._components.items():
            for name in components:
                available.append(f"{ns}:{name}")
        return available


# Global registry instance
registry = ComponentRegistry()
