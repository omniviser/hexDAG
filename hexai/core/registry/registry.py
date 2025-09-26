"""Refactored component registry with improved design patterns."""

from __future__ import annotations

import importlib.util
import inspect
import logging
import re
from typing import TYPE_CHECKING, Any

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
from hexai.core.registry.models import (
    ClassComponent,
    ComponentMetadata,
    ComponentType,
    FunctionComponent,
    InstanceComponent,
    InstanceFactory,
    NodeSubtype,
)

if TYPE_CHECKING:
    from hexai.core.config import ManifestEntry

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

        # Track configurable components
        self._configurable_components: dict[str, dict[str, Any]] = {}

        # Search priority is an implementation detail
        self._search_priority = _search_priority or self.DEFAULT_SEARCH_PRIORITY

        self._lock = ReadWriteLock()

        # Bootstrap state
        self._ready = False
        self._manifest: list[ManifestEntry] | None = None
        self._dev_mode = False  # If True, allows post-bootstrap registration
        self._bootstrap_context = False  # True during bootstrap process

    def _cleanup_state(self) -> None:
        """Clean up registry state on bootstrap failure."""
        self._components.clear()
        self._protected_components.clear()
        self._configurable_components.clear()
        self._ready = False
        self._manifest = None
        self._bootstrap_context = False

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
        self, manifest: list[ManifestEntry], dev_mode: bool
    ) -> list[ManifestEntry]:
        """Prepare registry for bootstrap.

        Returns validated list of ManifestEntry.
        """
        if self._ready:
            raise RegistryAlreadyBootstrappedError(
                "Registry has already been bootstrapped. Use reset() if you need to re-bootstrap."
            )

        # Validate entries for duplicates
        seen = set()
        for entry in manifest:
            key = (entry.namespace, entry.module)
            if key in seen:
                raise ValueError(
                    f"Duplicate manifest entry: namespace='{entry.namespace}', "
                    f"module='{entry.module}'"
                )
            seen.add(key)

        # Store configuration
        self._manifest = manifest
        self._dev_mode = dev_mode

        logger.info("Bootstrapping registry with %d entries", len(manifest))
        return manifest

    def _load_manifest_modules(self, manifest: list[ManifestEntry]) -> int:
        """Load and register components from all manifest modules.

        Returns total number of components registered.
        """
        total_registered = 0

        for entry in manifest:
            # Determine if this is a core module that must load successfully
            # Core modules are either:
            # 1. In the 'core' namespace (explicitly marked as core)
            # 2. Part of the framework (hexai.core.* or hexai.tools.builtin_tools)
            is_core_module = (
                entry.namespace == "core"
                or entry.module.startswith("hexai.core.")
                or entry.module == "hexai.tools.builtin_tools"
            )

            # For non-core modules, check if they exist before trying to load
            if not is_core_module:
                skip_reason = self._check_plugin_requirements(entry.module)
                if skip_reason:
                    logger.info(f"Skipping optional module {entry.module}: {skip_reason}")
                    continue

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
                # For non-core modules, just log a warning and continue
                if not is_core_module:
                    logger.warning(f"Optional module {entry.module} not available: {e}")
                else:
                    # For core modules, fail
                    logger.error("Failed to import core module %s: %s", entry.module, e)
                    self._cleanup_state()
                    raise
            except (
                ComponentAlreadyRegisteredError,
                InvalidComponentError,
                NamespacePermissionError,
            ) as e:
                logger.error("Failed to register components from %s: %s", entry.module, e)
                self._cleanup_state()
                raise

        return total_registered

    def _check_plugin_requirements(self, module_path: str) -> str | None:
        """Check if a plugin module exists and can be imported.

        This only checks if the module exists, not if it can be instantiated.
        Adapters should handle runtime requirements (like API keys) in their
        __init__ method, not at import time.

        Parameters
        ----------
        module_path : str
            The module path to check

        Returns
        -------
        str | None
            Reason for skipping, or None if module exists
        """
        # Only check if the module exists
        try:
            spec = importlib.util.find_spec(module_path)
            if spec is None:
                return f"Module {module_path} not found"
        except (ModuleNotFoundError, ValueError) as e:
            # Module or its parent doesn't exist
            return f"Module {module_path} not found: {e}"

        return None

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
    def manifest(self) -> list[ManifestEntry] | None:
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

        Raises ComponentAlreadyRegisteredError if component exists.

        Returns ComponentMetadata with details about the registration.
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
            wrapped_component = self._wrap_component(component)
            self._validate_component(name, wrapped_component)

            # Validate adapter if it's an adapter component
            if component_type_enum == ComponentType.ADAPTER:
                self._validate_adapter_registration(name, component, namespace_str)

            if namespace_str in self.PROTECTED_NAMESPACES and not privileged:
                raise NamespacePermissionError(name, namespace_str)

            # Check for duplicates - always error
            if namespace_str in self._components and name in self._components[namespace_str]:
                raise ComponentAlreadyRegisteredError(name, namespace_str)
            implements_port_str: str | None = None
            # Check direct class attribute for adapters
            if component_type_enum == ComponentType.ADAPTER and hasattr(
                component, "_hexdag_implements_port"
            ):
                implements_port_str = getattr(component, "_hexdag_implements_port")  # noqa: B009

            # Extract port requirements (for tools, nodes, etc.)
            port_requirements_list: list[str] = []
            if hasattr(component, "_hexdag_required_ports"):
                port_requirements_list = getattr(component, "_hexdag_required_ports", [])  # noqa: B009

            # Create metadata
            metadata = ComponentMetadata(
                name=name,
                component_type=component_type_enum,
                component=wrapped_component,
                namespace=namespace_str,
                subtype=subtype,
                description=description,
                implements_port=implements_port_str,  # For adapters
                port_requirements=port_requirements_list,  # For tools needing ports
            )

            # Store component
            if namespace_str not in self._components:
                self._components[namespace_str] = {}
            self._components[namespace_str][name] = metadata

            if namespace_str in self.PROTECTED_NAMESPACES:
                self._protected_components.add(f"{namespace_str}:{name}")

            # Check if component implements the ConfigurableComponent protocol
            if hasattr(component, "get_config_class"):
                try:
                    # Get the method and call it using getattr to avoid type issues
                    get_config_method = getattr(component, "get_config_class")  # noqa: B009
                    config_class = get_config_method()
                    # Use the component's registered name as the config namespace
                    self._configurable_components[name] = {
                        "component_class": component,
                        "config_class": config_class,
                        "namespace": namespace_str,
                        "name": name,
                        "type": component_type_enum,
                        "port": implements_port_str,
                    }
                    logger.debug("Registered configurable component: %s", name)
                except Exception as e:
                    logger.debug(
                        "Component %s does not implement ConfigurableComponent: %s", name, e
                    )

            logger.debug("Registered %s", metadata.qualified_name)
            return metadata

    def get_metadata(
        self, name: str, namespace: str | None = None, component_type: ComponentType | None = None
    ) -> ComponentMetadata:
        """Get component metadata without instantiation.

        This is useful when you want to inspect a component
        before deciding whether/how to instantiate it.

        Parameters
        ----------
        name : str
            Component name
        namespace : str | None
            Namespace to search in
        component_type : ComponentType | None
            Filter by component type
        """
        # No lock needed after bootstrap (immutable)
        if not self._ready:
            with self._lock.read():
                return self._get_metadata_unlocked(name, namespace, component_type)
        return self._get_metadata_unlocked(name, namespace, component_type)

    def _get_metadata_unlocked(
        self, name: str, namespace: str | None = None, component_type: ComponentType | None = None
    ) -> ComponentMetadata:
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

        # Check component type if specified
        if component_type and metadata.component_type != component_type:
            raise ComponentNotFoundError(
                f"Component '{name}' exists but "
                f"is type {metadata.component_type}, not {component_type}",
                namespace,
                self._get_available_components(),
            )

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

    def get(
        self, name: str, namespace: str | None = None, init_params: dict[str, object] | None = None
    ) -> object:
        """Get and instantiate a component.

        This is a convenience wrapper around get_metadata() + instantiation.
        """
        metadata = self.get_metadata(name, namespace)
        return InstanceFactory.create_instance(metadata.component, init_params)

    def get_info(self, name: str, namespace: str | None = None) -> ComponentMetadata:
        """Get detailed information about a component."""
        # No lock needed after bootstrap (immutable)
        if not self._ready:
            with self._lock.read():
                return self._get_info_unlocked(name, namespace)
        return self._get_info_unlocked(name, namespace)

    def _get_info_unlocked(self, name: str, namespace: str | None = None) -> ComponentMetadata:
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

        return metadata

    def list_components(
        self,
        component_type: ComponentType | None = None,
        namespace: str | None = None,
        subtype: NodeSubtype | None = None,
    ) -> list[ComponentMetadata]:
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
    ) -> list[ComponentMetadata]:
        """List components without holding lock (assumes lock already held or not needed)."""
        results = []

        # Normalize namespace if provided
        normalized_namespace = self._normalize_namespace(namespace) if namespace else None

        for ns_str, components in self._components.items():
            if normalized_namespace and ns_str != normalized_namespace:
                continue

            for metadata in components.values():
                if component_type and metadata.component_type != component_type:
                    continue
                if subtype and metadata.subtype != subtype:
                    continue

                results.append(metadata)

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
        except ValueError as e:
            valid = ", ".join(t.value for t in ComponentType)
            raise InvalidComponentError(
                component_type, f"Invalid component type. Must be one of: {valid}"
            ) from e

    def _wrap_component(
        self, component: object
    ) -> ClassComponent | FunctionComponent | InstanceComponent:
        """Wrap raw component in appropriate type wrapper."""
        if inspect.isclass(component):
            return ClassComponent(value=component)
        elif inspect.isfunction(component) or inspect.ismethod(component):
            return FunctionComponent(value=component)
        else:
            return InstanceComponent(value=component)

    def _validate_component(
        self, name: str, component: ClassComponent | FunctionComponent | InstanceComponent
    ) -> None:
        """Validate component name and value."""
        if not name:
            raise InvalidComponentError(
                name or "<empty>", "Component name must be a non-empty string"
            )

        if not re.match(r"^[a-zA-Z0-9_]+$", name):
            raise InvalidComponentError(name, f"Component name must be alphanumeric, got '{name}'")

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
        available: list[str] = []
        for ns, components in self._components.items():
            available.extend(f"{ns}:{name}" for name in components)
        return available

    def _get_available_ports(self) -> list[str]:
        """Get list of all available port names."""
        available: list[str] = []
        for ns, components in self._components.items():
            for name, metadata in components.items():
                if metadata.component_type == ComponentType.PORT:
                    available.append(f"{ns}:{name}")
        return available

    def _validate_adapter_registration(
        self,
        adapter_name: str,
        adapter_component: object,
        namespace: str,
    ) -> None:
        """Validate adapter implementation at registration time.

        Parameters
        ----------
        adapter_name : str
            Name of the adapter being registered
        adapter_component : object
            The adapter class/instance being registered
        namespace : str
            Namespace where adapter is being registered

        Raises
        ------
        InvalidComponentError
            If adapter doesn't properly implement its declared port
        """
        # Get adapter metadata from _hexdag_implements_port attribute
        implements_port = None

        # The component may be the actual class or wrapped in ClassComponent
        actual_component = adapter_component
        if hasattr(actual_component, "value"):
            # ClassComponent wrapper
            actual_component = getattr(actual_component, "value")  # noqa: B009
        if hasattr(actual_component, "_hexdag_implements_port"):
            implements_port = getattr(actual_component, "_hexdag_implements_port")  # noqa: B009

        if not implements_port:
            # No port declared, skip validation
            return

        # Try to find the port in registry - ports should be registered by now (Phase A)
        try:
            # Look for port with various namespace combinations
            port_meta = None
            search_attempts = []

            # Handle both qualified and unqualified port names
            if ":" in implements_port:
                # Qualified name provided
                search_attempts.append(implements_port)
            else:
                # Unqualified - search with priority
                # Follow DEFAULT_SEARCH_PRIORITY: "core", "user", "plugin"
                search_attempts = [
                    f"core:{implements_port}",  # Core namespace first
                    f"{namespace}:{implements_port}",  # Same namespace as adapter
                    implements_port,  # As declared (will search all)
                ]

            for attempt in search_attempts:
                try:
                    port_meta = self._get_metadata_unlocked(
                        attempt, component_type=ComponentType.PORT
                    )
                    if port_meta:
                        logger.debug(
                            "Found port '%s' for adapter '%s' as '%s'",
                            implements_port,
                            adapter_name,
                            attempt,
                        )
                        break
                except ComponentNotFoundError:
                    continue

            if not port_meta:
                # Port must exist - this is now an error since ports are registered first
                raise InvalidComponentError(
                    adapter_name,
                    f"Adapter '{adapter_name}' declares it implements port '{implements_port}', "
                    f"but port '{implements_port}' does not exist in registry. "
                    f"Available ports: {', '.join(self._get_available_ports())}",
                )
            # Validate required methods using introspection
            # Get the port Protocol class
            port_class = port_meta.raw_component if port_meta else None
            if port_class:
                from hexai.core.registry.introspection import validate_adapter_implementation

                # Get the actual adapter class to check

                if inspect.isclass(adapter_component):
                    adapter_class = adapter_component
                else:
                    adapter_class = type(adapter_component)

                # Validate implementation
                _, missing_methods = validate_adapter_implementation(
                    adapter_class,
                    port_class,  # type: ignore[arg-type]
                )

                if missing_methods:
                    raise InvalidComponentError(
                        adapter_name,
                        f"Adapter '{adapter_name}' does not implement required methods "
                        f"from port '{implements_port}': {', '.join(missing_methods)}",
                    )

        except ComponentNotFoundError as e:
            # Port doesn't exist
            raise InvalidComponentError(
                adapter_name,
                f"Adapter '{adapter_name}' declares it implements port '{implements_port}', "
                f"but port does not exist: {e}",
            ) from e

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
            adapters = []

            # Search all namespaces for adapters

            for components in self._components.values():
                for metadata in components.values():
                    # Check if it's an adapter
                    if metadata.component_type != ComponentType.ADAPTER:
                        continue

                    # Check if it implements the requested port
                    implements = metadata.implements_port  # From ComponentMetadata
                    if not implements:
                        # Check _implements_port attribute (set by adapter decorator)
                        # The component may be wrapped in ClassComponent
                        actual_component = metadata.component
                        if hasattr(actual_component, "value"):
                            # ClassComponent wrapper
                            actual_component = getattr(actual_component, "value")  # noqa: B009
                        if hasattr(actual_component, "_hexdag_implements_port"):
                            implements = getattr(actual_component, "_hexdag_implements_port")  # noqa: B009

                    if implements:
                        # Handle namespaced port names
                        port_base = port_name.split(":")[-1] if ":" in port_name else port_name
                        implements_base = (
                            implements.split(":")[-1] if ":" in implements else implements
                        )

                        if implements == port_name or implements_base == port_base:
                            adapters.append(metadata)

            return adapters


# Global registry instance
registry = ComponentRegistry()
