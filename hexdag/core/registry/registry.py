"""Simplified component registry - storage and lifecycle merged inline."""

from __future__ import annotations

import importlib
import importlib.util
from threading import Lock
from typing import TYPE_CHECKING, Any

from hexdag.core.logging import get_logger
from hexdag.core.registry.discovery import register_components as default_register_components
from hexdag.core.registry.exceptions import (
    ComponentAlreadyRegisteredError,
    ComponentNotFoundError,
    NamespacePermissionError,
    RegistryAlreadyBootstrappedError,
    RegistryImmutableError,
)
from hexdag.core.registry.models import (
    NAMESPACE_SEPARATOR,
    ComponentInfo,
    ComponentMetadata,
    ComponentType,
    InstanceFactory,
    NodeSubtype,
    RegistryValidator,
)
from hexdag.core.schema import SchemaGenerator

if TYPE_CHECKING:
    from collections.abc import (
        Callable as CallableType,  # noqa: F401 - used in cast() string literal
    )

    from hexdag.core.config import ManifestEntry

logger = get_logger(__name__)


class ComponentRegistry:
    """Central component registry with bootstrap-based initialization.

    Lifecycle:
    1. Empty at creation
    2. Populated via bootstrap(manifest)
    3. Read-only after bootstrap (unless dev_mode)

    This merges what was previously split across ComponentStore and BootstrapManager.
    """

    # Default namespace search priority
    DEFAULT_SEARCH_PRIORITY = ("core", "user", "plugin")

    def __init__(self, _search_priority: tuple[str, ...] | None = None) -> None:
        """Initialize an empty registry."""
        # Storage (was ComponentStore)
        self._components: dict[str, dict[str, ComponentMetadata]] = {}
        self._protected_components: set[str] = set()
        self._search_priority = _search_priority or self.DEFAULT_SEARCH_PRIORITY

        # Bootstrap state (was BootstrapManager)
        self._ready = False
        self._manifest: list[ManifestEntry] | None = None
        self._dev_mode = False
        self._bootstrap_context = False

        # Configurable components tracking
        self._configurable_components: dict[str, dict[str, Any]] = {}

        # Schema cache for performance
        self._schema_cache: dict[str, dict | str] = {}

        # Simple lock for bootstrap phase only
        self._lock = Lock()

    # ========================================================================
    # Bootstrap Lifecycle
    # ========================================================================

    def bootstrap(self, manifest: list[ManifestEntry], dev_mode: bool = False) -> None:
        """Bootstrap the registry from a manifest.

        1. Validates the manifest
        2. Imports modules
        3. Calls register_components() for each module
        4. Marks registry as ready (immutable in production)

        Parameters
        ----------
        manifest : list[ManifestEntry]
            Component manifest declaring what to load
        dev_mode : bool
            If True, allows post-bootstrap registration
        """
        with self._lock:
            # Validate not already bootstrapped
            if self._ready:
                raise RegistryAlreadyBootstrappedError(
                    "Registry already bootstrapped. Use _reset_for_testing() if needed."
                )

            # Validate manifest entries for duplicates
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
            logger.info("Bootstrapping registry with {count} entries", count=len(manifest))

            # Load modules and register components
            self._bootstrap_context = True
            total_registered = 0

            try:
                for entry in manifest:
                    is_core = self._is_core_module(entry)

                    # Check if optional plugin exists
                    if not is_core:
                        skip_reason = self._check_plugin_requirements(entry.module)
                        if skip_reason:
                            logger.info(f"Skipping optional module {entry.module}: {skip_reason}")
                            continue

                    try:
                        count = default_register_components(self, entry.namespace, entry.module)
                        total_registered += count
                        logger.info(
                            f"Registered {count} components from {entry.module} "
                            f"into namespace '{entry.namespace}'"
                        )
                    except ImportError as e:
                        if is_core:
                            logger.error(
                                "Failed to import core module {module}: {error}",
                                module=entry.module,
                                error=e,
                            )
                            raise
                        logger.warning(f"Optional module {entry.module} not available: {e}")
                    except (
                        ComponentAlreadyRegisteredError,
                        NamespacePermissionError,
                    ) as e:
                        logger.error(
                            "Failed to register components from {module}: {error}",
                            module=entry.module,
                            error=e,
                        )
                        raise

                # Mark as ready
                self._ready = True
                logger.info(
                    f"Bootstrap complete: {total_registered} components registered. "
                    f"Registry is {'mutable (dev mode)' if dev_mode else 'read-only'}."
                )

            except Exception:
                # Clean up on failure
                self._components.clear()
                self._protected_components.clear()
                self._ready = False
                self._manifest = None
                raise
            finally:
                self._bootstrap_context = False

    def _is_core_module(self, entry: ManifestEntry) -> bool:
        """Check if manifest entry is a core module (must load successfully)."""
        return (
            entry.namespace == "core"
            or entry.module.startswith("hexdag.core.")
            or entry.module == "hexdag.tools.builtin_tools"
        )

    def _check_plugin_requirements(self, module_path: str) -> str | None:
        """Check if plugin module exists. Returns skip reason or None."""
        try:
            spec = importlib.util.find_spec(module_path)
            if spec is None:
                return f"Module {module_path} not found"
        except (ModuleNotFoundError, ValueError) as e:
            return f"Module {module_path} not found: {e}"
        return None

    def _load_manifest_modules(self, manifest: list[ManifestEntry], register_fn: object) -> int:
        """Load modules from manifest. Helper for testing."""
        total = 0
        for entry in manifest:
            is_core = self._is_core_module(entry)
            if not is_core:
                skip_reason = self._check_plugin_requirements(entry.module)
                if skip_reason:
                    continue
            try:
                count = register_fn(self, entry.namespace, entry.module)  # type: ignore[operator]
                total += count
            except ImportError:
                if is_core:
                    raise
        return total

    @property
    def ready(self) -> bool:
        """Check if registry has been bootstrapped."""
        return self._ready

    @property
    def manifest(self) -> list[ManifestEntry] | None:
        """Get the current manifest."""
        return self._manifest

    @property
    def dev_mode(self) -> bool:
        """Check if registry is in development mode."""
        return self._dev_mode

    def _can_register(self) -> bool:
        """Check if registration is currently allowed."""
        return not self._ready or self._dev_mode or self._bootstrap_context

    # ========================================================================
    # Component Registration
    # ========================================================================

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

        After bootstrap, registration only allowed in dev mode.

        Returns
        -------
        ComponentMetadata
            Metadata for the registered component

        Raises
        ------
        RegistryImmutableError
            If registry is read-only and not in dev mode
        NamespacePermissionError
            If namespace permission is denied
        ComponentAlreadyRegisteredError
            If component already exists
        """
        # Check if we can register
        if not self._can_register():
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

        # Store component (inline from ComponentStore)
        if namespace_str in self._components and name in self._components[namespace_str]:
            raise ComponentAlreadyRegisteredError(name, namespace_str)

        if namespace_str not in self._components:
            self._components[namespace_str] = {}
        self._components[namespace_str][name] = metadata

        is_protected = RegistryValidator.is_protected_namespace(namespace_str)
        if is_protected:
            self._protected_components.add(f"{namespace_str}:{name}")

        # Track configurable components
        self._track_configurable_component(
            name, component, component_type_enum, namespace_str, implements_port_str
        )

        logger.debug("Registered {name}", name=metadata.qualified_name)
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
                logger.debug("Registered configurable component: {name}", name=name)
            except Exception as e:
                logger.debug(
                    "Component {name} does not implement ConfigurableComponent: {error}",
                    name=name,
                    error=e,
                )

    # ========================================================================
    # Component Retrieval (inline from ComponentStore)
    # ========================================================================

    def get_metadata(
        self, name: str, namespace: str | None = None, component_type: ComponentType | None = None
    ) -> ComponentMetadata:
        """Get component metadata without instantiation."""
        # Parse qualified names like "core:my_component"
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

    def get(
        self, name: str, namespace: str | None = None, init_params: dict[str, object] | None = None
    ) -> object:
        """Get and instantiate a component."""
        metadata = self.get_metadata(name, namespace)
        return InstanceFactory.create_instance(metadata.component, init_params)

    def get_info(self, name: str, namespace: str | None = None) -> ComponentInfo:
        """Get detailed component information."""
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

    def get_configurable_components(self) -> dict[str, dict[str, Any]]:
        """Get all registered configurable components."""
        return self._configurable_components.copy()

    def get_schema(
        self, name: str, namespace: str | None = None, format: str = "dict"
    ) -> dict | str:
        """Get auto-generated schema for a component.

        Generates JSON Schema from the component's callable signature.
        Schemas are cached for performance.

        Parameters
        ----------
        name : str
            Component name
        namespace : str | None
            Optional namespace (searches if not provided)
        format : str
            Output format - "dict", "yaml", or "json"

        Returns
        -------
        dict | str
            Schema in requested format


        Examples
        --------
        >>> schema = registry.get_schema("llm_node", format="yaml")  # doctest: +SKIP
        >>> print(schema)  # doctest: +SKIP
        type: object
        properties:
          template:
            type: string
        ...
        """
        # Create cache key
        cache_key = f"{namespace or 'auto'}:{name}:{format}"

        # Check cache
        if cache_key in self._schema_cache:
            return self._schema_cache[cache_key]

        # Get component metadata
        metadata = self.get_metadata(name, namespace)

        # Extract callable from component
        import inspect
        from typing import cast

        component: Any = metadata.raw_component

        # If it's a class, get the __call__ method
        if inspect.isclass(component) and callable(component):
            # Get __call__ method from an instance
            try:
                instance = component()
                component = instance.__call__
            except Exception:
                # Fallback: try to get __call__ from class
                component = component.__call__

        # Generate schema (component is now guaranteed to be callable)
        schema = SchemaGenerator.from_callable(cast("CallableType", component), format=format)

        # Cache the result
        self._schema_cache[cache_key] = schema

        return schema

    # ========================================================================
    # Component Listing
    # ========================================================================

    def list_components(
        self,
        component_type: ComponentType | None = None,
        namespace: str | None = None,
        subtype: NodeSubtype | None = None,
    ) -> list[ComponentInfo]:
        """List components matching criteria."""
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
        """List all registered namespaces."""
        return sorted(self._components.keys())

    def is_namespace_empty(self, namespace: str) -> bool:
        """Check if namespace has no components."""
        return namespace not in self._components or len(self._components[namespace]) == 0

    def get_adapters_for_port(self, port_name: str) -> list[ComponentMetadata]:
        """Get all adapters implementing a specific port."""
        adapters = []

        for components in self._components.values():
            for metadata in components.values():
                if metadata.component_type != ComponentType.ADAPTER:
                    continue

                implements = metadata.implements_port or RegistryValidator.get_implements_port(
                    metadata.component
                )

                if implements and self._port_names_match(port_name, implements):
                    adapters.append(metadata)

        return adapters

    # ========================================================================
    # Private Helpers (inline from ComponentStore)
    # ========================================================================

    def _resolve_component_location(
        self, name: str, namespace: str | None = None
    ) -> tuple[str, str | None]:
        """Resolve component name and namespace from input."""
        if NAMESPACE_SEPARATOR in name:
            namespace_str, component_name = name.split(NAMESPACE_SEPARATOR, 1)
            return component_name, namespace_str
        if namespace:
            return name, namespace
        return name, None

    def _get_from_namespace(self, name: str, namespace: str) -> ComponentMetadata | None:
        """Get metadata from specific namespace."""
        return self._components.get(namespace, {}).get(name)

    def _search_component(self, name: str) -> ComponentMetadata | None:
        """Search for component with priority order."""
        namespace = self._find_namespace(name)
        if namespace:
            return self._components[namespace][name]
        return None

    def _find_namespace(self, name: str) -> str | None:
        """Find namespace containing a component."""
        # Check priority namespaces first
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

    def _port_names_match(self, requested: str, implemented: str) -> bool:
        """Check if port names match (handling qualified names)."""
        if implemented == requested:
            return True

        # Match base names (ignoring namespace)
        requested_base = requested.split(":")[-1] if ":" in requested else requested
        implemented_base = implemented.split(":")[-1] if ":" in implemented else implemented

        return requested_base == implemented_base

    # ========================================================================
    # Testing Support
    # ========================================================================

    def _reset_for_testing(self) -> None:
        """Reset registry state (for testing only)."""
        self._components.clear()
        self._protected_components.clear()
        self._configurable_components.clear()
        self._schema_cache.clear()
        self._ready = False
        self._manifest = None
        self._dev_mode = False
        self._bootstrap_context = False


# Global registry instance
registry = ComponentRegistry()
