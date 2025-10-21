"""Simplified component registry - storage and lifecycle merged inline."""

from __future__ import annotations

import collections.abc
import importlib.util
from threading import Lock
from typing import TYPE_CHECKING, Any

from hexdag.core.logging import get_logger
from hexdag.core.registry.discovery import register_components as default_register_components
from hexdag.core.registry.exceptions import (
    ComponentAlreadyRegisteredError,
    ComponentNotFoundError,
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
    from hexdag.core.config import ManifestEntry

logger = get_logger(__name__)


class ComponentRegistry:
    """Central component registry with bootstrap-based initialization.

    Lifecycle:
    1. Empty at creation
    2. Populated via bootstrap(manifest)
    3. Read-only after bootstrap (unless dev_mode)

    This merges what was previously split across ComponentStore and BootstrapManager.

    Note
    ----
    Registry now uses flat storage - namespace is stripped from component names.
    Format "namespace:name" is allowed but namespace prefix is removed during registration.
    Component names must be globally unique across all namespaces.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        # Flat storage - namespace is metadata only, not used for lookup
        self._components: dict[str, ComponentMetadata] = {}

        # Bootstrap state (was BootstrapManager)
        self._ready = False
        self._manifest: list[ManifestEntry] | None = None
        self._dev_mode = False
        self._bootstrap_context = False

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

            self._manifest = manifest
            self._dev_mode = dev_mode
            logger.info("Bootstrapping registry with {count} entries", count=len(manifest))

            # Load modules and register components
            self._bootstrap_context = True
            total_registered = 0

            try:
                for entry in manifest:
                    is_core = self._is_core_module(entry)

                    if not is_core and (
                        skip_reason := self._check_plugin_requirements(entry.module)
                    ):
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
                    except ComponentAlreadyRegisteredError:
                        logger.error(
                            "Failed to register components from {module}: {error}",
                            module=entry.module,
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
            if not is_core and self._check_plugin_requirements(entry.module):
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
        subtype: NodeSubtype | str | None = None,
        description: str = "",
    ) -> ComponentMetadata:
        """Register a component in the registry.

        After bootstrap, registration only allowed in dev mode.

        Note
        ----
        Namespace is stored as metadata but stripped from the component name.
        If name contains "namespace:component_name", the namespace prefix is removed.
        Component names must be globally unique.

        Returns
        -------
        ComponentMetadata
            Metadata for the registered component

        Raises
        ------
        RegistryImmutableError
            If registry is read-only and not in dev mode
        ComponentAlreadyRegisteredError
            If component name already exists (collision detected)
        """
        if not self._can_register():
            raise RegistryImmutableError(
                f"Cannot register component '{name}' after bootstrap. "
                f"Registry is read-only in production mode. "
                f"Use dev_mode=True in bootstrap() for development."
            )

        # Strip namespace from name if present (e.g., "core:retry" -> "retry")
        clean_name = name.split(NAMESPACE_SEPARATOR)[-1] if NAMESPACE_SEPARATOR in name else name

        # Validate inputs
        namespace_str = RegistryValidator.validate_namespace(namespace)
        component_type_enum = RegistryValidator.validate_component_type(component_type)
        wrapped_component = RegistryValidator.wrap_component(component)
        RegistryValidator.validate_component_name(clean_name)

        implements_port_str = (
            RegistryValidator.get_implements_port(component)
            if component_type_enum == ComponentType.ADAPTER
            else None
        )
        port_requirements_list = RegistryValidator.get_required_ports(component)

        metadata = ComponentMetadata(
            name=clean_name,
            component_type=component_type_enum,
            component=wrapped_component,
            namespace=namespace_str,
            subtype=subtype,
            description=description,
            implements_port=implements_port_str,
            port_requirements=port_requirements_list,
        )

        # Collision detection - component names must be globally unique
        if clean_name in self._components:
            existing = self._components[clean_name]
            # ComponentAlreadyRegisteredError expects (name, namespace) signature
            raise ComponentAlreadyRegisteredError(clean_name, existing.namespace)

        self._components[clean_name] = metadata

        logger.debug(
            "Registered {name} (namespace: {namespace})", name=clean_name, namespace=namespace_str
        )
        return metadata

    # ========================================================================
    # Component Retrieval (inline from ComponentStore)
    # ========================================================================

    def get_metadata(
        self, name: str, namespace: str | None = None, component_type: ComponentType | None = None
    ) -> ComponentMetadata:
        """Get component metadata without instantiation.

        Note
        ----
        Namespace parameter is ignored - kept for API compatibility.
        Component names are unique and stored in flat dict.
        If name contains "namespace:component", the namespace prefix is stripped.
        """
        # Strip namespace if present in name (e.g., "core:retry" -> "retry")
        clean_name = name.split(NAMESPACE_SEPARATOR)[-1] if NAMESPACE_SEPARATOR in name else name

        # Direct lookup in flat dict
        metadata = self._components.get(clean_name)

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
        """Get and instantiate a component.

        Note
        ----
        Namespace parameter is ignored - kept for API compatibility.
        """
        metadata = self.get_metadata(name, namespace)
        return InstanceFactory.create_instance(metadata.component, init_params)

    def get_info(self, name: str, namespace: str | None = None) -> ComponentInfo:
        """Get detailed component information.

        Note
        ----
        Namespace parameter is ignored - kept for API compatibility.
        """
        # Strip namespace if present in name
        clean_name = name.split(NAMESPACE_SEPARATOR)[-1] if NAMESPACE_SEPARATOR in name else name

        metadata = self._components.get(clean_name)
        if not metadata:
            raise ComponentNotFoundError(name, namespace, self._get_available_components())

        qualified_name = f"{metadata.namespace}.{clean_name}" if metadata.namespace else clean_name
        return ComponentInfo(
            name=clean_name,
            namespace=metadata.namespace,
            qualified_name=qualified_name,
            component_type=metadata.component_type,
            metadata=metadata,
            is_protected=False,  # Protection removed - kept for API compatibility
        )

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
        cache_key = f"{namespace or 'auto'}:{name}:{format}"

        # Check cache
        if cache_key in self._schema_cache:
            return self._schema_cache[cache_key]

        metadata = self.get_metadata(name, namespace)

        import inspect
        from typing import cast

        component: Any = metadata.raw_component

        # If it's a class, get the __call__ method
        if inspect.isclass(component) and callable(component):
            try:
                instance = component()
                component = instance.__call__
            except Exception:
                # Fallback: try to get __call__ from class
                component = component.__call__

        # Generate schema (component is now guaranteed to be callable)
        schema = SchemaGenerator.from_callable(
            cast("collections.abc.Callable", component), format=format
        )

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
        """List components matching criteria.

        Note
        ----
        Namespace filtering still works - namespace is stored in metadata.
        """
        results = []

        for name, metadata in self._components.items():
            if namespace and metadata.namespace != namespace:
                continue
            if component_type and metadata.component_type != component_type:
                continue
            if subtype and metadata.subtype != subtype:
                continue

            qualified_name = f"{metadata.namespace}.{name}" if metadata.namespace else name
            results.append(
                ComponentInfo(
                    name=name,
                    namespace=metadata.namespace,
                    qualified_name=qualified_name,
                    component_type=metadata.component_type,
                    metadata=metadata,
                    is_protected=False,  # Protection removed - kept for API compatibility
                )
            )

        return results

    def get_adapters_for_port(self, port_name: str) -> list[ComponentMetadata]:
        """Get all adapters implementing a specific port."""
        adapters = []

        for metadata in self._components.values():
            if metadata.component_type != ComponentType.ADAPTER:
                continue

            implements = metadata.implements_port or RegistryValidator.get_implements_port(
                metadata.component
            )

            if implements and self._port_names_match(port_name, implements):
                adapters.append(metadata)

        return adapters

    # ========================================================================
    # Private Helpers
    # ========================================================================

    def _get_available_components(self) -> list[str]:
        """Get list of all available component names."""
        return list(self._components.keys())

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
        self._schema_cache.clear()
        self._ready = False
        self._manifest = None
        self._dev_mode = False
        self._bootstrap_context = False


# Global registry instance
registry = ComponentRegistry()
