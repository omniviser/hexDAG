"""Dependency injection container for port-adapter binding and resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hexai.core import registry as registry
from hexai.core.registry.models import ComponentType, InstanceFactory


@dataclass
class PortRequirement:
    """Specification for a port dependency requirement."""

    port_name: str
    adapter_name: str | None = None
    optional: bool = False
    capabilities: list[str] | None = None
    field_name: str | None = None  # Where to inject in __init__


class DIContainerError(Exception):
    """Base exception for DI container errors."""

    pass


class PortNotBoundError(DIContainerError):
    """Raised when trying to resolve an unbound port."""

    pass


class AdapterNotFoundError(DIContainerError):
    """Raised when specified adapter doesn't exist."""

    pass


class PortAdapterMismatchError(DIContainerError):
    """Raised when adapter doesn't implement required port."""

    pass


@dataclass
class DIContainer:
    """Dependency injection container for managing port-adapter bindings.

    Handles:
    - Port to adapter bindings
    - Adapter instance resolution with singleton support
    - Validation that adapters implement their declared ports
    - Capability checking
    """

    def __init__(self) -> None:
        """Initialize the DI container."""
        self._bindings: dict[str, str] = {}  # port_name -> default adapter_name
        self._named_bindings: dict[
            str, dict[str, str]
        ] = {}  # port_name -> {binding_name: adapter_name}
        self._singleton_cache: dict[str, Any] = {}  # adapter_name -> instance
        self._adapter_configs: dict[str, dict[str, Any]] = {}  # adapter_name -> config

    def bind_port(
        self,
        port_name: str,
        adapter_name: str,
        config: dict[str, Any] | None = None,
        binding_name: str | None = None,
        set_as_default: bool = True,
    ) -> None:
        """Bind a port to a specific adapter implementation.

        Parameters
        ----------
        port_name : str
            Name of the port to bind
        adapter_name : str
            Name of the adapter to use for this port
        config : dict[str, Any] | None
            Configuration to pass to adapter initialization
        binding_name : str | None
            Optional name for this binding (for multiple adapters per port)
        set_as_default : bool
            Whether to set this as the default adapter for the port

        Raises
        ------
        PortNotFoundError
            If the port doesn't exist in registry
        AdapterNotFoundError
            If the adapter doesn't exist in registry
        PortAdapterMismatchError
            If adapter doesn't implement the port
        """
        # Validate port exists
        try:
            port_meta = registry.get_metadata(port_name, component_type=ComponentType.PORT)
            if not port_meta:
                raise DIContainerError(f"Port '{port_name}' not found in registry")
        except Exception as e:
            # Wrap registry exceptions as DI container errors
            raise DIContainerError(f"Port '{port_name}' not found in registry") from e

        # Validate adapter exists
        try:
            adapter_meta = registry.get_metadata(adapter_name, component_type=ComponentType.ADAPTER)
            if not adapter_meta:
                raise AdapterNotFoundError(f"Adapter '{adapter_name}' not found in registry")
        except Exception as e:
            # Wrap registry exceptions as adapter not found
            raise AdapterNotFoundError(f"Adapter '{adapter_name}' not found in registry") from e

        # Validate adapter implements port
        if not self.validate_adapter_implements_port(adapter_name, port_name):
            raise PortAdapterMismatchError(
                f"Adapter '{adapter_name}' does not implement port '{port_name}'"
            )

        # Store binding and config
        if set_as_default or port_name not in self._bindings:
            self._bindings[port_name] = adapter_name

        # Store named binding if provided
        if binding_name:
            if port_name not in self._named_bindings:
                self._named_bindings[port_name] = {}
            self._named_bindings[port_name][binding_name] = adapter_name

        if config:
            self._adapter_configs[adapter_name] = config

    def resolve_port(
        self,
        requirement: PortRequirement,
        binding_name: str | None = None,
    ) -> Any | None:
        """Resolve a port requirement to an adapter instance.

        Parameters
        ----------
        requirement : PortRequirement
            Specification of the port dependency
        binding_name : str | None
            Optional binding name for multi-adapter scenarios

        Returns
        -------
        Any | None
            Adapter instance or None if optional and not bound

        Raises
        ------
        PortNotBoundError
            If required port is not bound
        AdapterNotFoundError
            If specified adapter doesn't exist
        """
        # Determine which adapter to use
        adapter_name = None

        # Priority: explicit adapter > named binding > default binding
        if hasattr(requirement, "adapter_name") and requirement.adapter_name:
            adapter_name = requirement.adapter_name
        elif binding_name and requirement.port_name in self._named_bindings:
            bindings = self._named_bindings[requirement.port_name]
            if binding_name in bindings:
                adapter_name = bindings[binding_name]
            else:
                raise PortNotBoundError(
                    f"No binding named '{binding_name}' for port '{requirement.port_name}'"
                )
        elif requirement.port_name in self._bindings:
            adapter_name = self._bindings[requirement.port_name]
        elif requirement.optional:
            return None
        else:
            raise PortNotBoundError(
                f"Required port '{requirement.port_name}' is not bound to any adapter"
            )

        if not adapter_name:
            if requirement.optional:
                return None
            raise PortNotBoundError(
                f"Required port '{requirement.port_name}' is not bound to any adapter"
            )

        # Get adapter metadata
        try:
            adapter_meta = registry.get_metadata(adapter_name, component_type=ComponentType.ADAPTER)
            if not adapter_meta:
                raise AdapterNotFoundError(f"Adapter '{adapter_name}' not found in registry")
        except Exception as e:
            # Wrap registry exceptions as adapter not found
            raise AdapterNotFoundError(f"Adapter '{adapter_name}' not found in registry") from e

        # Check capabilities if specified
        if requirement.capabilities and adapter_meta.adapter_metadata:
            adapter_caps = set(adapter_meta.adapter_metadata.capabilities)
            required_caps = set(requirement.capabilities)
            if not required_caps.issubset(adapter_caps):
                missing = required_caps - adapter_caps
                raise DIContainerError(
                    f"Adapter '{adapter_name}' missing required capabilities: {missing}"
                )

        # Handle singleton instances
        if adapter_meta.adapter_metadata and adapter_meta.adapter_metadata.singleton:
            if adapter_name not in self._singleton_cache:
                # Create and cache singleton instance
                instance = self._create_adapter_instance(adapter_name, adapter_meta)
                self._singleton_cache[adapter_name] = instance
            return self._singleton_cache[adapter_name]
        else:
            # Create new instance
            return self._create_adapter_instance(adapter_name, adapter_meta)

    def validate_adapter_implements_port(self, adapter_name: str, port_name: str) -> bool:
        """Validate that an adapter implements a specific port.

        Parameters
        ----------
        adapter_name : str
            Name of the adapter to check
        port_name : str
            Name of the port to validate against

        Returns
        -------
        bool
            True if adapter implements port, False otherwise
        """
        try:
            adapter_meta = registry.get_metadata(adapter_name, component_type=ComponentType.ADAPTER)
            if not adapter_meta or not adapter_meta.adapter_metadata:
                return False
        except Exception:
            # Adapter doesn't exist
            return False

        # Check if adapter declares it implements this port
        # Handle both namespaced and non-namespaced port names
        implements = adapter_meta.adapter_metadata.implements_port
        port_name_parts = port_name.split(":")
        port_base_name = port_name_parts[-1] if port_name_parts else port_name

        # Compare either full name or just base name
        if implements != port_name and implements != port_base_name:
            return False

        # Get port metadata to check required methods
        try:
            port_meta = registry.get_metadata(port_name, component_type=ComponentType.PORT)
            if not port_meta or not port_meta.port_metadata:
                return True  # No specific requirements
        except Exception:
            # Port doesn't exist - can't validate
            return True  # Assume it's ok if port doesn't exist

        # Check that adapter class has required methods
        adapter_class = adapter_meta.raw_component
        required_methods = port_meta.port_metadata.required_methods

        return all(hasattr(adapter_class, method_name) for method_name in required_methods)

    def _create_adapter_instance(self, adapter_name: str, adapter_meta: Any) -> Any:
        """Create an instance of an adapter.

        Parameters
        ----------
        adapter_name : str
            Name of the adapter
        adapter_meta : Any
            Adapter metadata from registry

        Returns
        -------
        Any
            New adapter instance
        """

        # Get configuration if provided
        config = self._adapter_configs.get(adapter_name, {})

        # Create instance using the factory
        return InstanceFactory.create_instance(adapter_meta.component, config)

    def clear_singletons(self) -> None:
        """Clear all cached singleton instances."""
        self._singleton_cache.clear()

    def clear_bindings(self) -> None:
        """Clear all port-adapter bindings."""
        self._bindings.clear()
        self._named_bindings.clear()
        self._adapter_configs.clear()

    def get_binding(self, port_name: str) -> str | None:
        """Get the adapter bound to a port.

        Parameters
        ----------
        port_name : str
            Name of the port

        Returns
        -------
        str | None
            Name of bound adapter or None if not bound
        """
        return self._bindings.get(port_name)

    def list_bindings(self) -> dict[str, str]:
        """Get all current default port-adapter bindings.

        Returns
        -------
        dict[str, str]
            Dictionary of port_name -> adapter_name bindings
        """
        return self._bindings.copy()

    def list_named_bindings(self, port_name: str) -> dict[str, str]:
        """Get all named bindings for a specific port.

        Parameters
        ----------
        port_name : str
            Name of the port

        Returns
        -------
        dict[str, str]
            Dictionary of binding_name -> adapter_name
        """
        return self._named_bindings.get(port_name, {}).copy()

    def get_all_bindings(self, port_name: str) -> list[str]:
        """Get all adapters bound to a port (default + named).

        Parameters
        ----------
        port_name : str
            Name of the port

        Returns
        -------
        list[str]
            List of all adapter names bound to this port
        """
        adapters = set()

        # Add default binding
        if port_name in self._bindings:
            adapters.add(self._bindings[port_name])

        # Add named bindings
        if port_name in self._named_bindings:
            adapters.update(self._named_bindings[port_name].values())

        return list(adapters)

    def create_with_dependencies(
        self,
        component_name: str,
        component_type: ComponentType | None = None,
        init_params: dict[str, Any] | None = None,
    ) -> Any:
        """Create a component instance with its declared port dependencies injected.

        Parameters
        ----------
        component_name : str
            Name of the component to create
        component_type : ComponentType | None
            Type of component (for filtering)
        init_params : dict[str, Any] | None
            Additional initialization parameters

        Returns
        -------
        Any
            Component instance with dependencies injected

        Raises
        ------
        DIContainerError
            If required ports are not bound
        """
        # Get component metadata
        metadata = registry.get_metadata(component_name, component_type=component_type)
        if not metadata:
            raise DIContainerError(f"Component '{component_name}' not found")

        # Prepare initialization parameters
        params = init_params or {}

        # Resolve port requirements
        for requirement in metadata.port_requirements:
            # Resolve the port
            adapter_instance = self.resolve_port(requirement)

            # Determine injection field name
            field_name = requirement.field_name or f"{requirement.port_name}_port"

            # Only inject if not already provided in init_params
            if field_name not in params:
                if adapter_instance is None and not requirement.optional:
                    raise PortNotBoundError(
                        f"Required port '{requirement.port_name}' "
                        f"for component '{component_name}'"
                        "is not bound"
                    )
                params[field_name] = adapter_instance

        # Create instance with injected dependencies
        return InstanceFactory.create_instance(metadata.component, params)

    def get_component_requirements(self, component_name: str) -> list[PortRequirement]:
        """Get the port requirements for a component.

        Parameters
        ----------
        component_name : str
            Name of the component

        Returns
        -------
        list[PortRequirement]
            List of port requirements
        """
        metadata = registry.get_metadata(component_name)
        return metadata.port_requirements if metadata else []
