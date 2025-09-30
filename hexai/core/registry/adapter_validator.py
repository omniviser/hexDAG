"""Adapter-specific validation logic."""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING

from hexai.core.registry.exceptions import (
    ComponentNotFoundError,
    InvalidComponentError,
)
from hexai.core.registry.models import ComponentMetadata, ComponentType
from hexai.core.registry.validation import RegistryValidator

if TYPE_CHECKING:
    from hexai.core.registry.component_store import ComponentStore

logger = logging.getLogger(__name__)


class AdapterValidator:
    """Validates adapter implementations against ports.

    This class handles:
    - Adapter port declaration validation
    - Port existence checking
    - Port implementation verification
    """

    def __init__(self, store: ComponentStore) -> None:
        """Initialize adapter validator.

        Args
        ----
        store : ComponentStore
            Component store to look up ports.
        """
        self._store = store

    def validate_adapter_registration(
        self,
        adapter_name: str,
        adapter_component: object,
        namespace: str,
    ) -> None:
        """Validate adapter implementation at registration time.

        Args
        ----
        adapter_name : str
            Name of the adapter being registered.
        adapter_component : object
            The adapter class/instance being registered.
        namespace : str
            Namespace where adapter is being registered.

        Raises
        ------
        InvalidComponentError
            If adapter doesn't properly implement its declared port.
        """
        # Get adapter metadata from _hexdag_implements_port attribute
        implements_port = self._extract_implements_port(adapter_component)

        if not implements_port:
            # No port declared, skip validation
            return

        # Find the port in registry
        port_meta = self._find_port(implements_port, namespace, adapter_name)

        if not port_meta:
            raise InvalidComponentError(
                adapter_name,
                f"Adapter '{adapter_name}' declares it implements port '{implements_port}', "
                f"but port '{implements_port}' does not exist in registry. "
                f"Available ports: {', '.join(self._store.get_available_ports())}",
            )

        # Validate implementation
        self._validate_implementation(adapter_name, adapter_component, implements_port, port_meta)

    def _extract_implements_port(self, adapter_component: object) -> str | None:
        """Extract the port name from adapter metadata.

        Args
        ----
        adapter_component : object
            Adapter component to inspect.

        Returns
        -------
        str | None
            Port name if declared, None otherwise.
        """
        return RegistryValidator.get_implements_port(adapter_component)

    def _find_port(
        self,
        port_name: str,
        adapter_namespace: str,
        adapter_name: str,
    ) -> ComponentMetadata | None:
        """Find port in registry with multiple search strategies.

        Args
        ----
        port_name : str
            Port name to find.
        adapter_namespace : str
            Namespace of the adapter (for same-namespace search).
        adapter_name : str
            Adapter name (for logging).

        Returns
        -------
        ComponentMetadata | None
            Port metadata if found, None otherwise.
        """
        search_attempts = self._build_search_attempts(port_name, adapter_namespace)

        for attempt in search_attempts:
            try:
                port_meta = self._store.get_metadata(attempt, component_type=ComponentType.PORT)
                if port_meta:
                    logger.debug(
                        "Found port '%s' for adapter '%s' as '%s'",
                        port_name,
                        adapter_name,
                        attempt,
                    )
                    return port_meta
            except ComponentNotFoundError:
                continue

        return None

    def _build_search_attempts(self, port_name: str, adapter_namespace: str) -> list[str]:
        """Build list of search attempts for finding a port.

        Args
        ----
        port_name : str
            Port name (may be qualified).
        adapter_namespace : str
            Adapter's namespace.

        Returns
        -------
        list[str]
            List of qualified names to try.
        """
        if ":" in port_name:
            # Qualified name provided
            return [port_name]

        # Unqualified - search with priority
        # Follow DEFAULT_SEARCH_PRIORITY: "core", "user", "plugin"
        return [
            f"core:{port_name}",  # Core namespace first
            f"{adapter_namespace}:{port_name}",  # Same namespace as adapter
            port_name,  # As declared (will search all)
        ]

    def _validate_implementation(
        self,
        adapter_name: str,
        adapter_component: object,
        port_name: str,
        port_meta: ComponentMetadata,
    ) -> None:
        """Validate that adapter implements all required port methods.

        Args
        ----
        adapter_name : str
            Adapter name.
        adapter_component : object
            Adapter class/instance.
        port_name : str
            Port name.
        port_meta : ComponentMetadata
            Port metadata.

        Raises
        ------
        InvalidComponentError
            If adapter is missing required methods.
        """
        # Get the port Protocol class
        port_class = port_meta.raw_component
        if not port_class:
            return

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
                f"from port '{port_name}': {', '.join(missing_methods)}",
            )
