"""Simplified component decorators with minimal side effects.

These decorators ONLY add metadata to classes. They do NOT register
components. Registration happens during bootstrap via the manifest.
"""

from __future__ import annotations

import re
import warnings
from functools import partial
from typing import TYPE_CHECKING, TypeVar

from hexai.core.registry.models import (
    ComponentType,
    NodeSubtype,
)

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


def _snake_case(name: str) -> str:
    """Convert CamelCase/PascalCase to snake_case, handling acronyms properly.

    Examples
    --------
    XMLHttpRequest -> xml_http_request
    HTMLParser -> html_parser
    SimpleHTTPServer -> simple_http_server
    MyAPIClass -> my_api_class
    IOError -> io_error
    """
    name = re.sub("([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub("([a-z])([A-Z]+)$", r"\1_\2", name)
    return name.lower()


def component(
    component_type: ComponentType | str,
    name: str | list[str] | None = None,
    *,
    namespace: str = "user",
    subtype: NodeSubtype | str | None = None,
    description: str | None = None,
    required_ports: list[str] | None = None,
) -> Callable[[T], T]:
    """Add metadata to components without registering them.

    This decorator ONLY attaches metadata. It does NOT register the component.
    Registration happens during bootstrap when modules are loaded from the manifest.

    Parameters
    ----------
    component_type: ComponentType | str
        Component type (required). Can be enum or string like "node", "tool", etc.
    name: str | list[str] | None
        Component name or list of names for aliases. If None, uses class/function name.
    namespace: str
        Component namespace. Defaults to 'user'.
    subtype: NodeSubtype | str | None
        Optional subtype (mainly for nodes).
    description: str | None
        Component description. If None, uses class/function docstring.
    required_ports: list[str] | None
        Simple list of required port names like ["llm", "database"].

    Returns
    -------
    Callable[[T], T]
        Decorator function that adds metadata to the class or function.

    Examples
    --------
    >>> @component('node', name='passthrough')
    >>> class PassthroughNode:
    ...     '''Passes data through unchanged.'''
    ...     pass
    >>>
    >>> @component('tool', name=['search', 'find'])
    >>> def search_tool(query: str) -> list:
    ...     '''Search for items.'''
    ...     return []
    >>>
    >>> # After decoration, they have attributes:
    >>> assert hasattr(PassthroughNode, '_hexdag_type')
    >>> assert hasattr(search_tool, '_hexdag_type')
    >>> assert PassthroughNode._hexdag_name == 'passthrough'
    >>> assert search_tool._hexdag_names == ['search', 'find']
    """

    def decorator(cls: T) -> T:
        # Simplify name handling with convention over configuration
        if isinstance(name, list):
            # Multiple names provided - first is primary, rest are aliases
            if not name:  # Validate list is not empty
                raise ValueError("If providing a list of names, it cannot be empty")
            all_names = name
            primary_name = name[0]
        elif name:
            # Single name provided
            all_names = [name]
            primary_name = name
        else:
            # No name provided - infer from class/function name
            # Use getattr to safely get __name__ from either class or function
            obj_name = getattr(cls, "__name__", "unknown")
            primary_name = _snake_case(obj_name)
            all_names = [primary_name]

        # Use class docstring as description if not provided
        component_description = description or (cls.__doc__ or "").strip()

        # Validate component type - handle both enum and string
        validated_type: ComponentType
        if isinstance(component_type, ComponentType):
            # Already a ComponentType enum, use it directly
            validated_type = component_type
        elif isinstance(component_type, str):
            # It's a string, convert to enum
            try:
                validated_type = ComponentType(component_type)
            except ValueError:
                raise ValueError(
                    f"Invalid component type '{component_type}'. "
                    f"Must be one of: {', '.join(ComponentType)}"
                ) from None
        else:
            raise TypeError(
                f"component_type must be ComponentType or str, not {type(component_type).__name__}"
            )

        # Store everything as attributes (no metadata object)
        cls._hexdag_type = validated_type  # type: ignore[attr-defined]
        cls._hexdag_name = primary_name  # type: ignore[attr-defined]
        cls._hexdag_names = all_names  # type: ignore[attr-defined]
        cls._hexdag_namespace = namespace  # type: ignore[attr-defined]
        cls._hexdag_subtype = subtype  # type: ignore[attr-defined]
        cls._hexdag_description = component_description  # type: ignore[attr-defined]
        cls._hexdag_required_ports = required_ports or []  # type: ignore[attr-defined]

        return cls

    return decorator


# Port decorator - no manual method lists needed!
def port(
    name: str | None = None,
    *,
    namespace: str = "core",
    description: str | None = None,
) -> Callable[[T], T]:
    """Port decorator using convention over configuration.

    Methods are extracted automatically from the Protocol:
    - Abstract methods (@abstractmethod) are required
    - Concrete methods are optional

    Parameters
    ----------
    name : str | None
        Port name. If None, uses class name in snake_case.
    namespace : str
        Port namespace. Defaults to 'core'.
    description : str | None
        Port description. If None, uses class docstring.
    """
    # Just use the base component decorator
    # Method extraction happens at runtime via inspect
    return component(ComponentType.PORT, name, namespace=namespace, description=description)


def adapter(
    implements_port: str | type,
    name: str | None = None,
    *,
    namespace: str = "user",
    description: str | None = None,
) -> Callable[[T], T]:
    """Adapter decorator with convention over configuration.

    Parameters
    ----------
    implements_port : str | type
        The port this adapter implements (required).
        Prefer using a direct reference to the port class for better type safety.
        String names are deprecated but still supported for backwards compatibility.
    name : str | None
        Adapter name. If None, inferred from class name.
    namespace : str
        Defaults to 'user' for adapters.
    description : str | None
        If None, uses class docstring.

    Examples
    --------
    >>> from hexai.core.ports import PolicyManagerPort
    >>> @adapter(PolicyManagerPort)  # Preferred: Direct port reference
    >>> class CustomPolicyManager:
    ...     pass
    >>>
    >>> @adapter("database")  # Deprecated: String-based reference
    >>> class SQLiteAdapter:  # Name becomes 'sqlite_adapter'
    ...     '''SQLite database implementation.'''
    ...
    """

    def decorator(cls: T) -> T:
        # Apply base component decorator
        cls = component(ComponentType.ADAPTER, name, namespace=namespace, description=description)(
            cls
        )

        # Handle both string and class reference for port
        if isinstance(implements_port, type):
            # It's a class reference - extract the port name (preferred approach)
            # Check if it has the _hexdag_name attribute (decorated port)
            if hasattr(implements_port, "_hexdag_name"):
                port_name = implements_port._hexdag_name
            else:
                # Fallback to snake_case conversion of class name
                port_name = _snake_case(implements_port.__name__)
                # Remove 'Port' suffix if present
                if port_name.endswith("_port"):
                    port_name = port_name[:-5]
        else:
            # It's a string (deprecated but supported)
            port_name = implements_port
            # Issue deprecation warning
            adapter_name = cls._hexdag_name if hasattr(cls, "_hexdag_name") else cls.__name__  # type: ignore[attr-defined]
            warnings.warn(
                f"Using string '{port_name}' for implements_port in @adapter decorator for"
                f"'{adapter_name}' is deprecated."
                f"Please use a direct reference to the port class instead."
                f"'{adapter_name}' is deprecated. "
                f"Please use a direct reference to the port class instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Store the implemented port name
        cls._hexdag_implements_port = port_name  # type: ignore[attr-defined]

        return cls

    return decorator


tool = partial(component, ComponentType.TOOL)
node = partial(component, ComponentType.NODE)
policy = partial(component, ComponentType.POLICY)
memory = partial(component, ComponentType.MEMORY)
observer = partial(component, ComponentType.OBSERVER)


def function_node(
    name: str | None = None, *, namespace: str = "core", description: str | None = None
) -> Callable[[T], T]:
    """Decorator for function nodes."""
    return component(
        ComponentType.NODE,
        name,
        namespace=namespace,
        subtype=NodeSubtype.FUNCTION,
        description=description,
    )


def llm_node(
    name: str | None = None, *, namespace: str = "core", description: str | None = None
) -> Callable[[T], T]:
    """Decorator for LLM nodes."""
    return component(
        ComponentType.NODE,
        name,
        namespace=namespace,
        subtype=NodeSubtype.LLM,
        description=description,
    )


def agent_node(
    name: str | None = None, *, namespace: str = "core", description: str | None = None
) -> Callable[[T], T]:
    """Decorator for agent nodes."""
    return component(
        ComponentType.NODE,
        name,
        namespace=namespace,
        subtype=NodeSubtype.AGENT,
        description=description,
    )
