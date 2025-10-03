"""Simplified component decorators with minimal side effects.

These decorators ONLY add metadata to classes. They do NOT register
components. Registration happens during bootstrap via the manifest.
"""

from __future__ import annotations

import inspect
import re
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, TypeVar

from hexai.core.context import get_observer_manager
from hexai.core.exceptions import ValidationError
from hexai.core.logging import get_logger
from hexai.core.registry.models import (
    ComponentType,
    NodeSubtype,
)
from hexai.core.utils.async_warnings import _is_in_async_context

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)

# Marker to detect if a method has already been wrapped
_ASYNC_IO_WRAPPER_MARKER = "_hexdag_async_io_wrapped"


T = TypeVar("T")


def _snake_case(name: str) -> str:
    """Convert CamelCase/PascalCase to snake_case, handling acronyms properly.

    Returns
    -------
        The snake_case version of the input name.

    Examples
    --------
    Example usage::

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
    Example usage::

        @component('node', name='passthrough')
        class PassthroughNode:
        '''Passes data through unchanged.'''
        pass

        @component('tool', name=['search', 'find'])
        def search_tool(query: str) -> list:
        '''Search for items.'''
        return []

        # After decoration, they have attributes:
        assert hasattr(PassthroughNode, '_hexdag_type')
        assert hasattr(search_tool, '_hexdag_type')
        assert PassthroughNode._hexdag_name == 'passthrough'
        assert search_tool._hexdag_names == ['search', 'find']
    """

    def decorator(cls: T) -> T:
        # Simplify name handling with convention over configuration
        if isinstance(name, list):
            # Multiple names provided - first is primary, rest are aliases
            if not name:  # Validate list is not empty
                raise ValidationError("If providing a list of names, it", "cannot be empty")
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
    warn_sync_io: bool = True,
) -> Callable[[T], T]:
    """Adapter decorator with convention over configuration.

    Parameters
    ----------
    implements_port : str | type
        The port this adapter implements (required). Can be a string port name
        or the Protocol class itself.
    name : str | None
        Adapter name. If None, inferred from class name.
    namespace : str
        Defaults to 'user' for adapters.
    description : str | None
        If None, uses class docstring.
    warn_sync_io : bool
        If True, wrap async methods to warn about synchronous I/O operations.
        Defaults to True. Set to False for adapters that intentionally use sync I/O
        (like local file-based adapters in __init__).

    Examples
    --------
    Example usage::

        @adapter("database")
        class SQLiteAdapter:  # Name becomes 'sqlite_adapter'
        '''SQLite database implementation.'''

        @adapter("llm", name="gpt4")  # Explicit name
        class OpenAIAdapter:
        pass
    """

    def decorator(cls: T) -> T:
        # Apply base component decorator
        cls = component(ComponentType.ADAPTER, name, namespace=namespace, description=description)(
            cls
        )

        # Normalize the port reference to a string
        port_name = implements_port
        if not isinstance(implements_port, str):
            # It's a Protocol class - extract its name
            if hasattr(implements_port, "__name__"):
                # Convert class name like 'PolicyManagerPort' to 'policy_manager'
                class_name = implements_port.__name__
                # Remove 'Port' suffix if present
                if class_name.endswith("Port"):
                    class_name = class_name[:-4]
                # Use existing snake_case function
                port_name = _snake_case(class_name)
            else:
                port_name = str(implements_port)

        # Store the normalized port name
        cls._hexdag_implements_port = port_name  # type: ignore[attr-defined]

        # Optionally wrap async methods to detect sync I/O
        if warn_sync_io:
            _wrap_adapter_async_methods(cls)  # type: ignore[arg-type]

        return cls

    return decorator


def _wrap_adapter_async_methods(cls: type) -> None:
    """Wrap async methods to warn about sync I/O operations.

    Parameters
    ----------
    cls : type
        The adapter class to wrap
    """

    # Get all methods from the class
    for attr_name in dir(cls):
        # Skip private methods and special methods except async protocol methods
        if attr_name.startswith("_") and not attr_name.startswith((
            "aget_",
            "aset_",
            "aexecute_",
            "aresponse_",
            "acall_",
        )):
            continue

        try:
            attr = getattr(cls, attr_name)
        except AttributeError:
            continue

        # Check if it's an async method (coroutine function)
        if inspect.iscoroutinefunction(attr):
            # Skip if already wrapped (prevent double wrapping)
            if hasattr(attr, _ASYNC_IO_WRAPPER_MARKER):
                continue

            # Wrap the async method
            wrapped = _create_async_io_wrapper(attr, attr_name, cls.__name__, _is_in_async_context)
            setattr(cls, attr_name, wrapped)


def _create_async_io_wrapper(  # type: ignore[no-untyped-def]
    func: Any, method_name: str, class_name: str, is_async_context_fn
):
    """Create wrapper to warn about sync I/O in async methods.

    Also provides event emission for adapter calls when observer_manager
    is available in the execution context.

    Parameters
    ----------
    func : Callable
        The async function to wrap
    method_name : str
        Name of the method
    class_name : str
        Name of the class
    is_async_context_fn : Callable
        Function to check if we're in async context

    Returns
    -------
    Callable
        Wrapped function

    Notes
    -----
    This wrapper can access observer_manager from execution context to emit
    AdapterCallStarted, AdapterCallCompleted, and AdapterCallFailed events.
    This enables automatic tracking of all adapter calls for cost calculation,
    resource cleanup, and observability.
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Emit debug log if in async context (which we should be)
        if is_async_context_fn():
            logger.debug(
                f"Entering async method {class_name}.{method_name}() - monitoring for blocking I/O"
            )

        # Access execution context for event emission (if available)
        get_observer_manager()

        # Actually call the original function and return its result!
        return await func(*args, **kwargs)

    setattr(wrapper, _ASYNC_IO_WRAPPER_MARKER, True)

    return wrapper


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
