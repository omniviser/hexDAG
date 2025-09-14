"""Simplified component decorators with minimal side effects.

These decorators ONLY add metadata to classes. They do NOT register
components. Registration happens during bootstrap via the manifest.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

from hexai.core.registry.models import (
    AdapterMetadata,
    ComponentType,
    DecoratorMetadata,
    NodeSubtype,
)

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
    name: str | None = None,
    *,
    namespace: str = "user",
    subtype: NodeSubtype | str | None = None,
    description: str | None = None,
) -> Callable[[type[T]], type[T]]:
    """Add metadata to components without registering them.

    This decorator ONLY attaches metadata. It does NOT register the component.
    Registration happens during bootstrap when modules are loaded from the manifest.

    Parameters
    ----------
    component_type: ComponentType | str
        Component type (required). Can be enum or string like "node", "tool", etc.
    name: str | None
        Component name. If None, uses class name in snake_case.
    namespace: str
        Component namespace. Defaults to 'user'.
    subtype: NodeSubtype | str | None
        Optional subtype (mainly for nodes).
    description: str | None
        Component description. If None, uses class docstring.

    Returns
    -------
    Callable[[type[T]], type[T]]
        Decorator function that adds metadata to the class.

    Examples
    --------
    >>> @component('node', name='passthrough')
    >>> class PassthroughNode:
    ...     '''Passes data through unchanged.'''
    ...     pass
    >>>
    >>> # After decoration, the class has metadata:
    >>> assert hasattr(PassthroughNode, '__hexdag_metadata__')
    >>> assert PassthroughNode.__hexdag_metadata__.type == 'node'
    """

    def decorator(cls: type[T]) -> type[T]:
        # Infer name from class name if not provided
        component_name = name or _snake_case(cls.__name__)

        # Use class docstring as description if not provided
        component_description = description or (cls.__doc__ or "").strip()

        # Validate component type if it's a string
        if isinstance(component_type, str):
            try:
                validated_type: ComponentType | str = ComponentType(component_type)
            except ValueError:
                # Invalid type - fail fast
                raise ValueError(
                    f"Invalid component type '{component_type}'. "
                    f"Must be one of: {', '.join(ComponentType)}"
                ) from None
        else:
            validated_type = component_type  # type: ignore[unreachable]

        metadata = DecoratorMetadata(
            type=validated_type,
            name=component_name,
            declared_namespace=namespace,
            subtype=subtype,
            description=component_description,
        )

        # Attach metadata to the class
        cls.__hexdag_metadata__ = metadata  # type: ignore[attr-defined]

        return cls

    return decorator


def make_component_decorator(
    component_type: ComponentType | str,
    subtype: NodeSubtype | str | None = None,
) -> Callable[..., Callable[[type[T]], type[T]]]:
    """Create specialized component decorators.

    Parameters
    ----------
    component_type : ComponentType | str
        The type of component this decorator creates.
    subtype : NodeSubtype | str | None
        Optional subtype to set automatically.

    Returns
    -------
    Callable
        A decorator function with the type/subtype pre-configured.
    """

    def wrapper(
        name: str | None = None,
        *,
        namespace: str = "user",
        description: str | None = None,
        **kwargs: str | None,  # Allows subtype override when needed
    ) -> Callable[[type[T]], type[T]]:
        # If subtype is provided via factory, it takes precedence
        actual_subtype = subtype or kwargs.get("subtype")
        return component(
            component_type,
            name,
            namespace=namespace,
            subtype=actual_subtype,
            description=description,
        )

    return wrapper


def adapter(
    implements_port: str,
    name: str | None = None,
    *,
    namespace: str = "user",
    description: str | None = None,
    capabilities: list[str] | None = None,
    singleton: bool = True,
) -> Callable[[type[T]], type[T]]:
    """Decorator for adapter components that implement ports.

    Parameters
    ----------
    implements_port : str
        Name of the port this adapter implements (required)
    name : str | None
        Adapter name. If None, uses class name in snake_case
    namespace : str
        Component namespace. Defaults to 'user'
    description : str | None
        Adapter description. If None, uses class docstring
    capabilities : list[str] | None
        List of capabilities this adapter provides
    singleton : bool
        Whether adapter should be a singleton (default True)

    Returns
    -------
    Callable[[type[T]], type[T]]
        Decorator function that adds metadata to the class

    Examples
    --------
    >>> @adapter(implements_port='llm_port')
    >>> class OpenAIAdapter:
    ...     def generate(self, prompt: str) -> str:
    ...         return "response"
    """

    def decorator(cls: type[T]) -> type[T]:
        # Infer name from class name if not provided
        adapter_name = name or _snake_case(cls.__name__)

        # Use class docstring as description if not provided
        adapter_description = description or (cls.__doc__ or "").strip()

        # Create adapter-specific metadata
        adapter_meta = AdapterMetadata(
            implements_port=implements_port,
            capabilities=capabilities or [],
            singleton=singleton,
        )

        # Create complete metadata with adapter info
        metadata = DecoratorMetadata(
            type=ComponentType.ADAPTER,
            name=adapter_name,
            declared_namespace=namespace,
            description=adapter_description,
            adapter_metadata=adapter_meta,
        )

        # Attach metadata to the class
        cls.__hexdag_metadata__ = metadata  # type: ignore[attr-defined]

        return cls

    return decorator


# Generate base type decorators using the factory
node = make_component_decorator(ComponentType.NODE)
tool = make_component_decorator(ComponentType.TOOL)
policy = make_component_decorator(ComponentType.POLICY)
memory = make_component_decorator(ComponentType.MEMORY)
observer = make_component_decorator(ComponentType.OBSERVER)

# Generate node subtype decorators
function_node = make_component_decorator(ComponentType.NODE, subtype=NodeSubtype.FUNCTION)
llm_node = make_component_decorator(ComponentType.NODE, subtype=NodeSubtype.LLM)
agent_node = make_component_decorator(ComponentType.NODE, subtype=NodeSubtype.AGENT)
