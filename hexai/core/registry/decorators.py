"""Decorators for automatic component registration.

These decorators automatically register components with the registry.
If you don't want auto-registration, simply don't use the decorator
and register manually with registry.register().
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, TypeVar

from hexai.core.registry.metadata import ComponentMetadata
from hexai.core.registry.types import ComponentType

T = TypeVar("T")

# Global registry of decorated components waiting to be registered
_pending_components: list[tuple[type, ComponentMetadata, str]] = []


def component(
    name: str | None = None,
    *,
    component_type: ComponentType | str | None = None,
    version: str | None = None,
    description: str | None = None,
    tags: set[str] | frozenset[str] | None = None,
    author: str = "hexdag",
    dependencies: set[str] | frozenset[str] | None = None,
    namespace: str,
    replaceable: bool = False,
) -> Callable[[type[T]], type[T]]:
    """Decorator for component registration.

    Marks a class as a component and optionally auto-registers it with
    the registry when the module is imported.

    Parameters
    ----------
    name : str | None
        Component name. If None, uses class name in snake_case.
    component_type : ComponentTypeEnum | None
        Component type. If None, inferred from class name or base class.
    version : str | None
        Component version.
    description : str | None
        Component description. If None, uses class docstring.
    tags : set[str] | frozenset[str] | None
        Component tags for categorization.
    author : str
        Component author (defaults to 'hexdag').
    dependencies : set[str] | frozenset[str] | None
        Other components this component depends on.
    namespace : str
        Required namespace for the component.
        Use 'hexai' for first-party or your plugin name for third-party.
    replaceable : bool
        If True, component can be replaced after registration.
        Defaults to False for production stability.

    Returns
    -------
    Callable[[type[T]], type[T]]
        Decorated class with metadata attached.

    Examples
    --------
    >>> # Decorator automatically registers the component
    >>> @component(
    ...     namespace='my_plugin',  # Required
    ...     component_type=ComponentType.NODE,
    ...     tags={'analysis', 'nlp'}
    ... )
    ... class SentimentAnalyzer:
    ...     '''Analyzes sentiment of text.'''
    ...     pass
    >>> # Component is now registered and available
    >>>
    >>> # For manual registration, don't use decorator:
    >>> class ManualComponent:
    ...     pass
    >>> registry.register(
    ...     name='manual',
    ...     component=ManualComponent(),
    ...     component_type=ComponentType.NODE,
    ...     namespace='my_plugin'
    ... )

    >>> @component()  # All conventions applied
    ... class DataProcessorNode:
    ...     '''Processes incoming data.'''
    ...     pass
    """

    def decorator(cls: type[T]) -> type[T]:
        # Infer name from class name if not provided
        component_name = name or _snake_case(cls.__name__)

        # Infer component type from class name or inheritance
        inferred_type = component_type or _infer_component_type(cls)
        if not inferred_type:
            raise ValueError(
                f"Could not infer component type for {cls.__name__}. "
                "Please specify component_type explicitly or follow naming conventions "
                "(e.g., suffix with Node, Tool, Adapter, Policy, Memory, Observer)"
            )

        # Use class docstring as description if not provided
        component_description = description or (inspect.getdoc(cls) or f"{cls.__name__} component")

        # Create metadata
        metadata = ComponentMetadata(
            name=component_name,
            component_type=inferred_type,
            version=version,
            description=component_description,
            tags=frozenset(tags) if tags else frozenset(),
            author=author,
            dependencies=frozenset(dependencies) if dependencies else frozenset(),
            replaceable=replaceable,
        )

        # Attach metadata to class
        cls.__component_metadata__ = metadata  # type: ignore
        cls.__component_namespace__ = namespace  # type: ignore

        # Add to pending registrations (will be registered when registry initializes)
        _pending_components.append((cls, metadata, namespace))

        return cls

    return decorator


def node(namespace: str, name: str | None = None, **kwargs: Any) -> Callable[[type[T]], type[T]]:
    """Shorthand decorator for node components.

    Examples
    --------
    >>> @node(namespace='my_plugin')
    ... class DataProcessor:
    ...     pass
    """
    return component(name, namespace=namespace, component_type=ComponentType.NODE, **kwargs)


def tool(namespace: str, name: str | None = None, **kwargs: Any) -> Callable[[type[T]], type[T]]:
    """Shorthand decorator for tool components.

    Examples
    --------
    >>> @tool()
    ... class WebScraper:
    ...     pass
    """
    return component(name, namespace=namespace, component_type=ComponentType.TOOL, **kwargs)


def adapter(namespace: str, name: str | None = None, **kwargs: Any) -> Callable[[type[T]], type[T]]:
    """Shorthand decorator for adapter components.

    Examples
    --------
    >>> @adapter()
    ... class PostgresAdapter:
    ...     pass
    """
    return component(name, namespace=namespace, component_type=ComponentType.ADAPTER, **kwargs)


def policy(namespace: str, name: str | None = None, **kwargs: Any) -> Callable[[type[T]], type[T]]:
    """Shorthand decorator for policy components.

    Examples
    --------
    >>> @policy()
    ... class RetryPolicy:
    ...     pass
    """
    return component(name, namespace=namespace, component_type=ComponentType.POLICY, **kwargs)


def memory(namespace: str, name: str | None = None, **kwargs: Any) -> Callable[[type[T]], type[T]]:
    """Shorthand decorator for memory components.

    Examples
    --------
    >>> @memory()
    ... class ConversationMemory:
    ...     pass
    """
    return component(name, namespace=namespace, component_type=ComponentType.MEMORY, **kwargs)


def observer(
    namespace: str, name: str | None = None, **kwargs: Any
) -> Callable[[type[T]], type[T]]:
    """Shorthand decorator for observer components.

    Examples
    --------
    >>> @observer()
    ... class MetricsObserver:
    ...     pass
    """
    return component(name, namespace=namespace, component_type=ComponentType.OBSERVER, **kwargs)


def _snake_case(name: str) -> str:
    """Convert CamelCase to snake_case."""
    import re

    # Insert underscore before uppercase letters
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    # Insert underscore before uppercase letters followed by lowercase
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


def _infer_component_type(cls: type) -> ComponentType | None:
    """Infer component type from class name or base classes."""
    class_name = cls.__name__.lower()

    # Check class name suffixes
    if class_name.endswith("node"):
        return ComponentType.NODE
    elif class_name.endswith("tool"):
        return ComponentType.TOOL
    elif class_name.endswith("adapter"):
        return ComponentType.ADAPTER
    elif class_name.endswith("policy"):
        return ComponentType.POLICY
    elif class_name.endswith("memory"):
        return ComponentType.MEMORY
    elif class_name.endswith("observer"):
        return ComponentType.OBSERVER

    # Check base class names
    for base in inspect.getmro(cls)[1:]:  # Skip the class itself
        base_name = base.__name__.lower()
        if "node" in base_name:
            return ComponentType.NODE
        elif "tool" in base_name:
            return ComponentType.TOOL
        elif "adapter" in base_name:
            return ComponentType.ADAPTER
        elif "policy" in base_name:
            return ComponentType.POLICY
        elif "memory" in base_name:
            return ComponentType.MEMORY
        elif "observer" in base_name:
            return ComponentType.OBSERVER

    return None


def get_component_metadata(cls: type) -> ComponentMetadata | None:
    """Get metadata attached to a decorated component.

    Parameters
    ----------
    cls : type
        The decorated class.

    Returns
    -------
    ComponentMetadata | None
        The metadata if the class was decorated, None otherwise.
    """
    return getattr(cls, "__component_metadata__", None)


def get_pending_components() -> list[tuple[type, ComponentMetadata, str]]:
    """Get list of components pending registration.

    Returns
    -------
    list[tuple[type, ComponentMetadata, str]]
        List of (class, metadata, namespace) tuples for pending components.
    """
    return _pending_components.copy()


def register_pending_components(registry: Any) -> int:
    """Register all pending components with the registry.

    This is called when the registry initializes to register all components
    that were decorated before the registry was created.

    Parameters
    ----------
    registry : ComponentRegistry
        The registry to register components with.

    Returns
    -------
    int
        Number of components registered.
    """
    registered = 0
    for cls, metadata, namespace in _pending_components:
        try:
            registry.register(
                name=metadata.name,
                component=cls,
                component_type=metadata.component_type,
                namespace=namespace,
                metadata=metadata,
            )
            registered += 1
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to register component {cls.__name__}: {e}")

    # Clear pending after registration
    clear_pending_components()
    return registered


def clear_pending_components() -> None:
    """Clear the list of pending components.

    Used after components have been registered.
    """
    _pending_components.clear()
