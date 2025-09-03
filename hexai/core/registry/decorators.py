"""Unified decorators for component registration.

These decorators work for both core and plugin components, providing
a consistent API across the entire hexDAG ecosystem.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional, TypeVar

from hexai.core.registry.types import ComponentType

T = TypeVar("T")


def component(
    name: Optional[str] = None,
    *,
    namespace: str = "core",
    component_type: Optional[ComponentType | str] = None,
    description: Optional[str] = None,
    tags: Optional[set[str] | frozenset[str]] = None,
    author: str = "hexdag",
    dependencies: Optional[set[str] | frozenset[str]] = None,
    replaceable: bool = False,
    version: str = "1.0.0",
) -> Callable[[type[T]], type[T]]:
    """Universal decorator for component registration.

    This decorator works for both core and plugin components. Core components
    use namespace='core' by default, while plugins must specify their namespace.

    Parameters
    ----------
    name : str | None
        Component name. If None, uses class name in snake_case.
    namespace : str
        Component namespace. Defaults to 'core' for core components.
        Plugins must specify their namespace.
    component_type : ComponentType | str | None
        Component type. If None, inferred from class name.
    description : str | None
        Component description. If None, uses class docstring.
    tags : set[str] | frozenset[str] | None
        Component tags for categorization.
    author : str
        Component author (defaults to 'hexdag').
    dependencies : set[str] | frozenset[str] | None
        Other components this component depends on.
    replaceable : bool
        If True, component can be replaced after registration.
    version : str
        Component version.

    Returns
    -------
    Callable[[type[T]], type[T]]
        Decorated class.

    Examples
    --------
    >>> # Core component (namespace='core' by default)
    >>> @component()
    >>> class PassthroughNode:
    ...     '''Passes data through unchanged.'''
    ...     pass
    >>>
    >>> # Plugin component (must specify namespace)
    >>> @component(namespace='my_plugin', tags={'analysis'})
    >>> class AnalyzerNode:
    ...     '''Analyzes data.'''
    ...     pass
    """

    def decorator(cls: type[T]) -> type[T]:
        # Import here to avoid circular dependency
        from hexai.core.registry.registry import ComponentRegistry

        # Infer name from class name if not provided
        component_name = name or _snake_case(cls.__name__)

        # Infer component type from class name or inheritance
        inferred_type = component_type or _infer_component_type(cls)
        if not inferred_type:
            raise ValueError(
                f"Cannot infer component type for {cls.__name__}. "
                f"Please specify component_type or follow naming conventions "
                f"(e.g., suffix with Node, Tool, Adapter, Policy, Memory, Observer)"
            )

        # Use class docstring as description if not provided
        component_description = description or (cls.__doc__ or "").strip()

        # Prepare metadata
        metadata = {
            "name": component_name,
            "namespace": namespace,
            "component_type": inferred_type,
            "description": component_description,
            "tags": tags or set(),
            "author": author,
            "dependencies": dependencies or set(),
            "replaceable": replaceable,
            "version": version,
        }

        # Add to pending registrations
        ComponentRegistry.add_pending(cls, metadata)

        # Mark class for debugging
        cls._hexdag_component = True  # type: ignore
        cls._hexdag_namespace = namespace  # type: ignore
        cls._hexdag_name = component_name  # type: ignore

        return cls

    return decorator


def node(
    name: Optional[str] = None, namespace: str = "core", **kwargs: Any
) -> Callable[[type[T]], type[T]]:
    """Shorthand decorator for node components.

    Examples
    --------
    >>> @node()  # Core node
    ... class DataProcessor:
    ...     pass
    >>>
    >>> @node(namespace='my_plugin')  # Plugin node
    ... class CustomProcessor:
    ...     pass
    """
    return component(name, namespace=namespace, component_type=ComponentType.NODE, **kwargs)


def tool(
    name: Optional[str] = None, namespace: str = "core", **kwargs: Any
) -> Callable[[type[T]], type[T]]:
    """Shorthand decorator for tool components.

    Examples
    --------
    >>> @tool()
    ... class WebScraper:
    ...     pass
    """
    return component(name, namespace=namespace, component_type=ComponentType.TOOL, **kwargs)


def adapter(
    name: Optional[str] = None, namespace: str = "core", **kwargs: Any
) -> Callable[[type[T]], type[T]]:
    """Shorthand decorator for adapter components.

    Examples
    --------
    >>> @adapter()
    ... class PostgresAdapter:
    ...     pass
    """
    return component(name, namespace=namespace, component_type=ComponentType.ADAPTER, **kwargs)


def policy(
    name: Optional[str] = None, namespace: str = "core", **kwargs: Any
) -> Callable[[type[T]], type[T]]:
    """Shorthand decorator for policy components.

    Examples
    --------
    >>> @policy()
    ... class RetryPolicy:
    ...     pass
    """
    return component(name, namespace=namespace, component_type=ComponentType.POLICY, **kwargs)


def memory(
    name: Optional[str] = None, namespace: str = "core", **kwargs: Any
) -> Callable[[type[T]], type[T]]:
    """Shorthand decorator for memory components.

    Examples
    --------
    >>> @memory()
    ... class ConversationMemory:
    ...     pass
    """
    return component(name, namespace=namespace, component_type=ComponentType.MEMORY, **kwargs)


def observer(
    name: Optional[str] = None, namespace: str = "core", **kwargs: Any
) -> Callable[[type[T]], type[T]]:
    """Shorthand decorator for observer components.

    Examples
    --------
    >>> @observer()
    ... class MetricsObserver:
    ...     pass
    """
    return component(name, namespace=namespace, component_type=ComponentType.OBSERVER, **kwargs)


def get_component_metadata(obj: Any) -> dict[str, Any] | None:
    """Get metadata from a decorated component.

    Parameters
    ----------
    obj : Any
        Object to check for metadata.

    Returns
    -------
    dict[str, Any] | None
        Component metadata if decorated, None otherwise.
    """
    if hasattr(obj, "_hexdag_component"):
        return {
            "name": getattr(obj, "_hexdag_name", None),
            "namespace": getattr(obj, "_hexdag_namespace", None),
        }
    return None


def _snake_case(name: str) -> str:
    """Convert CamelCase to snake_case.

    Parameters
    ----------
    name : str
        Name in CamelCase.

    Returns
    -------
    str
        Name in snake_case.
    """
    import re

    # Insert underscore before uppercase letters
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    # Insert underscore before uppercase letter sequences
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    # Convert to lowercase and remove leading underscore
    return s2.lower().lstrip("_")


def _infer_component_type(cls: type) -> ComponentType | None:
    """Infer component type from class name or base classes.

    Parameters
    ----------
    cls : type
        Class to infer type from.

    Returns
    -------
    ComponentType | None
        Inferred type or None if cannot infer.
    """
    name_lower = cls.__name__.lower()

    # Check class name suffixes
    if name_lower.endswith("node") or "node" in name_lower:
        return ComponentType.NODE
    elif name_lower.endswith("adapter") or "adapter" in name_lower:
        return ComponentType.ADAPTER
    elif name_lower.endswith("tool") or "tool" in name_lower:
        return ComponentType.TOOL
    elif name_lower.endswith("policy") or "policy" in name_lower:
        return ComponentType.POLICY
    elif name_lower.endswith("memory") or "memory" in name_lower:
        return ComponentType.MEMORY
    elif name_lower.endswith("observer") or "observer" in name_lower:
        return ComponentType.OBSERVER

    # Check base class names
    for base in cls.__bases__:
        base_type = _infer_component_type(base)
        if base_type:
            return base_type

    return None
