"""Simplified decorators for component registration."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, TypeVar

from hexai.core.registry.types import ComponentType, Namespace, NodeSubtype

T = TypeVar("T")


def component(
    component_type: ComponentType | str,
    name: str | None = None,
    *,
    namespace: str | Namespace = "user",
    subtype: NodeSubtype | str | None = None,
    description: str | None = None,
) -> Callable[[type[T]], type[T]]:
    """Universal decorator for component registration.

    Parameters
    ----------
    component_type : ComponentType | str
        Component type (required). Can be enum or string like "node", "tool", etc.
    name : str | None
        Component name. If None, uses class name in snake_case.
    namespace : str | Namespace
        Component namespace. Defaults to 'user'.
        Use 'core' for system components (requires privilege),
        or custom string for plugins.
    subtype : NodeSubtype | str | None
        Subtype for nodes (e.g., 'function', 'llm', 'agent').
    description : str | None
        Component description. If None, uses class docstring.

    Examples
    --------
    >>> @component("node")  # String works
    ... class PassthroughNode:
    ...     '''Passes data through unchanged.'''
    ...     pass

    >>> @component(ComponentType.NODE)  # Enum also works
    ... class DataNode:
    ...     pass
    """

    def decorator(cls: type[T]) -> type[T]:
        from hexai.core.registry import registry

        # Infer name from class name if not provided
        component_name = name or _snake_case(cls.__name__)

        # Use class docstring as description if not provided
        component_description = description or (cls.__doc__ or "").strip()

        # Only check for core namespace privilege (let registry handle conversion)
        privileged = namespace == "core" or namespace == Namespace.CORE

        # Register component (registry will handle string->enum conversion)
        registry.register(
            name=component_name,
            component=cls,
            component_type=component_type,
            namespace=namespace,
            privileged=privileged,
            subtype=subtype,
            description=component_description,
        )

        return cls

    return decorator


# Simple wrapper decorators for common component types
# Much cleaner than the factory pattern


def node(
    name: str | None = None,
    namespace: str | Namespace = "user",
    subtype: NodeSubtype | str | None = None,
    **kwargs: Any,
) -> Callable[[type[T]], type[T]]:
    """Decorator for node components."""
    return component("node", name, namespace=namespace, subtype=subtype, **kwargs)


def tool(
    name: str | None = None,
    namespace: str | Namespace = "user",
    **kwargs: Any,
) -> Callable[[type[T]], type[T]]:
    """Decorator for tool components."""
    return component("tool", name, namespace=namespace, **kwargs)


def adapter(
    name: str | None = None,
    namespace: str | Namespace = "user",
    **kwargs: Any,
) -> Callable[[type[T]], type[T]]:
    """Decorator for adapter components."""
    return component("adapter", name, namespace=namespace, **kwargs)


def policy(
    name: str | None = None,
    namespace: str | Namespace = "user",
    **kwargs: Any,
) -> Callable[[type[T]], type[T]]:
    """Decorator for policy components."""
    return component("policy", name, namespace=namespace, **kwargs)


def memory(
    name: str | None = None,
    namespace: str | Namespace = "user",
    **kwargs: Any,
) -> Callable[[type[T]], type[T]]:
    """Decorator for memory components."""
    return component("memory", name, namespace=namespace, **kwargs)


def observer(
    name: str | None = None,
    namespace: str | Namespace = "user",
    **kwargs: Any,
) -> Callable[[type[T]], type[T]]:
    """Decorator for observer components."""
    return component("observer", name, namespace=namespace, **kwargs)


# Node subtype decorators
def function_node(
    name: str | None = None,
    namespace: str | Namespace = "user",
    **kwargs: Any,
) -> Callable[[type[T]], type[T]]:
    """Decorator for function nodes."""
    return component("node", name, namespace=namespace, subtype="function", **kwargs)


def llm_node(
    name: str | None = None,
    namespace: str | Namespace = "user",
    **kwargs: Any,
) -> Callable[[type[T]], type[T]]:
    """Decorator for LLM nodes."""
    return component("node", name, namespace=namespace, subtype="llm", **kwargs)


def agent_node(
    name: str | None = None,
    namespace: str | Namespace = "user",
    **kwargs: Any,
) -> Callable[[type[T]], type[T]]:
    """Decorator for agent nodes."""
    return component("node", name, namespace=namespace, subtype="agent", **kwargs)


def _snake_case(name: str) -> str:
    """Convert CamelCase to snake_case."""
    # Insert underscore before uppercase letters
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    # Insert underscore before uppercase letter sequences
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    # Convert to lowercase and remove leading underscore
    return s2.lower().lstrip("_")
