"""Component metadata definitions for the registry system."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from hexai.core.registry.types import ComponentType, NodeSubtype


@dataclass
class ComponentMetadata:
    """Metadata for registered components.

    Simple data structure without behavior.
    Instance creation is handled by the registry, not here.
    """

    name: str
    component_type: ComponentType
    component: Any
    namespace: str = "core"
    subtype: NodeSubtype | str | None = None
    description: str = ""

    @property
    def is_core(self) -> bool:
        """Check if this is a core component."""
        return self.namespace == "core"

    @property
    def qualified_name(self) -> str:
        """Get fully qualified name."""
        return f"{self.namespace}:{self.name}"


class InstanceFactory:
    """Handles component instantiation logic.

    Separated from metadata to follow single responsibility principle.
    """

    @staticmethod
    def create_instance(component: Any, **kwargs: Any) -> Any:
        """Create instance from component.

        For classes: Creates new instance with provided kwargs
        For functions: Returns the function itself (not called)
        For instances: Returns as-is

        Parameters
        ----------
        component : Any
            Component class, function, or instance
        **kwargs : Any
            Arguments for class instantiation (ignored for functions/instances)

        Returns
        -------
        Any
            Component instance or callable

        Examples
        --------
        >>> # Class - gets instantiated
        >>> class MyNode:
        ...     def __init__(self, value=42):
        ...         self.value = value
        >>> instance = InstanceFactory.create_instance(MyNode, value=100)
        >>> assert instance.value == 100

        >>> # Function - returned as-is
        >>> def my_tool(x): return x * 2
        >>> tool = InstanceFactory.create_instance(my_tool)
        >>> assert tool(5) == 10
        """
        # Functions and methods are returned as-is (not called)
        if inspect.isfunction(component) or inspect.ismethod(component):
            return component

        # If it's not a class, it's already an instance
        if not inspect.isclass(component):
            return component

        # It's a class - instantiate it with kwargs
        return component(**kwargs)
