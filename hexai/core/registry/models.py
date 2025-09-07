"""Models and type definitions for the registry system."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ComponentType(StrEnum):
    """Enumeration of component types in the registry."""

    NODE = "node"
    ADAPTER = "adapter"
    TOOL = "tool"
    POLICY = "policy"
    MEMORY = "memory"
    OBSERVER = "observer"


class NodeSubtype(StrEnum):
    """Subtypes for NODE components - important for DAG boundaries."""

    FUNCTION = "function"  # Simple function node
    LLM = "llm"  # LLM-based node (has prompt templates)
    AGENT = "agent"  # Agent node (has tools, multi-step reasoning)
    LOOP = "loop"  # Loop control node
    CONDITIONAL = "conditional"  # Conditional branching node
    PASSTHROUGH = "passthrough"  # Simple passthrough node


class Namespace(StrEnum):
    """Standard namespaces for component organization."""

    CORE = "core"  # Protected core components
    USER = "user"  # Default for user-defined components
    TEST = "test"  # For testing purposes
    PLUGIN = "plugin"  # For plugin components


@dataclass(frozen=True)
class DecoratorMetadata:
    """Immutable metadata attached by decorators.

    This is what decorators attach to classes via __hexdag_metadata__.
    It's frozen to prevent accidental mutation after decoration.
    """

    type: ComponentType | str
    name: str
    declared_namespace: str = "user"  # Namespace declared by decorator (manifest can override)
    subtype: NodeSubtype | str | None = None
    description: str = ""


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
    # Extensible metadata for future features
    metadata_extensions: dict[str, Any] = field(default_factory=dict)

    @property
    def is_core(self) -> bool:
        """Check if this is a core component."""
        return self.namespace == "core"

    @property
    def qualified_name(self) -> str:
        """Get fully qualified name."""
        return f"{self.namespace}:{self.name}"


@dataclass
class ComponentInfo:
    """Rich information about a registered component."""

    name: str
    namespace: str
    qualified_name: str
    component_type: ComponentType
    metadata: ComponentMetadata
    is_protected: bool = False


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
