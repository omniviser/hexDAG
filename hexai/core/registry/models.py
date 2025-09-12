"""Models and type definitions for the registry system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hexai.core.registry.types import (
    ClassComponent,
    FunctionComponent,
    InstanceComponent,
    MetadataExtension,
)


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
    component: ClassComponent | FunctionComponent | InstanceComponent
    namespace: str = "core"
    subtype: NodeSubtype | str | None = None
    description: str = ""
    # Extensible metadata with strict typing
    metadata_extensions: MetadataExtension = field(default_factory=MetadataExtension)

    @property
    def raw_component(self) -> Any:
        """Get the raw unwrapped component (class, function, or instance)."""
        return self.component.value

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
    def create_instance(
        component: ClassComponent | FunctionComponent | InstanceComponent,
        init_params: dict[str, Any] | None = None,
    ) -> object:
        """Create instance from component.

        For classes: Creates new instance with provided kwargs
        For functions: Returns the function itself (not called)
        For instances: Returns as-is

        Parameters
        ----------
        component : ClassComponent | FunctionComponent | InstanceComponent
            Component wrapper with type information
        init_params : dict[str, Any] | None
            Arguments for class instantiation (ignored for functions/instances)

        Returns
        -------
        object
            Component instance or callable

        Examples
        --------
        >>> # Class - gets instantiated
        >>> class MyNode:
        ...     def __init__(self, value=42):
        ...         self.value = value
        >>> component = ClassComponent(value=MyNode)
        >>> instance = InstanceFactory.create_instance(component, value=100)
        >>> assert instance.value == 100

        >>> # Function - returned as-is
        >>> def my_tool(x): return x * 2
        >>> component = FunctionComponent(value=my_tool)
        >>> tool = InstanceFactory.create_instance(component)
        >>> assert tool(5) == 10
        """
        if isinstance(component, ClassComponent) and init_params:
            return component.instantiate(**init_params)
        elif isinstance(component, (FunctionComponent, InstanceComponent)):
            return component.instantiate()
        else:
            return component.instantiate()
