"""Models and type definitions for the registry system."""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 - needed at runtime for TypeAdapter
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Discriminator, Field, TypeAdapter, field_validator

from hexai.core.exceptions import TypeMismatchError

# Type variables
T = TypeVar("T")
TComponent = TypeVar("TComponent")
TInstance = TypeVar("TInstance")

# ============================================================================
# Registry Constants
# ============================================================================

# Namespace separator for qualified names (e.g., "core:my_component")
NAMESPACE_SEPARATOR = ":"

# Component attribute names (set by decorators)
IMPLEMENTS_PORT_ATTR = "_hexdag_implements_port"
REQUIRED_PORTS_ATTR = "_hexdag_required_ports"
COMPONENT_VALUE_ATTR = "value"


# ============================================================================
# Component Wrappers - Discriminated union for type-safe component storage
# ============================================================================


class ClassComponent(BaseModel):
    """Component that is a class."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["class"] = "class"
    value: Any  # Python class object - Any required here as type[Any] isn't valid syntax

    @field_validator("value")
    @classmethod
    def validate_is_class(cls, v: Any) -> Any:
        """Ensure value is actually a class.

        Returns
        -------
            The validated class value.

        Raises
        ------
        TypeMismatchError
            If value is not a class.
        """
        if not isinstance(v, type):
            raise TypeMismatchError("component", "a class", type(v).__name__)
        return v

    def instantiate(self, **init_params: Any) -> object:
        """Create instance with initialization parameters.

        Returns
        -------
            An instance of the component class.
        """
        return self.value(**init_params)


class FunctionComponent(BaseModel):
    """Component that is a function."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["function"] = "function"
    value: Callable[..., object]  # More specific than Any

    @field_validator("value")
    @classmethod
    def validate_is_callable(cls, v: Callable) -> Callable:
        """Ensure value is callable.

        Returns
        -------
            The validated callable value.

        Raises
        ------
        TypeMismatchError
            If value is not callable.
        """
        if not callable(v):
            raise TypeMismatchError("component", "callable", type(v).__name__)
        return v

    def instantiate(self) -> Callable[..., object]:
        """Return function as-is (not called).

        Returns
        -------
            The function component.
        """
        return self.value


class InstanceComponent(BaseModel):
    """Component that is already an instance."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["instance"] = "instance"
    value: Any  # Already instantiated object - Any required as we accept any instance

    def instantiate(self) -> object:
        """Return instance as-is.

        Returns
        -------
            The component instance.
        """
        return self.value


# Discriminated union for components - cheap and strict
Component = Annotated[
    ClassComponent | FunctionComponent | InstanceComponent,
    Discriminator("type"),
]

# Type adapter for component validation and parsing
ComponentAdapter: TypeAdapter[Component] = TypeAdapter(Component)


# ============================================================================
# Enumerations
# ============================================================================


class ComponentType(StrEnum):
    """Enumeration of component types in the registry."""

    NODE = "node"
    PORT = "port"  # The interface/protocol (e.g., LLM)
    ADAPTER = "adapter"  # Implementation of a port (e.g., AnthropicAdapter)
    TOOL = "tool"
    POLICY = "policy"
    MEMORY = "memory"
    OBSERVER = "observer"
    CONTROLLER = "controller"


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


class ComponentMetadata(BaseModel):
    """Metadata for registered components.

    Simple data structure without behavior.
    Instance creation is handled by the registry, not here.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    component_type: ComponentType
    component: ClassComponent | FunctionComponent | InstanceComponent
    namespace: str = Field(default="core")
    subtype: NodeSubtype | str | None = None
    description: str = Field(default="")
    # Only non-inferable metadata
    implements_port: str | None = None  # For adapters only
    port_requirements: list[str] = Field(default_factory=list)  # For tools needing ports

    @property
    def raw_component(self) -> object:
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


class ComponentInfo(BaseModel):
    """Rich information about a registered component."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    namespace: str
    qualified_name: str
    component_type: ComponentType
    metadata: ComponentMetadata
    is_protected: bool = Field(default=False)


class InstanceFactory:
    """Handles component instantiation logic.

    Separated from metadata to follow single responsibility principle.
    """

    @staticmethod
    def create_instance(
        component: ClassComponent | FunctionComponent | InstanceComponent,
        init_params: dict[str, object] | None = None,
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
        return component.instantiate()
