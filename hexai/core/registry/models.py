"""Models and type definitions for the registry system."""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 - needed at runtime for TypeAdapter
from enum import StrEnum
from typing import Annotated, Any, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Discriminator, Field, TypeAdapter, field_validator

# Type variables
T = TypeVar("T")
TComponent = TypeVar("TComponent")
TInstance = TypeVar("TInstance")


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
        """Ensure value is actually a class."""
        if not isinstance(v, type):
            raise ValueError(f"Expected a class, got {type(v).__name__}")
        return v

    def instantiate(self, **init_params: Any) -> object:
        """Create instance with initialization parameters."""
        return self.value(**init_params)


class FunctionComponent(BaseModel):
    """Component that is a function."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["function"] = "function"
    value: Callable[..., object]  # More specific than Any

    @field_validator("value")
    @classmethod
    def validate_is_callable(cls, v: Callable) -> Callable:
        """Ensure value is callable."""
        if not callable(v):
            raise ValueError(f"Expected callable, got {type(v).__name__}")
        return v

    def instantiate(self) -> Callable[..., object]:
        """Return function as-is (not called)."""
        return self.value


class InstanceComponent(BaseModel):
    """Component that is already an instance."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["instance"] = "instance"
    value: Any  # Already instantiated object - Any required as we accept any instance

    def instantiate(self) -> object:
        """Return instance as-is."""
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


class PortMetadata(BaseModel):
    """Metadata specific to PORT components (interfaces/protocols)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    protocol_class: type  # The Protocol being defined
    required_methods: list[str] = Field(default_factory=list)
    optional_methods: list[str] = Field(default_factory=list)


class AdapterMetadata(BaseModel):
    """Metadata specific to ADAPTER components (port implementations)."""

    implements_port: str  # Name of port it implements
    capabilities: list[str] = Field(default_factory=list)
    requires_config: list[str] = Field(default_factory=list)
    singleton: bool = Field(default=True)


class MetadataExtension(BaseModel):
    """Extension metadata with strict typing."""

    string_values: dict[str, str] = Field(default_factory=dict)
    int_values: dict[str, int] = Field(default_factory=dict)
    bool_values: dict[str, bool] = Field(default_factory=dict)
    float_values: dict[str, float] = Field(default_factory=dict)

    def get(
        self, key: str, default: str | int | bool | float | None = None
    ) -> str | int | bool | float | None:
        """Get value by key from any type dict."""
        stores: list[dict] = [
            self.string_values,
            self.int_values,
            self.bool_values,
            self.float_values,
        ]
        for store in stores:
            if key in store:
                return store[key]  # type: ignore[no-any-return]
        return default

    def set(self, key: str, value: str | int | bool | float) -> None:
        """Set value in appropriate type dict."""
        match value:
            case str():
                self.string_values[key] = value
            case int():
                self.int_values[key] = value
            case bool():
                self.bool_values[key] = value
            case float():
                self.float_values[key] = value
            case _:
                raise TypeError(f"Unsupported metadata type: {type(value)}")


class DecoratorMetadata(BaseModel):
    """Immutable metadata attached by decorators.

    This is what decorators attach to classes via __hexdag_metadata__.
    It's frozen to prevent accidental mutation after decoration.
    """

    model_config = ConfigDict(frozen=True)

    type: ComponentType | str
    name: str
    declared_namespace: str = Field(default="user")
    subtype: NodeSubtype | str | None = None
    description: str = Field(default="")
    adapter_metadata: AdapterMetadata | None = None  # For adapter components
    port_metadata: PortMetadata | None = None  # For port components


# ============================================================================
# Protocols for type safety
# ============================================================================


@runtime_checkable
class HasMetadata(Protocol):
    """Protocol for components with decorator metadata."""

    __hexdag_metadata__: DecoratorMetadata


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
    port_metadata: PortMetadata | None = None
    adapter_metadata: AdapterMetadata | None = None
    port_requirements: list = Field(default_factory=list)  # List of PortRequirement
    # Extensible metadata with strict typing
    metadata_extensions: MetadataExtension = Field(default_factory=MetadataExtension)

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


# ============================================================================
# Protocols and Extensions
# ============================================================================


class ComponentProtocol(Protocol):
    """Protocol for components that can be registered."""

    __hexdag_metadata__: object  # DecoratorMetadata
