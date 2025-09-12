"""Type definitions for the registry system using modern Python and Pydantic."""

from __future__ import annotations

from typing import Annotated, Any, Callable, Literal, Protocol, TypeVar, Union

from pydantic import BaseModel, Discriminator, Field, TypeAdapter

# Type variable for component classes
TComponent = TypeVar("TComponent")

# Type for component instances
TInstance = TypeVar("TInstance")


class ClassComponent(BaseModel):
    """Component that is a class."""

    type: Literal["class"] = "class"
    value: Any  # Python class object

    def instantiate(self, **init_params: Any) -> object:
        """Create instance with initialization parameters."""
        return self.value(**init_params)


class FunctionComponent(BaseModel):
    """Component that is a function."""

    type: Literal["function"] = "function"
    value: Callable[..., Any]

    def instantiate(self) -> Callable[..., Any]:
        """Return function as-is (not called)."""
        return self.value


class InstanceComponent(BaseModel):
    """Component that is already an instance."""

    type: Literal["instance"] = "instance"
    value: Any  # Already instantiated object

    def instantiate(self) -> Any:
        """Return instance as-is."""
        return self.value


# Discriminated union for components - cheap and strict
Component = Annotated[
    Union[ClassComponent, FunctionComponent, InstanceComponent],
    Discriminator("type"),
]

# Type adapter for component validation and parsing
ComponentAdapter: TypeAdapter[Component] = TypeAdapter(Component)


class ComponentProtocol(Protocol):
    """Protocol for components that can be registered."""

    __hexdag_metadata__: object  # DecoratorMetadata


class MetadataExtension(BaseModel):
    """Extension metadata with strict typing."""

    string_values: dict[str, str] = Field(default_factory=dict)
    int_values: dict[str, int] = Field(default_factory=dict)
    bool_values: dict[str, bool] = Field(default_factory=dict)
    float_values: dict[str, float] = Field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value by key from any type dict."""
        stores: list[dict] = [
            self.string_values,
            self.int_values,
            self.bool_values,
            self.float_values,
        ]
        for store in stores:
            if key in store:
                return store[key]
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
