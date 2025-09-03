"""Component metadata definitions for the registry system."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hexai.core.registry.types import ComponentType


class ComponentMetadata(BaseModel):
    """Standard metadata schema for registered components.

    Provides common metadata fields for consistent component
    discovery and management. Uses Pydantic for full validation
    and type safety.

    Attributes
    ----------
    name : str
        Unique identifier for the component within the registry.
    component_type : ComponentTypeEnum
        Type or category of the component (node, adapter, tool, etc.).
    description : str | None
        Human-readable description of the component's purpose.
    tags : frozenset[str]
        Set of tags for categorization and filtering.
    author : str
        Author or maintainer of the component (defaults to 'hexdag').
    dependencies : frozenset[str]
        Set of component names this component depends on.
        Used for dependency resolution and initialization order.
    config_schema : type[BaseModel] | None
        Pydantic model for component configuration validation.
    replaceable : bool
        Whether this component can be replaced after registration.
        Defaults to False for production stability.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    component_type: ComponentType | str
    description: str | None = None
    tags: frozenset[str] = Field(default_factory=frozenset)
    author: str = Field(default="hexdag")
    dependencies: frozenset[str] = Field(default_factory=frozenset)
    config_schema: type[BaseModel] | None = None
    replaceable: bool = Field(default=False, description="Whether this component can be replaced")
