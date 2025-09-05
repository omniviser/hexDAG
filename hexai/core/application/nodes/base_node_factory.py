"""Simplified BaseNodeFactory for creating nodes with Pydantic models."""

from abc import ABC, abstractmethod
from typing import Any, Type

from pydantic import BaseModel, create_model

from ...domain.dag import NodeSpec


class BaseNodeFactory(ABC):
    """Minimal base class for node factories with Pydantic models."""

    # Note: Event emission has been removed as it's now handled by the orchestrator
    # The new event system uses ObserverManager at the orchestrator level

    def create_pydantic_model(
        self, name: str, schema: dict[str, Any] | Type[BaseModel] | Type[Any] | None
    ) -> Type[BaseModel] | None:
        """Create a Pydantic model from a schema."""
        if schema is None:
            return None

        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema

        if isinstance(schema, dict):
            # Create field definitions for create_model
            field_definitions = {}
            for field_name, field_type in schema.items():
                field_definitions[field_name] = field_type

            return create_model(name, **field_definitions)

        # Handle primitive types - create a simple wrapper model
        if isinstance(schema, type):
            return create_model(name, value=(schema, ...))

        raise ValueError("Schema must be a dict, type, or Pydantic model")

    def create_node_with_mapping(
        self,
        name: str,
        wrapped_fn: Any,
        input_schema: dict[str, Any] | None,
        output_schema: dict[str, Any] | Type[BaseModel] | None,
        deps: list[str] | None = None,
        input_mapping: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Universal NodeSpec creation with consistent input mapping handling."""
        # Create Pydantic models
        input_model = self.create_pydantic_model(f"{name}Input", input_schema)
        output_model = self.create_pydantic_model(f"{name}Output", output_schema)

        # Determine output type
        out_type = output_model or str

        # Add input_mapping to params consistently
        params = kwargs.copy()
        if input_mapping is not None:
            params["input_mapping"] = input_mapping

        return NodeSpec(
            name=name,
            fn=wrapped_fn,
            in_type=input_model,
            out_type=out_type,
            deps=set(deps or []),
            params=params,
        )

    @abstractmethod
    def __call__(self, name: str, *args: Any, **kwargs: Any) -> NodeSpec:
        """Create a NodeSpec.

        Must be implemented by subclasses.
        """
        pass
