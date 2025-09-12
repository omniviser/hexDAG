"""Simplified BaseNodeFactory for creating nodes with Pydantic models and core event emission."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, create_model

from ...domain.dag import NodeSpec
from ..events.events import (
    NodeCompletedEvent,
    NodeFailedEvent,
    NodeStartedEvent,
    ToolCalledEvent,
    ToolCompletedEvent,
)


class BaseNodeFactory(ABC):
    """Minimal base class for node factories with Pydantic models and core event emission."""

    # Core node event emission methods (used by all nodes)
    async def emit_node_started(
        self,
        node_name: str,
        wave_index: int,
        dependencies: list[str] | None = None,
        event_manager: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit node started event."""
        if event_manager and dependencies is not None:
            await event_manager.emit(
                NodeStartedEvent(
                    node_name=node_name,
                    wave_index=wave_index,
                    dependencies=dependencies,
                    metadata=metadata or {},
                )
            )

    async def emit_node_completed(
        self,
        node_name: str,
        result: Any,
        execution_time: float,
        wave_index: int,
        event_manager: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit node completed event."""
        if event_manager:
            await event_manager.emit(
                NodeCompletedEvent(
                    node_name=node_name,
                    result=result,
                    execution_time=execution_time,
                    wave_index=wave_index,
                    metadata=metadata or {},
                )
            )

    async def emit_node_failed(
        self, node_name: str, error: Exception, wave_index: int, event_manager: Any = None
    ) -> None:
        """Emit node failed event."""
        if event_manager:
            await event_manager.emit(
                NodeFailedEvent(
                    node_name=node_name,
                    error=error,
                    wave_index=wave_index,
                )
            )

    async def emit_tool_called(
        self,
        node_name: str,
        tool_name: str,
        tool_params: dict[str, Any],
        event_manager: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit tool called event."""
        if event_manager:
            await event_manager.emit(
                ToolCalledEvent(
                    node_name=node_name,
                    tool_name=tool_name,
                    tool_params=tool_params,
                    metadata=metadata or {},
                )
            )

    async def emit_tool_completed(
        self,
        node_name: str,
        tool_name: str,
        result: Any,
        execution_time: float,
        event_manager: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit tool completed event."""
        if event_manager:
            await event_manager.emit(
                ToolCompletedEvent(
                    node_name=node_name,
                    tool_name=tool_name,
                    result=result,
                    execution_time=execution_time,
                    metadata=metadata or {},
                )
            )

    def create_pydantic_model(
        self, name: str, schema: dict[str, Any] | type[BaseModel] | type[Any] | None
    ) -> type[BaseModel] | None:
        """Create a Pydantic model from a schema."""
        if schema is None:
            return None

        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema

        if isinstance(schema, dict):
            # Create field definitions for create_model
            # Convert dict values to proper Pydantic field format
            field_definitions: dict[str, Any] = {}
            for field_name, field_type in schema.items():
                # Handle various type specifications
                if isinstance(field_type, str):
                    # String type names - convert to actual types
                    type_map = {
                        "str": str,
                        "int": int,
                        "float": float,
                        "bool": bool,
                        "list": list,
                        "dict": dict,
                        "Any": Any,
                    }
                    actual_type = type_map.get(field_type, Any)
                    field_definitions[field_name] = (actual_type, ...)
                elif isinstance(field_type, type):
                    # Already a type
                    field_definitions[field_name] = (field_type, ...)
                elif isinstance(field_type, tuple):
                    # Already in the correct format (type, default)
                    field_definitions[field_name] = field_type
                else:
                    # Unknown type specification - use Any
                    field_definitions[field_name] = (Any, ...)

            # create_model returns Type[BaseModel]
            return create_model(name, **field_definitions)

        # Handle primitive types - create a simple wrapper model
        # At this point, schema should be a type
        try:
            return create_model(name, value=(schema, ...))
        except Exception:
            # If we get here, schema is an unexpected type
            raise ValueError("Schema must be a dict, type, or Pydantic model") from None

    def create_node_with_mapping(
        self,
        name: str,
        wrapped_fn: Any,
        input_schema: dict[str, Any] | None,
        output_schema: dict[str, Any] | type[BaseModel] | None,
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Universal NodeSpec creation."""
        # Create Pydantic models
        input_model = self.create_pydantic_model(f"{name}Input", input_schema)
        output_model = self.create_pydantic_model(f"{name}Output", output_schema)

        return NodeSpec(
            name=name,
            fn=wrapped_fn,
            in_model=input_model,
            out_model=output_model,
            deps=set(deps or []),
            params=kwargs,
        )

    @abstractmethod
    def __call__(self, name: str, *args: Any, **kwargs: Any) -> NodeSpec:
        """Create a NodeSpec.

        Must be implemented by subclasses.
        """
        pass
