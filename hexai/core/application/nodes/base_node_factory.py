"""Simplified BaseNodeFactory for creating nodes with Pydantic models and core event emission."""

from abc import ABC, abstractmethod
from typing import Any, Type, cast

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
        self, name: str, schema: dict[str, Any] | Type[BaseModel] | Type[Any] | None
    ) -> Type[BaseModel] | None:
        """Create a Pydantic model from a schema."""
        if schema is None:
            return None

        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema

        if isinstance(schema, dict):
            # Create field definitions for create_model
            field_definitions: dict[str, Any] = {}
            for field_name, field_type in schema.items():
                # Pydantic v2 requires (type, default) tuple for field definitions
                # Use ... (Ellipsis) to indicate required field
                field_definitions[field_name] = (field_type, ...)

            # create_model returns Type[BaseModel]
            # Cast is safe here as create_model returns a BaseModel subclass
            return cast(Type[BaseModel], create_model(name, **field_definitions))

        # Handle primitive types - create a simple wrapper model
        if isinstance(schema, type):
            return create_model(name, value=(schema, ...))

        # If we get here, schema is an unexpected type
        return None  # type: ignore[unreachable]

    def create_node_with_mapping(
        self,
        name: str,
        wrapped_fn: Any,
        input_schema: dict[str, Any] | None,
        output_schema: dict[str, Any] | Type[BaseModel] | None,
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
