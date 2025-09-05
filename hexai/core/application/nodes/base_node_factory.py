"""Simplified BaseNodeFactory for creating nodes with Pydantic models and core event emission."""

from abc import ABC, abstractmethod
from typing import Any, Type

from pydantic import BaseModel, create_model

from ...domain.dag import NodeSpec
from ..events import ExecutionEvent, ExecutionLevel, ExecutionPhase, LLMEvent


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
            event = ExecutionEvent(
                level=ExecutionLevel.NODE,
                phase=ExecutionPhase.STARTED,
                name=node_name,
                wave_index=wave_index,
                dependencies=dependencies,
                metadata=metadata or {},
            )
            await event_manager.emit(event)

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
            event = ExecutionEvent(
                level=ExecutionLevel.NODE,
                phase=ExecutionPhase.COMPLETED,
                name=node_name,
                wave_index=wave_index,
                result=result,
                execution_time_ms=execution_time * 1000,  # Convert to milliseconds
                metadata=metadata or {},
            )
            await event_manager.emit(event)

    async def emit_node_failed(
        self, node_name: str, error: Exception, wave_index: int, event_manager: Any = None
    ) -> None:
        """Emit node failed event."""
        if event_manager:
            await event_manager.emit(
                ExecutionEvent(
                    level=ExecutionLevel.NODE,
                    phase=ExecutionPhase.FAILED,
                    name=node_name,
                    wave_index=wave_index,
                    error=error,
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
            event = LLMEvent(
                event_class="tool",
                action="called",
                node_name=node_name,
                tool_name=tool_name,
                input_data=tool_params,
                metadata=metadata or {},
            )
            await event_manager.emit(event)

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
            event = LLMEvent(
                event_class="tool",
                action="completed",
                node_name=node_name,
                tool_name=tool_name,
                output_data=result,
                execution_time_ms=execution_time * 1000 if execution_time else None,
                metadata=metadata or {},
            )
            await event_manager.emit(event)

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
