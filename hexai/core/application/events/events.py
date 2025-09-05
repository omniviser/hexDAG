"""Consolidated event types with reduced redundancy."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, TypeVar

from .base import EventType, PipelineEvent


class ExecutionLevel(Enum):
    """Execution hierarchy levels."""

    DAG = "dag"  # Entire DAG execution (was PIPELINE)
    WAVE = "wave"  # Wave of parallel nodes
    NODE = "node"  # Individual node


class ExecutionPhase(Enum):
    """Execution lifecycle phases."""

    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


T = TypeVar("T")


@dataclass
class ExecutionEvent(PipelineEvent):
    """Generic execution event for all levels (pipeline/wave/node).

    This consolidates:
    - NodeStartedEvent, NodeCompletedEvent, NodeFailedEvent
    - WaveStartedEvent, WaveCompletedEvent
    - PipelineStartedEvent, PipelineCompletedEvent
    """

    level: ExecutionLevel
    phase: ExecutionPhase
    name: str  # dag_name, wave_index, or node_name

    # Optional fields based on level/phase
    execution_time_ms: float | None = None  # Time in milliseconds
    result: Any = None
    error: Exception | None = None
    dependencies: list[str] = field(default_factory=list)
    nodes: list[str] = field(default_factory=list)
    wave_index: int | None = None
    total_waves: int | None = None
    total_nodes: int | None = None
    node_results: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set event type based on level and phase."""
        # Map to legacy event types for backward compatibility
        type_map = {
            (ExecutionLevel.NODE, ExecutionPhase.STARTED): EventType.NODE_STARTED,
            (ExecutionLevel.NODE, ExecutionPhase.COMPLETED): EventType.NODE_COMPLETED,
            (ExecutionLevel.NODE, ExecutionPhase.FAILED): EventType.NODE_FAILED,
            (ExecutionLevel.WAVE, ExecutionPhase.STARTED): EventType.WAVE_STARTED,
            (ExecutionLevel.WAVE, ExecutionPhase.COMPLETED): EventType.WAVE_COMPLETED,
            (ExecutionLevel.DAG, ExecutionPhase.STARTED): EventType.PIPELINE_STARTED,
            (ExecutionLevel.DAG, ExecutionPhase.COMPLETED): EventType.PIPELINE_COMPLETED,
        }

        self.event_type = type_map.get(
            (self.level, self.phase),
            EventType.PIPELINE_STARTED,  # fallback
        )
        self.timestamp = datetime.now()
        self.session_id = getattr(self, "session_id", "default")

    def _extra_fields(self) -> dict[str, Any]:
        """Get relevant fields for logging."""
        fields: dict[str, Any] = {
            "level": self.level.value,
            "phase": self.phase.value,
            "name": self.name,
        }

        if self.execution_time_ms is not None:
            fields["execution_time_ms"] = self.execution_time_ms
        if self.error is not None:
            fields["error_type"] = type(self.error).__name__
            fields["error_message"] = str(self.error)
        if self.wave_index is not None:
            fields["wave_index"] = self.wave_index
        if self.nodes:
            fields["node_count"] = len(self.nodes)

        return fields


@dataclass
class LLMEvent(PipelineEvent):
    """LLM and tool interaction event.

    This consolidates:
    - LLMPromptGeneratedEvent
    - LLMResponseReceivedEvent
    - ToolCalledEvent
    - ToolCompletedEvent
    """

    event_class: Literal["llm", "tool"]
    action: Literal["prompt", "response", "called", "completed"]
    node_name: str

    # Input/output data
    input_data: Any = None  # prompt/params
    output_data: Any = None  # response/result
    tool_name: str = ""  # Tool/adapter name or "llm"

    # Optional fields
    execution_time_ms: float | None = None  # Time in milliseconds
    messages: list[dict[str, str]] = field(default_factory=list)
    template: str = ""
    template_vars: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set event type based on event class and action."""
        # Map to legacy event types
        type_map = {
            ("llm", "prompt"): EventType.LLM_PROMPT_GENERATED,
            ("llm", "response"): EventType.LLM_RESPONSE_RECEIVED,
            ("tool", "called"): EventType.TOOL_CALLED,
            ("tool", "completed"): EventType.TOOL_COMPLETED,
        }

        self.event_type = type_map.get(
            (self.event_class, self.action),
            EventType.TOOL_CALLED,  # fallback
        )
        self.timestamp = datetime.now()
        self.session_id = getattr(self, "session_id", "default")

    def _extra_fields(self) -> dict[str, Any]:
        """Get relevant fields for logging."""
        fields: dict[str, Any] = {
            "event_class": self.event_class,
            "action": self.action,
            "node_name": self.node_name,
        }

        if self.tool_name:
            fields["tool_name"] = self.tool_name
        if self.execution_time_ms is not None:
            fields["execution_time_ms"] = self.execution_time_ms
        if self.messages:
            fields["message_count"] = len(self.messages)

        return fields


@dataclass
class HookEvent(PipelineEvent):
    """Hook execution event for extension points.

    Used for:
    - Node execution: pre_node, mid_node, post_node
    - Adapter/port calls: pre_adapter, post_adapter
    - LLM operations: pre_llm, post_llm, llm_retry
    - Wave execution: pre_wave, post_wave
    - DAG lifecycle: pre_dag, post_dag
    """

    hook_type: Literal[
        "pre_node",
        "post_node",
        "mid_node",  # Node execution hooks
        "pre_adapter",
        "post_adapter",  # Adapter/port hooks
        "pre_llm",
        "post_llm",
        "llm_retry",  # LLM-specific hooks
        "pre_wave",
        "post_wave",  # Wave hooks
        "pre_dag",
        "post_dag",  # DAG hooks
    ]
    hook_name: str
    target_name: str  # node/wave/pipeline name
    result: Any = None
    execution_time_ms: float | None = None  # Time in milliseconds
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize hook event."""
        # Hooks don't map to legacy event types, create new one if needed
        self.event_type = EventType.PIPELINE_STARTED  # placeholder
        self.timestamp = datetime.now()
        self.session_id = getattr(self, "session_id", "hooks")

    def _extra_fields(self) -> dict[str, Any]:
        """Get relevant fields for logging."""
        fields: dict[str, Any] = {
            "hook_type": self.hook_type,
            "hook_name": self.hook_name,
            "target_name": self.target_name,
        }

        if self.execution_time_ms is not None:
            fields["execution_time_ms"] = self.execution_time_ms

        return fields


@dataclass
class MetaEvent(PipelineEvent):
    """Generic meta/lifecycle event.

    This consolidates:
    - ValidationWarningEvent
    - PipelineBuildEvent
    - Future diagnostic events
    """

    category: Literal["validation", "build", "diagnostic"]
    pipeline_name: str
    message: str = ""
    details: Any = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set event type based on category."""
        type_map = {
            "validation": EventType.VALIDATION_WARNING,
            "build": EventType.PIPELINE_BUILD_STARTED,
        }

        self.event_type = type_map.get(self.category, EventType.PIPELINE_BUILD_STARTED)  # fallback
        self.timestamp = datetime.now()
        self.session_id = getattr(self, "session_id", self.category)

    def _extra_fields(self) -> dict[str, Any]:
        """Get relevant fields for logging."""
        fields: dict[str, Any] = {
            "category": self.category,
            "pipeline_name": self.pipeline_name,
            "message": self.message,
        }

        if self.warnings:
            fields["warning_count"] = len(self.warnings)

        return fields


# Factory functions for backward compatibility
def create_node_started(
    node_name: str, wave_index: int, dependencies: list[str] | None = None, **kwargs: Any
) -> ExecutionEvent:
    """Create a node started event."""
    return ExecutionEvent(
        level=ExecutionLevel.NODE,
        phase=ExecutionPhase.STARTED,
        name=node_name,
        wave_index=wave_index,
        dependencies=dependencies or [],
        **kwargs,
    )


def create_node_completed(
    node_name: str, wave_index: int, result: Any, execution_time: float, **kwargs: Any
) -> ExecutionEvent:
    """Create a node completed event."""
    return ExecutionEvent(
        level=ExecutionLevel.NODE,
        phase=ExecutionPhase.COMPLETED,
        name=node_name,
        wave_index=wave_index,
        result=result,
        execution_time_ms=execution_time * 1000,  # Convert to milliseconds
        **kwargs,
    )


def create_node_failed(
    node_name: str, wave_index: int, error: Exception, **kwargs: Any
) -> ExecutionEvent:
    """Create a node failed event."""
    return ExecutionEvent(
        level=ExecutionLevel.NODE,
        phase=ExecutionPhase.FAILED,
        name=node_name,
        wave_index=wave_index,
        error=error,
        **kwargs,
    )


def create_dag_started(
    dag_name: str, total_waves: int, total_nodes: int, **kwargs: Any
) -> ExecutionEvent:
    """Create a DAG started event."""
    return ExecutionEvent(
        level=ExecutionLevel.DAG,
        phase=ExecutionPhase.STARTED,
        name=dag_name,
        total_waves=total_waves,
        total_nodes=total_nodes,
        **kwargs,
    )


def create_llm_prompt(
    node_name: str,
    messages: list[dict[str, str]],
    template: str = "",
    template_vars: dict[str, Any] | None = None,
    **kwargs: Any,
) -> LLMEvent:
    """Create an LLM prompt generated event."""
    return LLMEvent(
        event_class="llm",
        action="prompt",
        node_name=node_name,
        tool_name="llm",
        input_data=messages,
        messages=messages,
        template=template,
        template_vars=template_vars or {},
        **kwargs,
    )


def create_tool_called(
    node_name: str, tool_name: str, params: dict[str, Any], **kwargs: Any
) -> LLMEvent:
    """Create a tool called event."""
    return LLMEvent(
        event_class="tool",
        action="called",
        node_name=node_name,
        tool_name=tool_name,
        input_data=params,
        **kwargs,
    )


def create_tool_completed(
    node_name: str, tool_name: str, result: Any, execution_time: float | None = None, **kwargs: Any
) -> LLMEvent:
    """Create a tool completed event."""
    return LLMEvent(
        event_class="tool",
        action="completed",
        node_name=node_name,
        tool_name=tool_name,
        output_data=result,
        execution_time_ms=execution_time * 1000 if execution_time else None,
        **kwargs,
    )


def create_llm_response(
    node_name: str, response: str, execution_time: float | None = None, **kwargs: Any
) -> LLMEvent:
    """Create an LLM response received event."""
    return LLMEvent(
        event_class="llm",
        action="response",
        node_name=node_name,
        tool_name="llm",
        output_data=response,
        execution_time_ms=execution_time * 1000 if execution_time else None,
        **kwargs,
    )


def create_hook_event(
    hook_type: Literal[
        "pre_node",
        "post_node",
        "mid_node",  # Node execution hooks
        "pre_adapter",
        "post_adapter",  # Adapter/port hooks
        "pre_llm",
        "post_llm",
        "llm_retry",  # LLM-specific hooks
        "pre_wave",
        "post_wave",  # Wave hooks
        "pre_dag",
        "post_dag",  # DAG hooks
    ],
    hook_name: str,
    target_name: str,
    result: Any = None,
    start_time: float | None = None,
    **kwargs: Any,
) -> HookEvent:
    """Create a hook execution event.

    Hook types:
    - mid_node: Inside node execution (e.g., between retries)
    - pre/post_adapter: Before/after adapter port calls
    - llm_retry: When LLM call is retried (for backoff/logging)
    """
    execution_time_ms = None
    if start_time is not None:
        execution_time_ms = (time.time() - start_time) * 1000

    return HookEvent(
        hook_type=hook_type,
        hook_name=hook_name,
        target_name=target_name,
        result=result,
        execution_time_ms=execution_time_ms,
        **kwargs,
    )
