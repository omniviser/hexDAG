"""Pipeline event implementations without Context dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .base import EventType, PipelineEvent

if TYPE_CHECKING:
    from collections.abc import Callable


def _init_event(event_type: EventType, session_id: str = "default") -> Callable[[Any], None]:
    """Create __post_init__ method for events."""

    def post_init_method(self: Any) -> None:
        self.event_type = event_type
        self.session_id = session_id
        self.timestamp = datetime.now()

    return post_init_method


@dataclass
class NodeStartedEvent(PipelineEvent):
    """Event emitted when a node starts execution."""

    node_name: str
    wave_index: int
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize event after dataclass creation."""
        _init_event(EventType.NODE_STARTED)(self)

    def _extra_fields(self) -> dict[str, Any]:
        return {
            "node_name": self.node_name,
            "wave_index": self.wave_index,
            "dependencies": self.dependencies,
        }


@dataclass
class NodeCompletedEvent(PipelineEvent):
    """Event emitted when a node completes successfully."""

    node_name: str
    wave_index: int
    result: Any
    execution_time: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize event after dataclass creation."""
        _init_event(EventType.NODE_COMPLETED)(self)

    def _extra_fields(self) -> dict[str, Any]:
        return {
            "node_name": self.node_name,
            "wave_index": self.wave_index,
            "execution_time": self.execution_time,
            "result_type": type(self.result).__name__,
        }


@dataclass
class NodeFailedEvent(PipelineEvent):
    """Event emitted when a node fails."""

    node_name: str
    wave_index: int
    error: Exception
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize event after dataclass creation."""
        _init_event(EventType.NODE_FAILED)(self)

    def _extra_fields(self) -> dict[str, Any]:
        return {
            "node_name": self.node_name,
            "wave_index": self.wave_index,
            "error_type": type(self.error).__name__,
            "error_message": str(self.error),
        }


@dataclass
class WaveStartedEvent(PipelineEvent):
    """Event emitted when a wave starts execution."""

    wave_index: int
    nodes: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize event after dataclass creation."""
        _init_event(EventType.WAVE_STARTED)(self)

    def _extra_fields(self) -> dict[str, Any]:
        return {"wave_index": self.wave_index, "nodes": self.nodes, "node_count": len(self.nodes)}


@dataclass
class WaveCompletedEvent(PipelineEvent):
    """Event emitted when a wave completes."""

    wave_index: int
    nodes: list[str]
    execution_time: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize event after dataclass creation."""
        _init_event(EventType.WAVE_COMPLETED)(self)

    def _extra_fields(self) -> dict[str, Any]:
        return {
            "wave_index": self.wave_index,
            "nodes": self.nodes,
            "node_count": len(self.nodes),
            "execution_time": self.execution_time,
        }


@dataclass
class PipelineStartedEvent(PipelineEvent):
    """Event emitted when pipeline execution starts."""

    pipeline_name: str
    total_waves: int
    total_nodes: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize event after dataclass creation."""
        _init_event(EventType.PIPELINE_STARTED)(self)

    def _extra_fields(self) -> dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "total_waves": self.total_waves,
            "total_nodes": self.total_nodes,
        }


@dataclass
class PipelineCompletedEvent(PipelineEvent):
    """Event emitted when pipeline execution completes."""

    pipeline_name: str
    total_execution_time: float
    node_results: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize event after dataclass creation."""
        _init_event(EventType.PIPELINE_COMPLETED)(self)

    def _extra_fields(self) -> dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "total_execution_time": self.total_execution_time,
            "nodes_executed": len(self.node_results),
        }


@dataclass
class ValidationWarningEvent(PipelineEvent):
    """Event emitted when validation warnings occur."""

    pipeline_name: str
    warnings: list[str]
    session_id: str = "validation"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize event after dataclass creation."""
        _init_event(EventType.VALIDATION_WARNING, "validation")(self)

    def _extra_fields(self) -> dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "warnings": self.warnings,
            "warning_count": len(self.warnings),
        }


@dataclass
class PipelineBuildEvent(PipelineEvent):
    """Event emitted during pipeline building."""

    pipeline_name: str
    message: str = ""
    node_count: int = 0
    session_id: str = "build"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize event after dataclass creation."""
        _init_event(EventType.PIPELINE_BUILD_STARTED, "build")(self)

    def _extra_fields(self) -> dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "message": self.message,
            "node_count": self.node_count,
        }


@dataclass
class LLMPromptGeneratedEvent(PipelineEvent):
    """Event emitted when an LLM prompt is generated."""

    node_name: str
    messages: list[dict[str, str]]
    template: str = ""
    template_vars: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize event after dataclass creation."""
        _init_event(EventType.LLM_PROMPT_GENERATED)(self)

    def _extra_fields(self) -> dict[str, Any]:
        return {
            "node_name": self.node_name,
            "template": self.template,
            "message_count": len(self.messages),
            "total_prompt_length": sum(len(msg["content"]) for msg in self.messages),
            "template_vars": self.template_vars,
        }


@dataclass
class LLMResponseReceivedEvent(PipelineEvent):
    """Event emitted when an LLM response is received."""

    node_name: str
    response: str
    response_length: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize event after dataclass creation."""
        _init_event(EventType.LLM_RESPONSE_RECEIVED)(self)

    def _extra_fields(self) -> dict[str, Any]:
        return {
            "node_name": self.node_name,
            "response_length": len(self.response),
            "response_preview": (
                self.response[:200] + "..." if len(self.response) > 200 else self.response
            ),
        }


@dataclass
class ToolCalledEvent(PipelineEvent):
    """Event emitted when a tool is called."""

    node_name: str
    tool_name: str
    tool_params: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize event after dataclass creation."""
        _init_event(EventType.TOOL_CALLED)(self)

    def _extra_fields(self) -> dict[str, Any]:
        return {
            "node_name": self.node_name,
            "tool_name": self.tool_name,
            "tool_params": self.tool_params,
        }


@dataclass
class ToolCompletedEvent(PipelineEvent):
    """Event emitted when a tool call completes."""

    node_name: str
    tool_name: str
    result: Any
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize event after dataclass creation."""
        _init_event(EventType.TOOL_COMPLETED)(self)

    def _extra_fields(self) -> dict[str, Any]:
        return {
            "node_name": self.node_name,
            "tool_name": self.tool_name,
            "result_type": type(self.result).__name__,
            "execution_time": self.execution_time,
        }
