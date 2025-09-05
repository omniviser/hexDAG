"""Simple event data classes - just data, no behavior."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# Node events
@dataclass
class NodeStarted:
    """A node has started execution."""

    name: str
    wave_index: int
    dependencies: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class NodeCompleted:
    """A node has completed successfully."""

    name: str
    wave_index: int
    result: Any
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class NodeFailed:
    """A node has failed."""

    name: str
    wave_index: int
    error: Exception
    timestamp: datetime = field(default_factory=datetime.now)


# Wave events
@dataclass
class WaveStarted:
    """A wave of parallel nodes has started."""

    wave_index: int
    nodes: list[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class WaveCompleted:
    """A wave has completed."""

    wave_index: int
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.now)


# Pipeline events
@dataclass
class PipelineStarted:
    """Pipeline execution has started."""

    name: str
    total_waves: int
    total_nodes: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PipelineCompleted:
    """Pipeline has completed successfully."""

    name: str
    duration_ms: float
    node_results: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


# LLM events
@dataclass
class LLMPromptSent:
    """LLM prompt has been sent."""

    node_name: str
    messages: list[dict[str, str]]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LLMResponseReceived:
    """LLM response has been received."""

    node_name: str
    response: str
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.now)


# Tool events
@dataclass
class ToolCalled:
    """A tool has been invoked."""

    node_name: str
    tool_name: str
    params: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ToolCompleted:
    """A tool has completed."""

    node_name: str
    tool_name: str
    result: Any
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.now)


# Type union for any event
Event = (
    NodeStarted
    | NodeCompleted
    | NodeFailed
    | WaveStarted
    | WaveCompleted
    | PipelineStarted
    | PipelineCompleted
    | LLMPromptSent
    | LLMResponseReceived
    | ToolCalled
    | ToolCompleted
)
