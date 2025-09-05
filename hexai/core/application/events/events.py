"""Simple event data classes - just data, no behavior."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Event:
    """Base class for all events - provides timestamp."""

    timestamp: datetime = field(default_factory=datetime.now, init=False)


# Node events
@dataclass
class NodeStarted(Event):
    """A node has started execution."""

    name: str
    wave_index: int
    dependencies: list[str] = field(default_factory=list)


@dataclass
class NodeCompleted(Event):
    """A node has completed successfully."""

    name: str
    wave_index: int
    result: Any
    duration_ms: float


@dataclass
class NodeFailed(Event):
    """A node has failed."""

    name: str
    wave_index: int
    error: Exception


# Wave events
@dataclass
class WaveStarted(Event):
    """A wave of parallel nodes has started."""

    wave_index: int
    nodes: list[str]


@dataclass
class WaveCompleted(Event):
    """A wave has completed."""

    wave_index: int
    duration_ms: float


# Pipeline events
@dataclass
class PipelineStarted(Event):
    """Pipeline execution has started."""

    name: str
    total_waves: int
    total_nodes: int


@dataclass
class PipelineCompleted(Event):
    """Pipeline has completed successfully."""

    name: str
    duration_ms: float
    node_results: dict[str, Any] = field(default_factory=dict)


# LLM events
@dataclass
class LLMPromptSent(Event):
    """LLM prompt has been sent."""

    node_name: str
    messages: list[dict[str, str]]


@dataclass
class LLMResponseReceived(Event):
    """LLM response has been received."""

    node_name: str
    response: str
    duration_ms: float


# Tool events
@dataclass
class ToolCalled(Event):
    """A tool has been invoked."""

    node_name: str
    tool_name: str
    params: dict[str, Any]


@dataclass
class ToolCompleted(Event):
    """A tool has completed."""

    node_name: str
    tool_name: str
    result: Any
    duration_ms: float
