"""Simple event data classes - just data, no behavior."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar


@dataclass
class Event:
    """Base class for all events - provides timestamp."""

    event_namespace: ClassVar[str]
    event_action: ClassVar[str]
    timestamp: datetime = field(default_factory=datetime.now, init=False)

    @property
    def event_type(self) -> str:
        """Return canonical <namespace>:<action> identifier."""
        try:
            namespace = self.__class__.event_namespace
            action = self.__class__.event_action
        except AttributeError as exc:
            raise NotImplementedError(
                "Event subclasses must define event_namespace and event_action"
            ) from exc
        return f"{namespace}:{action}"

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Get a formatted log message for this event.

        Override in subclasses to provide custom formatting.

        Args
        ----
            log_level: The logging level (can be used for different verbosity)

        Returns
        -------
            A formatted string suitable for logging
        """
        _ = log_level  # Mark as intentionally unused in base class
        return f"{self.__class__.__name__} at {self.timestamp.isoformat()}"


# Node events
@dataclass
class NodeStarted(Event):
    """A node has started execution."""

    event_namespace: ClassVar[str] = "node"
    event_action: ClassVar[str] = "started"
    name: str
    wave_index: int
    dependencies: list[str] = field(default_factory=list)

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for node start event."""
        _ = log_level  # Unused but kept for interface consistency
        deps = f" (deps: {', '.join(self.dependencies)})" if self.dependencies else ""
        return f"🚀 Node '{self.name}' started in wave {self.wave_index}{deps}"


@dataclass
class NodeCompleted(Event):
    """A node has completed successfully."""

    event_namespace: ClassVar[str] = "node"
    event_action: ClassVar[str] = "completed"
    name: str
    wave_index: int
    result: Any
    duration_ms: float

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for node completion event."""
        _ = log_level  # Unused but kept for interface consistency
        return f"✅ Node '{self.name}' completed in {self.duration_ms / 1000:.2f}s"


@dataclass
class NodeFailed(Event):
    """A node has failed."""

    event_namespace: ClassVar[str] = "node"
    event_action: ClassVar[str] = "failed"
    name: str
    wave_index: int
    error: Exception

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for node failure event."""
        _ = log_level  # This event always uses ERROR level
        return f"❌ Node '{self.name}' failed: {self.error}"


# Wave events
@dataclass
class WaveStarted(Event):
    """A wave of parallel nodes has started."""

    event_namespace: ClassVar[str] = "wave"
    event_action: ClassVar[str] = "started"
    wave_index: int
    nodes: list[str]

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for wave start event."""
        _ = log_level  # Unused but kept for interface consistency
        return f"🌊 Wave {self.wave_index} started with {len(self.nodes)} nodes"


@dataclass
class WaveCompleted(Event):
    """A wave has completed."""

    event_namespace: ClassVar[str] = "wave"
    event_action: ClassVar[str] = "completed"
    wave_index: int
    duration_ms: float

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for wave completion event."""
        _ = log_level  # Unused but kept for interface consistency
        return f"✅ Wave {self.wave_index} completed in {self.duration_ms / 1000:.2f}s"


# Pipeline events
@dataclass
class PipelineStarted(Event):
    """Pipeline execution has started."""

    event_namespace: ClassVar[str] = "pipeline"
    event_action: ClassVar[str] = "started"
    name: str
    total_waves: int
    total_nodes: int

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for pipeline start event."""
        _ = log_level  # Unused but kept for interface consistency
        return (
            f"🎬 Pipeline '{self.name}' started "
            f"({self.total_nodes} nodes, {self.total_waves} waves)"
        )


@dataclass
class PipelineCompleted(Event):
    """Pipeline has completed successfully."""

    event_namespace: ClassVar[str] = "pipeline"
    event_action: ClassVar[str] = "completed"
    name: str
    duration_ms: float
    node_results: dict[str, Any] = field(default_factory=dict)

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for pipeline completion event."""
        _ = log_level  # Unused but kept for interface consistency
        return f"🎉 Pipeline '{self.name}' completed in {self.duration_ms / 1000:.2f}s"


# LLM events
@dataclass
class LLMPromptSent(Event):
    """LLM prompt has been sent."""

    event_namespace: ClassVar[str] = "llm"
    event_action: ClassVar[str] = "prompt_sent"
    node_name: str
    messages: list[dict[str, str]]

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for LLM prompt event."""
        _ = log_level  # Unused but kept for interface consistency
        return f"💭 LLM prompt sent from '{self.node_name}' ({len(self.messages)} messages)"


@dataclass
class LLMResponseReceived(Event):
    """LLM response has been received."""

    event_namespace: ClassVar[str] = "llm"
    event_action: ClassVar[str] = "response_received"
    node_name: str
    response: str
    duration_ms: float

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for LLM response event."""
        _ = log_level  # Unused but kept for interface consistency
        return f"🤖 LLM response for '{self.node_name}' in {self.duration_ms / 1000:.2f}s"


# Tool events
@dataclass
class ToolCalled(Event):
    """A tool has been invoked."""

    event_namespace: ClassVar[str] = "tool"
    event_action: ClassVar[str] = "called"
    node_name: str
    tool_name: str
    params: dict[str, Any]

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for tool call event."""
        _ = log_level  # Unused but kept for interface consistency
        return f"🔧 Tool '{self.tool_name}' called from '{self.node_name}'"


@dataclass
class ToolCompleted(Event):
    """A tool has completed."""

    event_namespace: ClassVar[str] = "tool"
    event_action: ClassVar[str] = "completed"
    node_name: str
    tool_name: str
    result: Any
    duration_ms: float

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for tool completion event."""
        _ = log_level  # Unused but kept for interface consistency
        return f"✅ Tool '{self.tool_name}' completed in {self.duration_ms / 1000:.2f}s"
