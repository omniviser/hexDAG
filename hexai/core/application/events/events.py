"""Simple event data classes - just data, no behavior."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Event:
    """Base class for all events - provides timestamp."""

    timestamp: datetime = field(default_factory=datetime.now, init=False)

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Get a formatted log message for this event.

        Override in subclasses to provide custom formatting.

        Args
        ----
            log_level: The logging level (can be used for different verbosity)

        Returns
        -------
        str
            A formatted string suitable for logging
        """
        _ = log_level  # Mark as intentionally unused in base class
        return f"{self.__class__.__name__} at {self.timestamp.isoformat()}"


# Node events
@dataclass
class NodeStarted(Event):
    """A node has started execution."""

    name: str
    wave_index: int
    dependencies: list[str] = field(default_factory=list)

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for node start event.

        Returns
        -------
        str
            Formatted log message for node start
        """
        _ = log_level  # Unused but kept for interface consistency
        deps = f" (deps: {', '.join(self.dependencies)})" if self.dependencies else ""
        return f"🚀 Node '{self.name}' started in wave {self.wave_index}{deps}"


@dataclass
class NodeCompleted(Event):
    """A node has completed successfully."""

    name: str
    wave_index: int
    result: Any
    duration_ms: float

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for node completion event.

        Returns
        -------
        str
            Formatted log message for node completion
        """
        _ = log_level  # Unused but kept for interface consistency
        return f"✅ Node '{self.name}' completed in {self.duration_ms / 1000:.2f}s"


@dataclass
class NodeFailed(Event):
    """A node has failed."""

    name: str
    wave_index: int
    error: Exception

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for node failure event.

        Returns
        -------
        str
            Formatted log message for node failure
        """
        _ = log_level  # This event always uses ERROR level
        return f"❌ Node '{self.name}' failed: {self.error}"


# Wave events
@dataclass
class WaveStarted(Event):
    """A wave of parallel nodes has started."""

    wave_index: int
    nodes: list[str]

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for wave start event.

        Returns
        -------
        str
            Formatted log message for wave start
        """
        _ = log_level  # Unused but kept for interface consistency
        return f"🌊 Wave {self.wave_index} started with {len(self.nodes)} nodes"


@dataclass
class WaveCompleted(Event):
    """A wave has completed."""

    wave_index: int
    duration_ms: float

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for wave completion event.

        Returns
        -------
        str
            Formatted log message for wave completion
        """
        _ = log_level  # Unused but kept for interface consistency
        return f"✅ Wave {self.wave_index} completed in {self.duration_ms / 1000:.2f}s"


# Pipeline events
@dataclass
class PipelineStarted(Event):
    """Pipeline execution has started."""

    name: str
    total_waves: int
    total_nodes: int

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for pipeline start event.

        Returns
        -------
        str
            Formatted log message for pipeline start
        """
        _ = log_level  # Unused but kept for interface consistency
        return (
            f"🎬 Pipeline '{self.name}' started "
            f"({self.total_nodes} nodes, {self.total_waves} waves)"
        )


@dataclass
class PipelineCompleted(Event):
    """Pipeline has completed successfully."""

    name: str
    duration_ms: float
    node_results: dict[str, Any] = field(default_factory=dict)

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for pipeline completion event.

        Returns
        -------
        str
            Formatted log message for pipeline completion
        """
        _ = log_level  # Unused but kept for interface consistency
        return f"🎉 Pipeline '{self.name}' completed in {self.duration_ms / 1000:.2f}s"


# LLM events
@dataclass
class LLMPromptSent(Event):
    """LLM prompt has been sent."""

    node_name: str
    messages: list[dict[str, str]]

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for LLM prompt event.

        Returns
        -------
        str
            Formatted log message for LLM prompt
        """
        _ = log_level  # Unused but kept for interface consistency
        return f"💭 LLM prompt sent from '{self.node_name}' ({len(self.messages)} messages)"


@dataclass
class LLMResponseReceived(Event):
    """LLM response has been received."""

    node_name: str
    response: str
    duration_ms: float

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for LLM response event.

        Returns
        -------
        str
            Formatted log message for LLM response
        """
        _ = log_level  # Unused but kept for interface consistency
        return f"🤖 LLM response for '{self.node_name}' in {self.duration_ms / 1000:.2f}s"


# Tool events
@dataclass
class ToolCalled(Event):
    """A tool has been invoked."""

    node_name: str
    tool_name: str
    params: dict[str, Any]

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for tool call event.

        Returns
        -------
        str
            Formatted log message for tool call
        """
        _ = log_level  # Unused but kept for interface consistency
        return f"🔧 Tool '{self.tool_name}' called from '{self.node_name}'"


@dataclass
class ToolCompleted(Event):
    """A tool has completed."""

    node_name: str
    tool_name: str
    result: Any
    duration_ms: float

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for tool completion event.

        Returns
        -------
        str
            Formatted log message for tool completion
        """
        _ = log_level  # Unused but kept for interface consistency
        return f"✅ Tool '{self.tool_name}' completed in {self.duration_ms / 1000:.2f}s"
