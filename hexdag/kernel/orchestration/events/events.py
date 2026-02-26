"""Simple event data classes for the hexDAG event system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class Event:
    """Base class for all events - provides timestamp."""

    timestamp: datetime = field(default_factory=datetime.now, init=False)

    def log_message(self) -> str:
        """Get a formatted log message for this event.

        Override in subclasses to provide custom formatting.

        Returns
        -------
        str
            A formatted string suitable for logging
        """
        return f"{self.__class__.__name__} at {self.timestamp.isoformat()}"


# Node events
@dataclass(slots=True)
class NodeStarted(Event):
    """A node has started execution."""

    name: str
    wave_index: int
    dependencies: tuple[str, ...] | list[str] = field(default_factory=tuple)

    def log_message(self) -> str:
        """Format log message for node start event.

        Returns
        -------
        str
            Formatted log message for node start
        """
        deps = f" (deps: {', '.join(self.dependencies)})" if self.dependencies else ""
        return f"🚀 Node '{self.name}' started in wave {self.wave_index}{deps}"


@dataclass(slots=True)
class NodeCompleted(Event):
    """A node has completed successfully."""

    name: str
    wave_index: int
    result: Any
    duration_ms: float

    def log_message(self) -> str:
        """Format log message for node completion event.

        Returns
        -------
        str
            Formatted log message for node completion
        """
        return f"✅ Node '{self.name}' completed in {self.duration_ms / 1000:.2f}s"


@dataclass(slots=True)
class NodeFailed(Event):
    """A node has failed."""

    name: str
    wave_index: int
    error: Exception

    def log_message(self) -> str:
        """Format log message for node failure event.

        Returns
        -------
        str
            Formatted log message for node failure
        """
        return f"❌ Node '{self.name}' failed: {self.error}"


@dataclass(slots=True)
class NodeCancelled(Event):
    """A node execution was cancelled.

    Attributes
    ----------
    name : str
        Node name
    wave_index : int
        Wave index where node was executing
    reason : str | None
        Reason for cancellation (e.g., "timeout", "user_cancel")
    """

    name: str
    wave_index: int
    reason: str | None = None

    def log_message(self) -> str:
        """Format log message for node cancellation event."""
        return f"🚫 Node '{self.name}' cancelled: {self.reason or 'unknown'}"


@dataclass(slots=True)
class NodeSkipped(Event):
    """A node execution was skipped due to when clause evaluation.

    Attributes
    ----------
    name : str
        Node name
    wave_index : int
        Wave index where node was scheduled
    reason : str | None
        Reason for skipping (e.g., "when clause 'status == active' evaluated to False")
    """

    name: str
    wave_index: int
    reason: str | None = None

    def log_message(self) -> str:
        """Format log message for node skip event."""
        return f"⏭️ Node '{self.name}' skipped: {self.reason or 'unknown'}"


# Wave events (WaveStarted absorbed into WaveCompleted)
@dataclass(slots=True)
class WaveCompleted(Event):
    """A wave has completed.

    Attributes
    ----------
    wave_index : int
        Index of the completed wave
    duration_ms : float
        Duration of the wave in milliseconds
    nodes : list[str]
        Node names that were part of this wave
    """

    wave_index: int
    duration_ms: float
    nodes: list[str] = field(default_factory=list)

    def log_message(self) -> str:
        """Format log message for wave completion event.

        Returns
        -------
        str
            Formatted log message for wave completion
        """
        node_info = f" ({len(self.nodes)} nodes)" if self.nodes else ""
        return f"✅ Wave {self.wave_index} completed in {self.duration_ms / 1000:.2f}s{node_info}"


# Pipeline events (PipelineCancelled absorbed into PipelineCompleted)
@dataclass(slots=True)
class PipelineStarted(Event):
    """Pipeline execution has started."""

    name: str
    total_waves: int
    total_nodes: int

    def log_message(self) -> str:
        """Format log message for pipeline start event.

        Returns
        -------
        str
            Formatted log message for pipeline start
        """
        return (
            f"🎬 Pipeline '{self.name}' started "
            f"({self.total_nodes} nodes, {self.total_waves} waves)"
        )


@dataclass(slots=True)
class PipelineCompleted(Event):
    """Pipeline has completed (successfully, cancelled, or failed).

    Attributes
    ----------
    name : str
        Pipeline name
    duration_ms : float
        Total pipeline duration in milliseconds
    node_results : dict[str, Any]
        Results from completed nodes
    status : str
        Completion status: "completed", "cancelled", or "failed"
    reason : str | None
        Reason for cancellation/failure (None when completed successfully)
    """

    name: str
    duration_ms: float
    node_results: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"
    reason: str | None = None

    def log_message(self) -> str:
        """Format log message for pipeline completion event.

        Returns
        -------
        str
            Formatted log message for pipeline completion
        """
        if self.status == "cancelled":
            completed_nodes = len(self.node_results)
            return (
                f"🛑 Pipeline '{self.name}' cancelled after {self.duration_ms / 1000:.2f}s "
                f"({completed_nodes} nodes completed): {self.reason or 'unknown'}"
            )
        return f"🎉 Pipeline '{self.name}' completed in {self.duration_ms / 1000:.2f}s"
