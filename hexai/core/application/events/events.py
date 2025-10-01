"""Simple event data classes - just data, no behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
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
@dataclass
class NodeStarted(Event):
    """A node has started execution."""

    name: str
    wave_index: int
    dependencies: list[str] = field(default_factory=list)

    def log_message(self) -> str:
        """Format log message for node start event.

        Returns
        -------
        str
            Formatted log message for node start
        """
        deps = f" (deps: {', '.join(self.dependencies)})" if self.dependencies else ""
        return f"🚀 Node '{self.name}' started in wave {self.wave_index}{deps}"


@dataclass
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


@dataclass
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


@dataclass
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


# Wave events
@dataclass
class WaveStarted(Event):
    """A wave of parallel nodes has started."""

    wave_index: int
    nodes: list[str]

    def log_message(self) -> str:
        """Format log message for wave start event.

        Returns
        -------
        str
            Formatted log message for wave start
        """
        return f"🌊 Wave {self.wave_index} started with {len(self.nodes)} nodes"


@dataclass
class WaveCompleted(Event):
    """A wave has completed."""

    wave_index: int
    duration_ms: float

    def log_message(self) -> str:
        """Format log message for wave completion event.

        Returns
        -------
        str
            Formatted log message for wave completion
        """
        return f"✅ Wave {self.wave_index} completed in {self.duration_ms / 1000:.2f}s"


# Pipeline events
@dataclass
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


@dataclass
class PipelineCompleted(Event):
    """Pipeline has completed successfully."""

    name: str
    duration_ms: float
    node_results: dict[str, Any] = field(default_factory=dict)

    def log_message(self) -> str:
        """Format log message for pipeline completion event.

        Returns
        -------
        str
            Formatted log message for pipeline completion
        """
        return f"🎉 Pipeline '{self.name}' completed in {self.duration_ms / 1000:.2f}s"


@dataclass
class PipelineCancelled(Event):
    """Pipeline execution was cancelled.

    Attributes
    ----------
    name : str
        Pipeline name
    duration_ms : float
        Duration until cancellation in milliseconds
    reason : str | None
        Reason for cancellation (e.g., "timeout", "user_cancel")
    partial_results : dict[str, Any]
        Results from nodes that completed before cancellation
    """

    name: str
    duration_ms: float
    reason: str | None = None
    partial_results: dict[str, Any] = field(default_factory=dict)

    def log_message(self) -> str:
        """Format log message for pipeline cancellation event."""
        completed_nodes = len(self.partial_results)
        return (
            f"🛑 Pipeline '{self.name}' cancelled after {self.duration_ms / 1000:.2f}s "
            f"({completed_nodes} nodes completed): {self.reason or 'unknown'}"
        )


# LLM events
@dataclass
class LLMPromptSent(Event):
    """LLM prompt has been sent."""

    node_name: str
    messages: list[dict[str, str]]

    def log_message(self) -> str:
        """Format log message for LLM prompt event.

        Returns
        -------
        str
            Formatted log message for LLM prompt
        """
        return f"💭 LLM prompt sent from '{self.node_name}' ({len(self.messages)} messages)"


@dataclass
class LLMResponseReceived(Event):
    """LLM response has been received."""

    node_name: str
    response: str
    duration_ms: float

    def log_message(self) -> str:
        """Format log message for LLM response event.

        Returns
        -------
        str
            Formatted log message for LLM response
        """
        return f"🤖 LLM response for '{self.node_name}' in {self.duration_ms / 1000:.2f}s"


# Tool events
@dataclass
class ToolCalled(Event):
    """A tool has been invoked."""

    node_name: str
    tool_name: str
    params: dict[str, Any]

    def log_message(self) -> str:
        """Format log message for tool call event.

        Returns
        -------
        str
            Formatted log message for tool call
        """
        return f"🔧 Tool '{self.tool_name}' called from '{self.node_name}'"


@dataclass
class ToolCompleted(Event):
    """A tool has completed."""

    node_name: str
    tool_name: str
    result: Any
    duration_ms: float

    def log_message(self) -> str:
        """Format log message for tool completion event.

        Returns
        -------
        str
            Formatted log message for tool completion
        """
        return f"✅ Tool '{self.tool_name}' completed in {self.duration_ms / 1000:.2f}s"


# Policy-related events for tracking policy evaluations and decisions
@dataclass
class PolicyEvaluated(Event):
    """Event emitted after a policy has been evaluated."""

    policy_name: str
    signal: str  # The signal returned by the policy
    duration_ms: float
    context_node: str | None = None

    def log_message(self) -> str:
        """Format log message for policy evaluation."""
        return (
            f"📋 Policy '{self.policy_name}' evaluated -> {self.signal} ({self.duration_ms:.1f}ms)"
        )


@dataclass
class PolicyTriggered(Event):
    """Event emitted when a policy's condition is triggered."""

    policy_name: str
    trigger_reason: str
    context_node: str | None = None

    def log_message(self) -> str:
        """Format log message for policy trigger."""
        node_info = f" for node '{self.context_node}'" if self.context_node else ""
        return f"🎯 Policy '{self.policy_name}' triggered{node_info}: {self.trigger_reason}"


@dataclass
class PolicySkipped(Event):
    """Event emitted when a policy causes a node to be skipped."""

    policy_name: str
    node_name: str
    reason: str | None = None

    def log_message(self) -> str:
        """Format log message for policy skip."""
        reason_info = f": {self.reason}" if self.reason else ""
        return f"⏭️ Policy '{self.policy_name}' skipped node '{self.node_name}'{reason_info}"


@dataclass
class PolicyFallback(Event):
    """Event emitted when a policy provides a fallback value."""

    policy_name: str
    node_name: str
    fallback_value: Any
    reason: str | None = None

    def log_message(self) -> str:
        """Format log message for policy fallback."""
        reason_info = f": {self.reason}" if self.reason else ""
        return (
            f"🔄 Policy '{self.policy_name}' provided fallback for '{self.node_name}'{reason_info}"
        )


@dataclass
class PolicyRetry(Event):
    """Event emitted when a policy triggers a retry."""

    policy_name: str
    node_name: str
    attempt: int
    delay_ms: float | None = None
    max_attempts: int | None = None

    def log_message(self) -> str:
        """Format log message for policy retry."""
        attempt_info = f" (attempt {self.attempt}"
        if self.max_attempts:
            attempt_info += f"/{self.max_attempts}"
        attempt_info += ")"
        delay_info = f" with {self.delay_ms:.0f}ms delay" if self.delay_ms else ""
        return (
            f"🔁 Policy '{self.policy_name}' retrying node "
            f"'{self.node_name}'{attempt_info}{delay_info}"
        )
