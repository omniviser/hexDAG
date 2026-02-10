"""Simple event data classes and registry metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class EventSpec:
    """Specification defining taxonomy metadata for an event class."""

    event_type: str
    envelope_fields: dict[str, str]
    attr_fields: tuple[str, ...] | None = None


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


EVENT_REGISTRY: dict[str, EventSpec] = {
    "PipelineStarted": EventSpec(
        "pipeline:started",
        {"pipeline": "name"},
        ("total_waves", "total_nodes"),
    ),
    "PipelineCompleted": EventSpec(
        "pipeline:completed",
        {"pipeline": "name"},
        ("duration_ms", "node_results"),
    ),
    "NodeStarted": EventSpec(
        "node:started",
        {"node": "name", "wave": "wave_index"},
        ("dependencies",),
    ),
    "NodeCompleted": EventSpec(
        "node:completed",
        {"node": "name", "wave": "wave_index"},
        ("result", "duration_ms"),
    ),
    "NodeFailed": EventSpec(
        "node:failed",
        {"node": "name", "wave": "wave_index"},
        ("error",),
    ),
    "WaveStarted": EventSpec(
        "wave:started",
        {"wave": "wave_index"},
        ("nodes",),
    ),
    "WaveCompleted": EventSpec(
        "wave:completed",
        {"wave": "wave_index"},
        ("duration_ms",),
    ),
    "LLMPromptSent": EventSpec(
        "llm:prompt",
        {"node": "node_name"},
        ("messages",),
    ),
    "LLMResponseReceived": EventSpec(
        "llm:response",
        {"node": "node_name"},
        ("response", "duration_ms", "usage"),
    ),
    "ToolCalled": EventSpec(
        "tool:called",
        {"node": "node_name"},
        ("tool_name", "params"),
    ),
    "ToolCompleted": EventSpec(
        "tool:completed",
        {"node": "node_name"},
        ("tool_name", "result", "duration_ms"),
    ),
}


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


# Wave events
@dataclass(slots=True)
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


@dataclass(slots=True)
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


@dataclass(slots=True)
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
@dataclass(slots=True)
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


@dataclass(slots=True)
class LLMResponseReceived(Event):
    """LLM response has been received."""

    node_name: str
    response: str
    duration_ms: float
    usage: dict[str, int] | None = None

    def log_message(self) -> str:
        """Format log message for LLM response event.

        Returns
        -------
        str
            Formatted log message for LLM response
        """
        return f"🤖 LLM response for '{self.node_name}' in {self.duration_ms / 1000:.2f}s"


# Tool events
@dataclass(slots=True)
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


@dataclass(slots=True)
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
@dataclass(slots=True)
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


@dataclass(slots=True)
class PolicyTriggered(Event):
    """Event emitted when a policy's condition is triggered."""

    policy_name: str
    trigger_reason: str
    context_node: str | None = None

    def log_message(self) -> str:
        """Format log message for policy trigger."""
        node_info = f" for node '{self.context_node}'" if self.context_node else ""
        return f"🎯 Policy '{self.policy_name}' triggered{node_info}: {self.trigger_reason}"


@dataclass(slots=True)
class PolicySkipped(Event):
    """Event emitted when a policy causes a node to be skipped."""

    policy_name: str
    node_name: str
    reason: str | None = None

    def log_message(self) -> str:
        """Format log message for policy skip."""
        reason_info = f": {self.reason}" if self.reason else ""
        return f"⏭️ Policy '{self.policy_name}' skipped node '{self.node_name}'{reason_info}"


@dataclass(slots=True)
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


@dataclass(slots=True)
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


# Checkpoint events
@dataclass(slots=True)
class CheckpointSaved(Event):
    """Emitted when a checkpoint is saved.

    Attributes
    ----------
    run_id : str
        The unique run identifier
    dag_id : str
        The DAG identifier
    node_id : str
        The node that triggered the checkpoint
    checkpoint_type : str
        Type of checkpoint ("node_completed", "wave_completed", "pipeline_paused")
    data : dict[str, Any]
        Additional checkpoint metadata
    """

    run_id: str
    dag_id: str
    node_id: str
    checkpoint_type: str
    data: dict[str, Any] = field(default_factory=dict)

    def log_message(self) -> str:
        """Format log message for checkpoint save event."""
        return (
            f"💾 Checkpoint saved for run '{self.run_id}' "
            f"(node: {self.node_id}, type: {self.checkpoint_type})"
        )


@dataclass(slots=True)
class CheckpointRestored(Event):
    """Emitted when execution resumes from a checkpoint.

    Attributes
    ----------
    run_id : str
        The unique run identifier
    dag_id : str
        The DAG identifier
    node_id : str
        The node where execution is resuming
    restored_nodes : list[str]
        List of nodes already completed (skipped on resume)
    restored_at : str
        ISO timestamp of restore operation
    """

    run_id: str
    dag_id: str
    node_id: str
    restored_nodes: list[str]
    restored_at: str

    def log_message(self) -> str:
        """Format log message for checkpoint restore event."""
        return (
            f"🔄 Checkpoint restored for run '{self.run_id}' "
            f"({len(self.restored_nodes)} nodes already completed)"
        )


# Health check events
@dataclass(slots=True)
class HealthCheckCompleted(Event):
    """Adapter health check completed.

    Emitted when an adapter's ahealth_check() method completes during pre-DAG hooks.

    Attributes
    ----------
    adapter_name : str
        Name of the adapter that was checked
    port_name : str
        Name of the port this adapter implements
    status : HealthStatus
        Health check result with status and details
    """

    adapter_name: str
    port_name: str
    status: Any  # HealthStatus, but using Any to avoid circular import

    def log_message(self) -> str:
        """Format log message for health check event."""
        status_emoji = {
            "healthy": "✅",
            "degraded": "⚠️",
            "unhealthy": "❌",
        }.get(getattr(self.status, "status", "unknown"), "ℹ️")

        latency = ""
        if hasattr(self.status, "latency_ms") and self.status.latency_ms:
            latency = f" ({self.status.latency_ms:.1f}ms)"

        return f"{status_emoji} Health check [{self.port_name}]: {self.status.status}{latency}"
