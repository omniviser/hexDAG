"""Simple event data classes and registry metadata."""

from __future__ import annotations

import logging
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
        ("response", "duration_ms"),
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


# Policy-related events for tracking policy evaluations and decisions
@dataclass
class PolicyEvaluated(Event):
    """Event emitted after a policy has been evaluated."""

    policy_name: str
    signal: str  # The signal returned by the policy
    duration_ms: float
    context_node: str | None = None

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for policy evaluation."""
        _ = log_level
        return (
            f"📋 Policy '{self.policy_name}' evaluated -> {self.signal} ({self.duration_ms:.1f}ms)"
        )


@dataclass
class PolicyTriggered(Event):
    """Event emitted when a policy's condition is triggered."""

    policy_name: str
    trigger_reason: str
    context_node: str | None = None

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for policy trigger."""
        _ = log_level
        node_info = f" for node '{self.context_node}'" if self.context_node else ""
        return f"🎯 Policy '{self.policy_name}' triggered{node_info}: {self.trigger_reason}"


@dataclass
class PolicySkipped(Event):
    """Event emitted when a policy causes a node to be skipped."""

    policy_name: str
    node_name: str
    reason: str | None = None

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for policy skip."""
        _ = log_level
        reason_info = f": {self.reason}" if self.reason else ""
        return f"⏭️ Policy '{self.policy_name}' skipped node '{self.node_name}'{reason_info}"


@dataclass
class PolicyFallback(Event):
    """Event emitted when a policy provides a fallback value."""

    policy_name: str
    node_name: str
    fallback_value: Any
    reason: str | None = None

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for policy fallback."""
        _ = log_level
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

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for policy retry."""
        _ = log_level
        attempt_info = f" (attempt {self.attempt}"
        if self.max_attempts:
            attempt_info += f"/{self.max_attempts}"
        attempt_info += ")"
        delay_info = f" with {self.delay_ms:.0f}ms delay" if self.delay_ms else ""
        return (
            f"🔁 Policy '{self.policy_name}' retrying node "
            f"'{self.node_name}'{attempt_info}{delay_info}"
        )


# Lifecycle instrumentation events
@dataclass
class LifecycleEvent(Event):
    """Base instrumentation event emitted by @emits_events."""

    component_type: str
    component_name: str | None
    function_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] | None = None
    duration_ms: float | None = None
    result: Any | None = None
    error: str | None = None
    exception_type: str | None = None
    phase: str = field(init=False)
    event_type_override: str = field(init=False)

    def __post_init__(self) -> None:
        """Populate derived fields after initialization."""
        # Ensure phase is provided by subclasses
        if not getattr(self, "phase", None):
            raise ValueError("LifecycleEvent subclasses must define a phase value")
        self.event_type_override = f"{self.component_type}:{self.phase}"

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Basic log message describing the lifecycle transition."""
        _ = log_level
        component = self.component_name or "unknown"
        return f"{self.component_type} '{component}' {self.phase}"


@dataclass
class LifecycleEventStarted(LifecycleEvent):
    """Lifecycle event emitted when an instrumented function starts."""

    phase: str = field(init=False, default="started")


@dataclass
class LifecycleEventCompleted(LifecycleEvent):
    """Lifecycle event emitted when an instrumented function completes."""

    phase: str = field(init=False, default="completed")


@dataclass
class LifecycleEventFailed(LifecycleEvent):
    """Lifecycle event emitted when an instrumented function fails."""

    phase: str = field(init=False, default="failed")


# Register lifecycle events in registry metadata
EVENT_REGISTRY.update({
    "LifecycleEventStarted": EventSpec(
        "component:started",
        {"component": "component_name", "function": "function_name"},
        ("component_type", "metadata", "payload"),
    ),
    "LifecycleEventCompleted": EventSpec(
        "component:completed",
        {"component": "component_name", "function": "function_name"},
        ("component_type", "metadata", "payload", "duration_ms", "result"),
    ),
    "LifecycleEventFailed": EventSpec(
        "component:failed",
        {"component": "component_name", "function": "function_name"},
        (
            "component_type",
            "metadata",
            "payload",
            "duration_ms",
            "error",
            "exception_type",
        ),
    ),
})
