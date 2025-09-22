"""Simple event data classes - just data, no behavior."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hexai.core.application.policies.models import PolicySignal


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
            A formatted string suitable for logging
        """
        del log_level  # Unused in base implementation
        return f"{self.__class__.__name__} at {self.timestamp.isoformat()}"


# Node events
@dataclass
class NodeStarted(Event):
    """A node has started execution."""

    name: str
    wave_index: int
    dependencies: list[str] = field(default_factory=list)

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for node start event."""
        del log_level  # Unused
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
        """Format log message for node completion event."""
        del log_level  # Unused
        return f"✅ Node '{self.name}' completed in {self.duration_ms / 1000:.2f}s"


@dataclass
class NodeFailed(Event):
    """A node has failed."""

    name: str
    wave_index: int
    error: Exception

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for node failure event."""
        del log_level  # Unused
        return f"❌ Node '{self.name}' failed: {self.error}"


# Wave events
@dataclass
class WaveStarted(Event):
    """A wave of parallel nodes has started."""

    wave_index: int
    nodes: list[str]

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for wave start event."""
        del log_level  # Unused
        return f"🌊 Wave {self.wave_index} started with {len(self.nodes)} nodes"


@dataclass
class WaveCompleted(Event):
    """A wave has completed."""

    wave_index: int
    duration_ms: float

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for wave completion event."""
        del log_level  # Unused
        return f"✅ Wave {self.wave_index} completed in {self.duration_ms / 1000:.2f}s"


# Pipeline events
@dataclass
class PipelineStarted(Event):
    """Pipeline execution has started."""

    name: str
    total_waves: int
    total_nodes: int

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for pipeline start event."""
        del log_level  # Unused
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
        """Format log message for pipeline completion event."""
        del log_level  # Unused
        return f"🎉 Pipeline '{self.name}' completed in {self.duration_ms / 1000:.2f}s"


# LLM events
@dataclass
class LLMPromptSent(Event):
    """LLM prompt has been sent."""

    node_name: str
    messages: list[dict[str, str]]

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for LLM prompt event."""
        del log_level  # Unused
        return f"💭 LLM prompt sent from '{self.node_name}' ({len(self.messages)} messages)"


@dataclass
class LLMResponseReceived(Event):
    """LLM response has been received."""

    node_name: str
    response: str
    duration_ms: float

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for LLM response event."""
        del log_level  # Unused
        return f"🤖 LLM response for '{self.node_name}' in {self.duration_ms / 1000:.2f}s"


# Tool events
@dataclass
class ToolCalled(Event):
    """A tool has been invoked."""

    node_name: str
    tool_name: str
    params: dict[str, Any]

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for tool call event."""
        del log_level  # Unused
        return f"🔧 Tool '{self.tool_name}' called from '{self.node_name}'"


@dataclass
class ToolCompleted(Event):
    """A tool has completed."""

    node_name: str
    tool_name: str
    result: Any
    duration_ms: float

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for tool completion event."""
        del log_level  # Unused
        return f"✅ Tool '{self.tool_name}' completed in {self.duration_ms / 1000:.2f}s"


# Policy events
@dataclass
class PolicyEvaluated(Event):
    """A policy has been evaluated."""

    context_point: str  # Where in the execution (e.g., "pipeline_start", "node_start")
    dag_id: str
    node_id: str | None
    signal: PolicySignal
    data: Any = None
    duration_ms: float = 0

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for policy evaluation event."""
        del log_level  # Unused
        node_info = f" for node '{self.node_id}'" if self.node_id else ""
        return f"⚖️ Policy evaluated at {self.context_point}{node_info}: {self.signal.value}"


@dataclass
class PolicyTriggered(Event):
    """A policy has triggered an action (non-PROCEED signal)."""

    context_point: str
    dag_id: str
    node_id: str | None
    signal: PolicySignal
    data: Any = None
    reason: str | None = None

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for policy trigger event."""
        del log_level  # Unused
        node_info = f" for node '{self.node_id}'" if self.node_id else ""
        reason_info = f": {self.reason}" if self.reason else ""
        return (
            f"🚨 Policy triggered at {self.context_point}{node_info} "
            f"- {self.signal.value}{reason_info}"
        )


@dataclass
class PolicySkipped(Event):
    """A node was skipped due to policy decision."""

    node_name: str
    dag_id: str
    reason: str | None = None

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for policy skip event."""
        del log_level  # Unused
        reason_info = f": {self.reason}" if self.reason else ""
        return f"⏭️ Node '{self.node_name}' skipped by policy{reason_info}"


@dataclass
class PolicyRetry(Event):
    """A retry was triggered by policy decision."""

    node_name: str
    dag_id: str
    attempt: int
    delay: float = 0
    reason: str | None = None

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for policy retry event."""
        del log_level  # Unused
        delay_info = f" with {self.delay}s delay" if self.delay > 0 else ""
        reason_info = f": {self.reason}" if self.reason else ""
        return f"🔄 Node '{self.node_name}' retry #{self.attempt}{delay_info}{reason_info}"


@dataclass
class PolicyFallback(Event):
    """A fallback value was used due to policy decision."""

    node_name: str
    dag_id: str
    fallback_value: Any
    original_error: str | None = None

    def log_message(self, log_level: int = logging.INFO) -> str:
        """Format log message for policy fallback event."""
        del log_level  # Unused
        error_info = f" after error: {self.original_error}" if self.original_error else ""
        return f"↩️ Node '{self.node_name}' using fallback value{error_info}"
