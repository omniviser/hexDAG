"""Observer models and data classes.

This module provides typed data structures for observer implementations,
following the framework's "Pydantic validation everywhere" principle.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AlertType(StrEnum):
    """Types of alerts that can be triggered."""

    SLOW_NODE = "SLOW_NODE"
    NODE_FAILURE = "NODE_FAILURE"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    QUALITY_ISSUE = "QUALITY_ISSUE"


class AlertSeverity(StrEnum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(slots=True)
class Alert:
    """Typed alert data structure.

    Replaces untyped dict[str, Any] with proper validation and type safety.

    Attributes
    ----------
    type : AlertType
        Type of alert
    node : str
        Node that triggered the alert
    message : str
        Human-readable alert message
    timestamp : float
        Unix timestamp when alert was triggered
    severity : AlertSeverity
        Alert severity level
    duration_ms : float, optional
        Duration in milliseconds (for slow node alerts)
    threshold_ms : float, optional
        Threshold that was exceeded (for slow node alerts)
    error : str, optional
        Error message (for failure alerts)
    metadata : dict[str, Any]
        Additional alert metadata
    """

    type: AlertType
    node: str
    message: str
    timestamp: float
    severity: AlertSeverity = AlertSeverity.WARNING
    duration_ms: float | None = None
    threshold_ms: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NodeMetrics:
    """Consolidated metrics for a single node.

    Uses single dataclass instead of multiple parallel dict structures.
    Follows the HandlerEntry pattern from README.md.

    Attributes
    ----------
    timings : list[float]
        List of execution times in milliseconds
    executions : int
        Total number of executions
    failures : int
        Total number of failures
    """

    timings: list[float] = field(default_factory=list)
    executions: int = 0
    failures: int = 0

    @property
    def average_ms(self) -> float:
        """Calculate average execution time."""
        return sum(self.timings) / len(self.timings) if self.timings else 0.0

    @property
    def min_ms(self) -> float:
        """Get minimum execution time."""
        return min(self.timings) if self.timings else 0.0

    @property
    def max_ms(self) -> float:
        """Get maximum execution time."""
        return max(self.timings) if self.timings else 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.executions == 0:
            return 0.0
        successes = self.executions - self.failures
        return (successes / self.executions) * 100.0
