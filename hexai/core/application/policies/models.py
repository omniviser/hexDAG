"""Data models for policy management system."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class PolicySignal(Enum):
    """Signals that policies can return to control execution flow."""

    PROCEED = "proceed"  # Continue normal execution
    RETRY = "retry"  # Retry the operation
    SKIP = "skip"  # Skip this operation
    FALLBACK = "fallback"  # Use fallback value/behavior
    FAIL = "fail"  # Fail the operation


class SubscriberType(Enum):
    """Types of policy subscribers for lifecycle management."""

    CORE = "core"  # Core framework policies (strong reference)
    PLUGIN = "plugin"  # Plugin policies (strong reference)
    USER = "user"  # User-defined policies (weak reference)
    TEMPORARY = "temporary"  # Temporary policies (weak reference)


@dataclass
class PolicyContext:
    """Context information provided to policies for evaluation.

    Attributes:
        event: The event that triggered the policy evaluation
        dag_id: Identifier of the DAG being executed
        node_id: Current node being executed (optional)
        wave_index: Current wave index in the DAG execution
        attempt: Current attempt number (1-based)
        error: Exception that triggered policy evaluation (if any)
        metadata: Additional context-specific information
    """

    event: Any  # Event that triggered the evaluation
    dag_id: str
    node_id: str | None = None
    wave_index: int = 0
    attempt: int = 1
    error: Exception | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class PolicyResponse:
    """Response from a policy evaluation.

    Attributes:
        signal: The control signal indicating how to proceed
        data: Optional data to use (e.g., fallback value)
        metadata: Optional metadata about the decision
    """

    signal: PolicySignal = PolicySignal.PROCEED
    data: Any = None
    metadata: dict[str, Any] | None = None
