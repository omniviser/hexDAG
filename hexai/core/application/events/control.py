"""Control signals and responses for execution control."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ControlSignal(Enum):
    """Signals that control handlers can return to affect execution flow."""

    PROCEED = "proceed"  # Continue normal execution
    RETRY = "retry"  # Retry the node execution
    SKIP = "skip"  # Skip this node entirely
    FALLBACK = "fallback"  # Use a fallback value instead of executing
    FAIL = "fail"  # Fail immediately with error


@dataclass
class ControlResponse:
    """Response from control handlers with signal and optional data."""

    signal: ControlSignal = ControlSignal.PROCEED
    data: Any = None  # Fallback value for FALLBACK, error for FAIL, etc.

    def should_interrupt(self) -> bool:
        """Check if this response should interrupt normal flow."""
        return self.signal != ControlSignal.PROCEED
