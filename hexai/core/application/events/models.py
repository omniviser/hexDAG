"""Data models for the event system.

This module contains all data classes and enums used by the event system,
providing a single location for model definitions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol, Union

# Control Models
# --------------


class ControlSignal(Enum):
    """Signals that control handlers can return to affect execution flow."""

    PROCEED = "proceed"  # Continue normal execution
    RETRY = "retry"  # Retry the node execution
    SKIP = "skip"  # Skip this node entirely
    FALLBACK = "fallback"  # Use a fallback value instead of executing
    FAIL = "fail"  # Fail immediately with error
    ERROR = "error"  # Policy encountered an error and must not silently fail


@dataclass
class ControlResponse:
    """Response from control handlers with signal and optional data."""

    signal: ControlSignal = ControlSignal.PROCEED
    data: Any = None  # Fallback value for FALLBACK, error for FAIL, etc.

    def should_interrupt(self) -> bool:
        """Check if this response should interrupt normal flow."""
        return self.signal != ControlSignal.PROCEED


# Handler Models
# --------------


@dataclass
class HandlerMetadata:
    """Metadata for handler registration."""

    priority: int = 100  # Lower number = higher priority
    name: str = ""
    description: str = ""


# Observer Models
# --------------


class Observer(Protocol):
    """Protocol for observers that monitor events."""

    async def handle(self, event: Any) -> None:
        """Handle an event."""
        ...


# Types that can be observers
ObserverFunc = Callable[[Any], None]
AsyncObserverFunc = Callable[[Any], Any]  # Returns awaitable
ObserverLike = Union[Observer, ObserverFunc, AsyncObserverFunc]


# Note: ExecutionContext is defined in context.py due to its specific frozen/immutable design
# Note: Event classes are defined in events.py to maintain existing structure
