"""Data models and protocols for the event system.

This module contains all data classes, protocols, and types used by the event system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

# Execution Context
# -----------------


@dataclass
class ExecutionContext:
    """Context that flows through node and event execution.

    Carries metadata through the execution pipeline.
    """

    dag_id: str
    node_id: str | None = None
    wave_index: int = 0
    attempt: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_node(self, node_id: str, wave_index: int) -> "ExecutionContext":
        """Create new context for a specific node execution."""
        return ExecutionContext(
            dag_id=self.dag_id,
            node_id=node_id,
            wave_index=wave_index,
            attempt=self.attempt,
            metadata=self.metadata.copy(),
        )

    def with_attempt(self, attempt: int) -> "ExecutionContext":
        """Create new context with updated attempt number."""
        return ExecutionContext(
            dag_id=self.dag_id,
            node_id=self.node_id,
            wave_index=self.wave_index,
            attempt=attempt,
            metadata=self.metadata.copy(),
        )


# Control Models
# --------------


class ControlSignal(Enum):
    """Signals that control handlers can return to affect execution flow."""

    PROCEED = "proceed"  # Continue normal execution
    RETRY = "retry"  # Retry the node execution
    SKIP = "skip"  # Skip this node entirely
    FALLBACK = "fallback"  # Use a fallback value instead
    FAIL = "fail"  # Fail immediately with error
    ERROR = "error"  # Policy error, must not silently fail


@dataclass
class ControlResponse:
    """Response from control handlers with signal and optional data."""

    signal: ControlSignal = ControlSignal.PROCEED
    data: Any = None  # Fallback value, error message, etc.

    def should_interrupt(self) -> bool:
        """Check if this response should interrupt normal flow."""
        return self.signal != ControlSignal.PROCEED


# Handler Metadata
# ----------------


@dataclass
class HandlerMetadata:
    """Metadata for handler registration."""

    priority: int = 100  # Lower number = higher priority
    name: str = ""
    description: str = ""


# Observer Protocol
# -----------------


class Observer(Protocol):
    """Protocol for observers that monitor events."""

    async def handle(self, event: Any) -> None:
        """Handle an event (read-only, no return value)."""
        ...


# Control Handler Protocol
# ------------------------


class ControlHandler(Protocol):
    """Protocol for control handlers that can affect execution."""

    async def handle(self, event: Any, context: ExecutionContext) -> ControlResponse:
        """Handle an event and return control response."""
        ...


# Base Manager ABC
# ----------------


class BaseEventManager(ABC):
    """Abstract base class for event system managers.

    Defines common interface for both ObserverManager and ControlManager.
    """

    def __init__(self) -> None:
        """Initialize the base manager."""
        self._handlers: dict[str, Any] = {}

    @abstractmethod
    def register(self, handler: Any, **kwargs: Any) -> Any:
        """Register a handler with the manager.

        Args
        ----
            handler: The handler to register
            **kwargs: Additional registration parameters

        Returns
        -------
            Registration identifier or None
        """
        ...

    def unregister(self, handler_id: str) -> bool:
        """Unregister a handler by ID.

        Args
        ----
            handler_id: The ID of the handler to unregister

        Returns
        -------
            bool: True if handler was found and removed, False otherwise
        """
        if handler_id in self._handlers:
            del self._handlers[handler_id]
            return True
        return False

    def clear(self) -> None:
        """Remove all registered handlers."""
        self._handlers.clear()

    def __len__(self) -> int:
        """Return number of registered handlers."""
        return len(self._handlers)

    async def close(self) -> None:
        """Close the manager and cleanup resources."""
        self.clear()


# Type Aliases
# ------------

# Observer types
ObserverFunc = Callable[[Any], None]
AsyncObserverFunc = Callable[[Any], Any]  # Returns awaitable

# Control handler types
ControlHandlerFunc = Callable[[Any, ExecutionContext], ControlResponse]
AsyncControlHandlerFunc = Callable[[Any, ExecutionContext], Any]  # Returns awaitable
