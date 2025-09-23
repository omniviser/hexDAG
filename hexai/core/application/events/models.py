"""Data models and protocols for the event system.

This module contains all data classes, protocols, and types used by the event system.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .events import Event

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
        """Create new context for a specific node execution.

        Returns
        -------
        ExecutionContext
            New context with updated node and wave information
        """
        return ExecutionContext(
            dag_id=self.dag_id,
            node_id=node_id,
            wave_index=wave_index,
            attempt=self.attempt,
            metadata=self.metadata.copy(),
        )

    def with_attempt(self, attempt: int) -> "ExecutionContext":
        """Create new context with updated attempt number.

        Returns
        -------
        ExecutionContext
            New context with updated attempt number
        """
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
        """Check if this response should interrupt normal flow.

        Returns
        -------
        bool
            True if signal is not PROCEED, indicating flow interruption
        """
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

    async def handle(self, event: "Event") -> None:
        """Handle an event (read-only, no return value)."""
        ...


# Control Handler Protocol
# ------------------------


class ControlHandler(Protocol):
    """Protocol for control handlers that can affect execution."""

    async def handle(self, event: "Event", context: ExecutionContext) -> ControlResponse:
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
        Any
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
        bool
            True if handler was found and removed, False otherwise
        """
        if handler_id in self._handlers:
            del self._handlers[handler_id]
            return True
        return False

    def clear(self) -> None:
        """Remove all registered handlers."""
        self._handlers.clear()

    def __len__(self) -> int:
        """Return number of registered handlers.

        Returns
        -------
        int
            Total number of registered handlers
        """
        # Count handlers from appropriate storage
        # Subclasses may override if they use different storage
        count = len(self._handlers)

        # Check if this is an ObserverManager with weak refs
        # Use getattr to avoid type checker complaints
        use_weak_refs = getattr(self, "_use_weak_refs", False)
        weak_handlers = getattr(self, "_weak_handlers", None)

        if use_weak_refs and weak_handlers is not None:
            # Count weak handlers that are still alive
            for handler_id in list(weak_handlers.keys()):
                if handler_id not in self._handlers and weak_handlers.get(handler_id) is not None:
                    count += 1

        return count

    async def close(self) -> None:
        """Close the manager and cleanup resources."""
        self.clear()


# Type Aliases
# ------------

# Observer types
ObserverFunc = Callable[["Event"], None]
AsyncObserverFunc = Callable[["Event"], Any]  # Returns awaitable

# Control handler types
ControlHandlerFunc = Callable[["Event", ExecutionContext], ControlResponse]
AsyncControlHandlerFunc = Callable[["Event", ExecutionContext], Any]  # Returns awaitable


# Event Filtering Mixin
# ---------------------


class EventFilterMixin:
    """Mixin for event type filtering logic.

    Provides common filtering functionality for both ObserverManager
    and ControlManager to avoid code duplication.
    """

    def _should_process_event(self, event_filter: set[type] | None, event: "Event") -> bool:
        """Check if an event should be processed based on type filter.

        Args
        ----
            event_filter: Set of event types to accept, or None for all
            event: The event to check

        Returns
        -------
        bool
            True if event should be processed, False otherwise
        """
        # None means accept all events
        if event_filter is None:
            return True

        # Check if event type is in the filter
        return type(event) in event_filter


# Error Handling
# --------------


class ErrorHandler(Protocol):
    """Protocol for handling errors in event system."""

    def handle_error(self, error: Exception, context: dict[str, Any]) -> None:
        """Handle an error that occurred during event processing.

        Args
        ----
            error: The exception that occurred
            context: Additional context about where/when the error occurred
        """
        ...


class LoggingErrorHandler:
    """Default error handler that logs errors."""

    def __init__(self, logger: Any | None = None):
        """Initialize with optional logger.

        Args
        ----
            logger: Logger instance, or None to use default
        """
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def handle_error(self, error: Exception, context: dict[str, Any]) -> None:
        """Log the error with context."""
        handler_name = context.get("handler_name", "unknown")
        event_type = context.get("event_type", "unknown")

        if context.get("is_critical", False):
            self.logger.error(
                "Critical handler %s failed for %s: %s",
                handler_name,
                event_type,
                error,
                exc_info=True,
            )
        else:
            self.logger.warning("Handler %s failed for %s: %s", handler_name, event_type, error)
