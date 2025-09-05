"""Base classes for the pipeline event system."""

from __future__ import annotations

import asyncio
import heapq
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of pipeline events."""

    VALIDATION_WARNING = "validation_warning"
    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_COMPLETED = "pipeline_completed"
    PIPELINE_BUILD_STARTED = "pipeline_build_started"
    WAVE_STARTED = "wave_started"
    WAVE_COMPLETED = "wave_completed"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    LLM_PROMPT_GENERATED = "llm_prompt_generated"
    LLM_RESPONSE_RECEIVED = "llm_response_received"
    TOOL_CALLED = "tool_called"
    TOOL_COMPLETED = "tool_completed"


class PipelineEvent:
    """Base class for all pipeline events."""

    def __init__(self) -> None:
        self.event_type: EventType
        self.timestamp: datetime = datetime.now()
        self.session_id: str = ""
        self.metadata: dict[str, Any] = {}

    def __post_init__(self) -> None:
        """Initialize timestamp and metadata if not provided."""
        if not hasattr(self, "timestamp") or self.timestamp is None:
            self.timestamp = datetime.now()
        if not hasattr(self, "metadata") or self.metadata is None:
            self.metadata = {}


class Observer(ABC):
    """Abstract base class for event observers."""

    @abstractmethod
    def can_handle(self, event: PipelineEvent) -> bool:
        """Check if this observer can handle the given event type."""

    @abstractmethod
    async def handle(self, event: PipelineEvent) -> None:
        """Handle the event asynchronously."""


class SyncObserver(Observer):
    """Base for synchronous observers."""

    async def handle(self, event: PipelineEvent) -> None:
        """Handle event by delegating to sync method."""
        await asyncio.to_thread(self.handle_sync, event)

    @abstractmethod
    def handle_sync(self, event: PipelineEvent) -> None:
        """Handle event synchronously."""

    def can_handle(self, event: PipelineEvent) -> bool:
        """Check if this observer can handle the event (default: all events)."""
        return True


HandlerFunction = Callable[[PipelineEvent], Awaitable[Any]]


@dataclass
class PrioritizedHandler:
    """Handler with priority for sorting in event queues."""

    priority: int
    handler: Any
    event_filter: Any = None
    name: str = ""
    filter_func: Callable[[PipelineEvent], bool] | None = None

    def __lt__(self, other: "PrioritizedHandler") -> bool:
        """Compare handlers by priority (lower number = higher priority)."""
        return self.priority < other.priority

    def matches_event(self, event: PipelineEvent) -> bool:
        """Check if this handler should process the given event."""
        # First check event_filter (event type or None for all events)
        if self.event_filter is not None and event.event_type != self.event_filter:
            return False

        # Then check custom filter function
        if self.filter_func is not None:
            return self.filter_func(event)

        return True


class BasePriorityEventDispatcher:
    """Base class for priority-based event dispatching.

    Provides common functionality for both EventBus (control flow)
    and PipelineEventManager (observability).
    """

    def __init__(self) -> None:
        # Priority queue of handlers
        self._handlers: dict[Any, list[PrioritizedHandler]] = defaultdict(list)
        self._global_handlers: list[PrioritizedHandler] = []
        # Initialize lock as None, will be created on first async use
        self._lock: asyncio.Lock | None = None

    async def _ensure_lock(self) -> None:
        """Ensure lock is initialized (must be called from async context)."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        # After this method, self._lock is guaranteed to be non-None

    def _add_handler(
        self,
        handler: Any,
        event_filter: Any = None,
        priority: int = 0,
        name: str = "",
        filter_func: Callable[[PipelineEvent], bool] | None = None,
    ) -> None:
        """Add handler to appropriate priority queue.

        Args
        ----
        handler : Any
            The handler object or function
        event_filter : Any, optional
            Event type filter, None for all events
        priority : int, optional
            Handler priority (lower = higher priority)
        name : str, optional
            Handler name for debugging
        filter_func : callable, optional
            Custom filter function for events
        """
        prioritized = PrioritizedHandler(
            priority=priority,
            handler=handler,
            event_filter=event_filter,
            name=name,
            filter_func=filter_func,
        )

        if event_filter is None:
            heapq.heappush(self._global_handlers, prioritized)
        else:
            heapq.heappush(self._handlers[event_filter], prioritized)

    def _remove_handler(self, handler_name: str) -> None:
        """Remove handler by name from all queues.

        Args
        ----
        handler_name : str
            Name of handler to remove
        """
        # Remove from global handlers
        self._global_handlers = [h for h in self._global_handlers if h.name != handler_name]
        heapq.heapify(self._global_handlers)

        # Remove from event-specific handlers
        for event_type, handlers in self._handlers.items():
            self._handlers[event_type] = [h for h in handlers if h.name != handler_name]
            heapq.heapify(self._handlers[event_type])

    def _get_applicable_handlers(self, event: PipelineEvent) -> list[PrioritizedHandler]:
        """Get all handlers that should process this event, sorted by priority.

        Args
        ----
        event : PipelineEvent
            The event to process

        Returns
        -------
        list[PrioritizedHandler]
            Sorted list of applicable handlers
        """
        handlers: list[PrioritizedHandler] = []

        # Add global handlers
        handlers.extend(h for h in self._global_handlers if h.matches_event(event))

        # Add event-specific handlers
        if event.event_type in self._handlers:
            handlers.extend(h for h in self._handlers[event.event_type] if h.matches_event(event))

        # Sort by priority
        return sorted(handlers, key=lambda h: h.priority)

    def clear_handlers(self) -> None:
        """Clear all handlers.

        Note
        ----
        This method clears all registered handlers from the dispatcher.
        """
        self._global_handlers.clear()
        self._handlers.clear()

    def get_handler_stats(self) -> dict[str, Any]:
        """Get statistics about registered handlers.

        Returns
        -------
        dict[str, Any]
            Statistics including total handlers, global handlers, and event-specific handlers
        """
        return {
            "total_global_handlers": len(self._global_handlers),
            "total_event_handlers": sum(len(handlers) for handlers in self._handlers.values()),
            "event_types_with_handlers": len(self._handlers),
        }
