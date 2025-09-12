"""Event manager for pipeline events."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

from .base import EventType

if TYPE_CHECKING:
    from .base import Observer, PipelineEvent

logger = logging.getLogger(__name__)


class PipelineEventManager:
    """Central event manager for pipeline events using observer pattern."""

    def __init__(self, max_events: int = 10000, enable_caching: bool = True) -> None:
        """Initialize event manager with optimization options.

        Args:
        ----
            max_events: Maximum events to keep in history (for memory management)
            enable_caching: Enable event caching for fast lookups
        """
        self._observers: dict[EventType, list[Observer]] = defaultdict(list)
        self._global_observers: list[Observer] = []
        self._async_queue: asyncio.Queue | None = None
        self._processing_task: asyncio.Task | None = None

        # Optimization features
        self._max_events = max_events
        self._enable_caching = enable_caching

        # Event storage with memory management
        self._events: deque[PipelineEvent] = deque(maxlen=max_events)

        # Caches for O(1) lookups
        if enable_caching:
            self._event_cache_by_type: dict[EventType, deque[PipelineEvent]] = defaultdict(
                lambda: deque(maxlen=max_events // 10)  # Limit per-type cache
            )
            self._event_cache_by_node: dict[str, deque[PipelineEvent]] = defaultdict(
                lambda: deque(maxlen=max_events // 20)  # Limit per-node cache
            )

        # Batch processing
        self._batch_size = 100
        self._pending_batch: list[PipelineEvent] = []

    def subscribe(self, observer: Observer, event_type: EventType | None = None) -> None:
        """Subscribe observer to events."""
        if event_type is None:
            self._global_observers.append(observer)
            logger.debug(f"Registered global observer: {observer.__class__.__name__}")
        else:
            self._observers[event_type].append(observer)
            logger.debug(
                f"Registered observer {observer.__class__.__name__} for {event_type.value}"
            )

    def unsubscribe(self, observer: Observer, event_type: EventType | None = None) -> None:
        """Unsubscribe observer from events."""
        if event_type is None and observer in self._global_observers:
            self._global_observers.remove(observer)
        elif event_type and observer in self._observers[event_type]:
            self._observers[event_type].remove(observer)

    async def emit(self, event: PipelineEvent) -> None:
        """Emit event to all relevant observers with caching."""
        # Store event in history
        self._events.append(event)

        # Update caches for O(1) lookups
        if self._enable_caching:
            self._update_caches(event)

        # Add to batch for processing
        self._pending_batch.append(event)
        if len(self._pending_batch) >= self._batch_size:
            await self._process_batch()

        # Notify observers
        observers = self._global_observers + self._observers[event.event_type]
        observers = [obs for obs in observers if obs.can_handle(event)]

        if observers:
            logger.debug(f"Emitting event: {event.event_type.value} to {len(observers)} observers")
            await asyncio.gather(*[obs.handle(event) for obs in observers], return_exceptions=True)

    def _update_caches(self, event: PipelineEvent) -> None:
        """Update event caches for fast retrieval."""
        # Cache by event type
        self._event_cache_by_type[event.event_type].append(event)

        # Cache by node name if available
        if hasattr(event, "node_name"):
            node_name = getattr(event, "node_name", None)
            if node_name is not None:
                self._event_cache_by_node[node_name].append(event)

    async def _process_batch(self) -> None:
        """Process pending batch of events."""
        # This is where you'd send to external systems, persist, etc.
        # For now, just clear the batch
        self._pending_batch.clear()

    def emit_sync(self, event: PipelineEvent) -> None:
        """Emit event synchronously."""
        try:
            # Try to get the current event loop
            asyncio.get_running_loop()
            # If we're in an async context, schedule the coroutine
            asyncio.create_task(self.emit(event))
            # Don't wait for it to complete to maintain sync behavior
        except RuntimeError:
            # No event loop is running, safe to use asyncio.run
            asyncio.run(self.emit(event))

    async def emit_async(self, event: PipelineEvent) -> None:
        """Add event to async processing queue."""
        if self._async_queue:
            await self._async_queue.put(event)

    async def start_async_processing(self) -> None:
        """Start async event processing."""
        if self._async_queue is None:
            self._async_queue = asyncio.Queue()
            self._processing_task = asyncio.create_task(self._process_events())

    async def stop_async_processing(self) -> None:
        """Stop async event processing."""
        if self._processing_task:
            self._processing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._processing_task
            self._processing_task = None
            self._async_queue = None

    async def _process_events(self) -> None:
        """Process events from async queue."""
        if not self._async_queue:
            return

        while True:
            try:
                event = await self._async_queue.get()
                await self.emit(event)
                self._async_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing event: {e}")

    def get_observer_count(self, event_type: EventType | None = None) -> int:
        """Get total number of observers."""
        if event_type is None:
            return len(self._global_observers) + sum(
                len(obs_list) for obs_list in self._observers.values()
            )
        return len(self._global_observers) + len(self._observers[event_type])

    def clear_observers(self, event_type: EventType | None = None) -> None:
        """Clear observers."""
        if event_type is None:
            self._global_observers.clear()
            self._observers.clear()
        else:
            self._observers[event_type].clear()

    # --- Optimized Event Retrieval Methods ---

    def get_events_by_type(
        self, event_type: EventType, limit: int | None = None
    ) -> list[PipelineEvent]:
        """Get events by type from cache (O(1) lookup).

        Args
        ----
            event_type: Type of events to retrieve
            limit: Maximum number of events to return

        Returns
        -------
            List of events of the specified type
        """
        if not self._enable_caching:
            # Fallback to scanning if caching disabled
            events = [e for e in self._events if e.event_type == event_type]
            return events[:limit] if limit else events

        cached = list(self._event_cache_by_type.get(event_type, []))
        return cached[:limit] if limit else cached

    def get_events_by_node(self, node_name: str, limit: int | None = None) -> list[PipelineEvent]:
        """Get events for a specific node from cache (O(1) lookup).

        Args
        ----
            node_name: Name of the node
            limit: Maximum number of events to return

        Returns
        -------
            List of events for the specified node
        """
        if not self._enable_caching:
            # Fallback to scanning if caching disabled
            events = [
                e
                for e in self._events
                if hasattr(e, "node_name") and getattr(e, "node_name", None) == node_name
            ]
            return events[:limit] if limit else events

        cached = list(self._event_cache_by_node.get(node_name, []))
        return cached[:limit] if limit else cached

    def get_recent_events(self, limit: int = 10) -> list[PipelineEvent]:
        """Get most recent events efficiently.

        Args
        ----
            limit: Number of recent events to return

        Returns
        -------
            List of most recent events
        """
        if len(self._events) <= limit:
            return list(self._events)

        # Use negative indexing for efficient access to recent items
        return list(self._events)[-limit:]

    def get_all_events(self) -> list[PipelineEvent]:
        """Get all events in history.

        Returns
        -------
            List of all events
        """
        return list(self._events)

    def clear_event_history(self) -> None:
        """Clear event history and caches to free memory."""
        self._events.clear()
        if self._enable_caching:
            self._event_cache_by_type.clear()
            self._event_cache_by_node.clear()
        self._pending_batch.clear()

    def get_event_stats(self) -> dict[str, Any]:
        """Get statistics about events.

        Returns
        -------
            Dictionary with event statistics
        """
        stats = {
            "total_events": len(self._events),
            "max_events": self._max_events,
            "caching_enabled": self._enable_caching,
            "pending_batch_size": len(self._pending_batch),
            "events_by_type": {},
        }

        # Count events by type
        for event_type in EventType:
            count = len(self.get_events_by_type(event_type))
            if count > 0:
                stats["events_by_type"][event_type.value] = count  # type: ignore[index]

        return stats
