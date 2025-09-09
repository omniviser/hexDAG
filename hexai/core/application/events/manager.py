"""Event manager for pipeline events."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import EventType, Observer, PipelineEvent

logger = logging.getLogger(__name__)


class PipelineEventManager:
    """Central event manager for pipeline events using observer pattern."""

    def __init__(self) -> None:
        self._observers: dict[EventType, list[Observer]] = defaultdict(list)
        self._global_observers: list[Observer] = []
        self._async_queue: asyncio.Queue | None = None
        self._processing_task: asyncio.Task | None = None

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
        """Emit event to all relevant observers."""
        observers = self._global_observers + self._observers[event.event_type]
        observers = [obs for obs in observers if obs.can_handle(event)]

        if observers:
            logger.debug(f"Emitting event: {event.event_type.value} to {len(observers)} observers")
            await asyncio.gather(*[obs.handle(event) for obs in observers], return_exceptions=True)

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
