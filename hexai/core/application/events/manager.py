"""Event manager for pipeline events - simplified version."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from .base import BasePriorityEventDispatcher, EventType, Observer, PipelineEvent

logger = logging.getLogger(__name__)


class PipelineEventManager(BasePriorityEventDispatcher):
    """Central event manager for pipeline events using observer pattern.

    This manager is designed for observability - logging, metrics, telemetry,
    and other monitoring tasks that don't affect pipeline execution.

    For control-flow handlers that can modify execution (policies, circuit
    breakers), use EventBus instead.
    """

    def __init__(self) -> None:
        """Initialize event manager."""
        super().__init__()

        # Legacy observers for backward compatibility
        self._observers: dict[EventType, list[Observer]] = defaultdict(list)
        self._global_observers_compat: list[Observer] = []

    def subscribe(
        self, observer: Observer, event_type: EventType | None = None, priority: int = 0
    ) -> None:
        """Subscribe observer to events.

        Parameters
        ----------
        observer : Observer
            Observer instance
        event_type : EventType, optional
            Optional event type to filter (None for all events)
        priority : int, optional
            Optional priority (lower = higher priority, default 0)
        """
        # Store in legacy lists for backward compatibility
        if event_type is None:
            self._global_observers_compat.append(observer)
        else:
            self._observers[event_type].append(observer)

        # Add to base priority system
        self._add_handler(observer, event_type, priority, observer.__class__.__name__)

        logger.debug(
            f"Registered observer {observer.__class__.__name__} "
            f"for {event_type.value if event_type else 'all events'} "
            f"with priority {priority}"
        )

    def unsubscribe(self, observer: Observer, event_type: EventType | None = None) -> None:
        """Unsubscribe observer from events."""
        # Remove from legacy lists
        if event_type is None and observer in self._global_observers_compat:
            self._global_observers_compat.remove(observer)
        elif event_type and observer in self._observers[event_type]:
            self._observers[event_type].remove(observer)

        # Remove from priority system
        self._remove_handler(observer.__class__.__name__)

    async def emit(self, event: PipelineEvent) -> None:
        """Emit event to all relevant observers with error isolation."""
        await self._ensure_lock()

        # Get applicable handlers sorted by priority
        applicable_handlers = self._get_applicable_handlers(event)

        # Filter for observers that can handle this event
        applicable_observers = []
        for prioritized in applicable_handlers:
            observer = prioritized.handler
            if isinstance(observer, Observer):
                if observer.can_handle(event):
                    applicable_observers.append(observer)

        if applicable_observers:
            logger.debug(
                f"Emitting event: {event.event_type.value} to {len(applicable_observers)} observers"
            )

            # Use gather with return_exceptions for error isolation
            results = await asyncio.gather(
                *[obs.handle(event) for obs in applicable_observers], return_exceptions=True
            )

            # Log any observer errors
            for obs, result in zip(applicable_observers, results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Observer {obs.__class__.__name__} failed handling "
                        f"{event.event_type.value}: {result}"
                    )

    async def get_observer_count(self, event_type: EventType | None = None) -> int:
        """Get total number of observers."""
        await self._ensure_lock()
        if self._lock is None:  # Type guard for mypy
            raise RuntimeError("Lock not initialized")

        async with self._lock:
            if event_type is None:
                return len([h for h in self._global_handlers if isinstance(h.handler, Observer)])
            return len(
                [h for h in self._handlers.get(event_type, []) if isinstance(h.handler, Observer)]
            )

    async def clear_observers(self, event_type: EventType | None = None) -> None:
        """Clear observers."""
        await self._ensure_lock()
        if self._lock is None:  # Type guard for mypy
            raise RuntimeError("Lock not initialized")

        async with self._lock:
            if event_type is None:
                self._global_observers_compat.clear()
                self._observers.clear()
                self.clear_handlers()
            else:
                self._observers[event_type].clear()
                # Remove from priority handlers
                self._handlers[event_type] = [
                    h
                    for h in self._handlers.get(event_type, [])
                    if not isinstance(h.handler, Observer)
                ]

    async def get_stats(self) -> dict[str, Any]:
        """Get event manager statistics.

        Returns
        -------
        dict[str, Any]
            Dictionary with stats about observers.
        """
        await self._ensure_lock()
        if self._lock is None:  # Type guard for mypy
            raise RuntimeError("Lock not initialized")

        async with self._lock:
            total_observers = len(
                [
                    h
                    for h in self._global_handlers + sum(self._handlers.values(), [])
                    if isinstance(h.handler, Observer)
                ]
            )
            return {
                "total_observers": total_observers,
                "event_types_monitored": len(self._observers),
                "global_observers": len(self._global_observers_compat),
            }

    # Backward compatibility method (synchronous wrapper)
    def get_observer_count_sync(self, event_type: EventType | None = None) -> int:
        """Return synchronous observer count for backward compatibility."""
        if event_type is None:
            return len(self._global_observers_compat) + sum(
                len(obs_list) for obs_list in self._observers.values()
            )
        return len(self._global_observers_compat) + len(self._observers[event_type])
