"""Basic observer implementations for the event system."""

import asyncio
import logging
from typing import Protocol

from .events import Event


class Observer(Protocol):
    """Protocol for observers that monitor events."""

    async def handle(self, event: Event) -> None:
        """Handle an event (read-only, no return value)."""
        ...


class LoggingObserver:
    """Simple observer that logs events."""

    def __init__(self, logger: logging.Logger | None = None, level: int = logging.INFO):
        """Initialize logging observer.

        Args
        ----
            logger: Logger instance to use (defaults to module logger)
            level: Log level for events (default INFO)
        """
        self.logger = logger or logging.getLogger(__name__)
        self.level = level

    async def handle(self, event: Event) -> None:
        """Log the event using its log_message method."""
        message = event.log_message(self.level)
        self.logger.log(self.level, message)


class CollectingObserver:
    """Observer that collects events for testing or analysis."""

    def __init__(self) -> None:
        """Initialize with empty event list."""
        self.events: list[Event] = []

    async def handle(self, event: Event) -> None:
        """Add event to collection."""
        self.events.append(event)

    def clear(self) -> None:
        """Clear collected events."""
        self.events.clear()

    def get_events_by_type(self, event_type: type[Event]) -> list[Event]:
        """Get all events of a specific type."""
        return [e for e in self.events if isinstance(e, event_type)]


class FilteringObserver:
    """Observer that only handles specific event types."""

    def __init__(self, wrapped: Observer, event_types: set[type[Event]]):
        """Initialize with wrapped observer and event type filter.

        Args
        ----
            wrapped: The observer to delegate to
            event_types: Set of event types to handle
        """
        self.wrapped = wrapped
        self.event_types = event_types

    async def handle(self, event: Event) -> None:
        """Handle event only if it matches the filter."""
        if type(event) in self.event_types:
            await self.wrapped.handle(event)


class CompositeObserver:
    """Observer that delegates to multiple observers."""

    def __init__(self, observers: list[Observer]):
        """Initialize with list of observers."""
        self.observers = observers

    async def handle(self, event: Event) -> None:
        """Notify all observers concurrently."""
        await asyncio.gather(
            *[observer.handle(event) for observer in self.observers],
            return_exceptions=True,  # Don't fail if one observer fails
        )


class NullObserver:
    """Observer that does nothing (null object pattern)."""

    async def handle(self, event: Event) -> None:
        """Do nothing."""
        pass
