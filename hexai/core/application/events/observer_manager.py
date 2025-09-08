"""Observer Manager: Manages event distribution to observers.

This is a hexagonal architecture PORT that allows external systems
to observe pipeline execution without affecting it.
"""

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Type

from .models import AsyncObserverFunc, BaseEventManager, Observer, ObserverFunc

logger = logging.getLogger(__name__)


class ObserverManager(BaseEventManager):
    """Observer Manager for distributing events to observers.

    Key principles:
    - Observers are READ-ONLY - they cannot affect execution
    - Failures in observers don't crash the pipeline (fault isolation)
    - Fire-and-forget pattern (async, non-blocking)
    - Async-first design
    - Event type filtering for efficiency
    - Sync functions run in thread pool to avoid blocking
    """

    def __init__(
        self,
        max_concurrent_observers: int = 10,
        observer_timeout: float = 5.0,
        max_sync_workers: int = 4,
    ) -> None:
        """Initialize the observer manager.

        Args
        ----
            max_concurrent_observers: Maximum number of observers to run concurrently
            observer_timeout: Timeout in seconds for each observer
            max_sync_workers: Maximum thread pool workers for sync observers
        """
        super().__init__()
        self._max_concurrent = max_concurrent_observers
        self._timeout = observer_timeout

        # Create semaphore once, not per event
        self._semaphore = asyncio.Semaphore(max_concurrent_observers)

        # Thread pool for sync functions to avoid blocking
        self._executor = ThreadPoolExecutor(max_workers=max_sync_workers)

        # Track which event types each observer wants
        self._event_filters: dict[str, set[Type] | None] = {}

    def register(self, handler: Any, **kwargs: Any) -> str:
        """Register an observer with optional event type filtering.

        Args
        ----
            handler: Either an Observer protocol implementation or
                    a function (sync/async) that takes an event
            **kwargs: Can include:
                - 'observer_id': Optional ID for the observer
                - 'event_types': List of event types to observe (None = all events)

        Returns
        -------
            str: The ID of the registered observer
        """
        observer_id = kwargs.get("observer_id", str(uuid.uuid4()))
        event_types = kwargs.get("event_types", None)

        # Wrap function if needed
        if hasattr(handler, "handle"):
            # Already implements Observer protocol
            self._handlers[observer_id] = handler
        elif callable(handler):
            # Wrap function to implement the protocol
            self._handlers[observer_id] = FunctionObserver(handler, self._executor)
        else:
            raise TypeError(
                f"Observer must be callable or implement Observer protocol, got {type(handler)}"
            )

        # Store event type filter
        if event_types is not None:
            # Convert to set for O(1) lookup
            self._event_filters[observer_id] = set(event_types)
        else:
            # None means accept all events
            self._event_filters[observer_id] = None

        return str(observer_id)

    def _should_notify(self, observer_id: str, event: Any) -> bool:
        """Check if observer should be notified of this event type."""
        event_filter = self._event_filters.get(observer_id)

        # None means accept all events
        if event_filter is None:
            return True

        # Check if event type is in the filter
        return type(event) in event_filter

    async def notify(self, event: Any) -> None:
        """Notify all interested observers of an event.

        Only observers registered for this event type will be notified.
        Errors are logged but don't affect execution.
        """
        if not self._handlers:
            return

        # Filter observers based on event type
        interested_observers = [
            observer
            for obs_id, observer in self._handlers.items()
            if self._should_notify(obs_id, event)
        ]

        if not interested_observers:
            return

        # Fire only interested observers with limited concurrency
        tasks = [self._limited_invoke(observer, event) for observer in interested_observers]

        # Wait for all with simplified timeout
        # Total timeout = base timeout + buffer for concurrency
        total_timeout = self._timeout + (len(tasks) * 0.1)

        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=total_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Observer notification timed out after {total_timeout}s")

    async def _limited_invoke(self, observer: Observer, event: Any) -> None:
        """Invoke observer with concurrency limit."""
        async with self._semaphore:
            await self._safe_invoke(observer, event)

    async def _safe_invoke(self, observer: Observer, event: Any) -> None:
        """Safely invoke an observer with timeout."""
        try:
            # Apply timeout to individual observer
            await asyncio.wait_for(
                observer.handle(event),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            name = getattr(observer, "__name__", observer.__class__.__name__)
            logger.warning(f"Observer {name} timed out after {self._timeout}s")
        except Exception as e:
            # Get appropriate name for logging
            name = getattr(observer, "__name__", observer.__class__.__name__)
            logger.error(f"Observer {name} failed: {e}")

    def unregister(self, handler_id: str) -> bool:
        """Unregister an observer by ID.

        Args
        ----
            handler_id: The ID of the observer to unregister

        Returns
        -------
            bool: True if observer was found and removed, False otherwise
        """
        if handler_id in self._handlers:
            del self._handlers[handler_id]
            # Also remove event filter
            if handler_id in self._event_filters:
                del self._event_filters[handler_id]
            return True
        return False

    def clear(self) -> None:
        """Remove all registered observers."""
        super().clear()
        self._event_filters.clear()

    async def close(self) -> None:
        """Close the manager and cleanup resources."""
        await super().close()
        self._executor.shutdown(wait=False)

    def __del__(self) -> None:
        """Cleanup executor on deletion."""
        try:
            self._executor.shutdown(wait=False)
        except Exception:  # nosec B110
            # Ignore cleanup errors in destructor - can't handle them meaningfully
            pass


class FunctionObserver:
    """Wrapper to make functions implement the Observer protocol."""

    def __init__(self, func: ObserverFunc | AsyncObserverFunc, executor: ThreadPoolExecutor):
        self._func = func
        self._executor = executor
        self.__name__ = getattr(func, "__name__", "anonymous_observer")

    async def handle(self, event: Any) -> None:
        """Handle the event by calling the wrapped function."""
        if asyncio.iscoroutinefunction(self._func):
            await self._func(event)
        else:
            # Run sync function in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self._func, event)
