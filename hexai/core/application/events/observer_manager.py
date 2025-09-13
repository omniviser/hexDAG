"""Observer Manager: Manages event distribution to observers.

This is a hexagonal architecture PORT that allows external systems
to observe pipeline execution without affecting it.
"""

import asyncio
import uuid
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .events import Event

from .models import (
    AsyncObserverFunc,
    BaseEventManager,
    ErrorHandler,
    EventFilterMixin,
    LoggingErrorHandler,
    Observer,
    ObserverFunc,
)


class ObserverManager(BaseEventManager, EventFilterMixin):
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
        error_handler: ErrorHandler | None = None,
        use_weak_refs: bool = True,
    ) -> None:
        """Initialize the observer manager.

        Args
        ----
            max_concurrent_observers: Maximum number of observers to run concurrently
            observer_timeout: Timeout in seconds for each observer
            max_sync_workers: Maximum thread pool workers for sync observers
            error_handler: Optional error handler, defaults to LoggingErrorHandler
            use_weak_refs: If True, use weak references to prevent memory leaks
        """
        super().__init__()
        self._max_concurrent = max_concurrent_observers
        self._timeout = observer_timeout
        self._error_handler = error_handler or LoggingErrorHandler()
        self._use_weak_refs = use_weak_refs

        # Create semaphore once, not per event
        self._semaphore = asyncio.Semaphore(max_concurrent_observers)

        # Thread pool for sync functions to avoid blocking
        self._executor = ThreadPoolExecutor(max_workers=max_sync_workers)
        self._executor_shutdown = False

        # Track which event types each observer wants
        self._event_filters: dict[str, set[type] | None] = {}

        # Use WeakValueDictionary for automatic cleanup if enabled
        # Store strong references only for wrapped functions that need to be kept alive
        if use_weak_refs:
            self._weak_handlers: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
            self._strong_refs: dict[str, Any] = {}  # Keep functions alive

    def register(self, handler: Observer | ObserverFunc | AsyncObserverFunc, **kwargs: Any) -> str:
        """Register an observer with optional event type filtering.

        Args
        ----
            handler: Either an Observer protocol implementation or
                    a function (sync/async) that takes an event
            **kwargs: Can include:
                - 'observer_id': Optional ID for the observer
                - 'event_types': List of event types to observe (None = all events)
                - 'keep_alive': Whether to keep strong reference (for weak-referenceable objects)

        Returns
        -------
            str: The ID of the registered observer
        """
        observer_id = kwargs.get("observer_id", str(uuid.uuid4()))
        event_types = kwargs.get("event_types")

        # Wrap function if needed
        if hasattr(handler, "handle"):
            # Already implements Observer protocol
            observer = handler
            keep_alive = kwargs.get("keep_alive", False)
        elif callable(handler):
            # Wrap function to implement the protocol
            observer = FunctionObserver(handler, self._executor)
            # Functions need to be kept alive since they're wrapped
            keep_alive = True
        else:
            raise TypeError(
                f"Observer must be callable or implement Observer protocol, got {type(handler)}"
            )

        # Store the observer with appropriate reference type
        if self._use_weak_refs:
            try:
                # Try to create weak reference
                self._weak_handlers[observer_id] = observer
                # Keep strong ref if requested or for wrapped functions
                if keep_alive:
                    self._strong_refs[observer_id] = observer
            except TypeError:
                # Some objects can't be weakly referenced (e.g., bound methods)
                # Fall back to strong reference
                self._handlers[observer_id] = observer
                self._strong_refs[observer_id] = observer
        else:
            # Normal strong reference when weak refs disabled
            self._handlers[observer_id] = observer

        # Store event type filter
        if event_types is not None:
            # Convert to set for O(1) lookup
            self._event_filters[observer_id] = set(event_types)
        else:
            # None means accept all events
            self._event_filters[observer_id] = None

        return str(observer_id)

    def _should_notify(self, observer_id: str, event: "Event") -> bool:
        """Check if observer should be notified of this event type."""
        event_filter = self._event_filters.get(observer_id)
        return self._should_process_event(event_filter, event)

    async def notify(self, event: "Event") -> None:
        """Notify all interested observers of an event.

        Only observers registered for this event type will be notified.
        Errors are logged but don't affect execution.
        """
        # Check both strong and weak handlers
        if not self._handlers and (not self._use_weak_refs or not self._weak_handlers):
            return

        # Filter observers based on event type
        # Collect from both weak and strong references
        all_observers = {}

        if self._use_weak_refs:
            # Get observers from weak references (auto-cleaned)
            for obs_id in list(self._weak_handlers.keys()):
                observer = self._weak_handlers.get(obs_id)
                if observer is not None:
                    all_observers[obs_id] = observer

        # Add strong references
        all_observers.update(self._handlers)

        interested_observers = [
            observer
            for obs_id, observer in all_observers.items()
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
        except TimeoutError:
            self._error_handler.handle_error(
                TimeoutError(f"Observer notification timed out after {total_timeout}s"),
                {"event_type": type(event).__name__, "handler_name": "ObserverManager"},
            )

    async def _limited_invoke(self, observer: Observer, event: "Event") -> None:
        """Invoke observer with concurrency limit."""
        async with self._semaphore:
            await self._safe_invoke(observer, event)

    async def _safe_invoke(self, observer: Observer, event: "Event") -> None:
        """Safely invoke an observer with timeout."""
        try:
            # Apply timeout to individual observer
            await asyncio.wait_for(
                observer.handle(event),
                timeout=self._timeout,
            )
        except TimeoutError as e:
            name = getattr(observer, "__name__", observer.__class__.__name__)
            self._error_handler.handle_error(
                e,
                {
                    "handler_name": name,
                    "event_type": type(event).__name__,
                    "is_critical": False,
                },
            )
        except Exception as e:
            # Get appropriate name for logging
            name = getattr(observer, "__name__", observer.__class__.__name__)
            self._error_handler.handle_error(
                e,
                {
                    "handler_name": name,
                    "event_type": type(event).__name__,
                    "is_critical": False,
                },
            )

    def unregister(self, handler_id: str) -> bool:
        """Unregister an observer by ID.

        Args
        ----
            handler_id: The ID of the observer to unregister

        Returns
        -------
            bool: True if observer was found and removed, False otherwise
        """
        found = False

        # Remove from all storage locations
        if handler_id in self._handlers:
            del self._handlers[handler_id]
            found = True

        if self._use_weak_refs:
            if handler_id in self._weak_handlers:
                del self._weak_handlers[handler_id]
                found = True

            if handler_id in self._strong_refs:
                del self._strong_refs[handler_id]
                found = True

        # Also remove event filter
        if handler_id in self._event_filters:
            del self._event_filters[handler_id]

        return found

    def clear(self) -> None:
        """Remove all registered observers."""
        super().clear()
        self._event_filters.clear()
        if self._use_weak_refs:
            self._weak_handlers.clear()
            self._strong_refs.clear()

    async def close(self) -> None:
        """Close the manager and cleanup resources."""
        await super().close()
        if not self._executor_shutdown:
            self._executor.shutdown(wait=False)
            self._executor_shutdown = True

    def __enter__(self) -> "ObserverManager":
        """Context manager entry."""
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        if not self._executor_shutdown:
            self._executor.shutdown(wait=True)
            self._executor_shutdown = True

    async def __aenter__(self) -> "ObserverManager":
        """Async context manager entry."""
        return self

    async def __aexit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Async context manager exit with cleanup."""
        await self.close()


class FunctionObserver:
    """Wrapper to make functions implement the Observer protocol."""

    def __init__(self, func: ObserverFunc | AsyncObserverFunc, executor: ThreadPoolExecutor):
        self._func = func
        self._executor = executor
        self.__name__ = getattr(func, "__name__", "anonymous_observer")

    async def handle(self, event: "Event") -> None:
        """Handle the event by calling the wrapped function."""
        if asyncio.iscoroutinefunction(self._func):
            await self._func(event)
        else:
            # Run sync function in thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._func, event)
