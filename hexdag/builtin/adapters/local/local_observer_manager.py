"""Local Observer Manager Adapter - Standalone implementation of observer pattern.

This adapter provides a complete, standalone implementation of the ObserverManagerPort
interface with all safety features including weak references, event filtering,
concurrency control, and fault isolation.
"""

import asyncio
import uuid
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

from hexdag.core.logging import get_logger
from hexdag.core.orchestration.events.events import Event
from hexdag.core.ports.observer_manager import (
    AsyncObserverFunc,
    Observer,
    ObserverFunc,
    ObserverManagerPort,
)
from hexdag.core.registry import adapter


class ErrorHandler(Protocol):
    """Protocol for handling errors in event system."""

    def handle_error(self, error: Exception, context: dict[str, Any]) -> None:
        """Handle an error that occurred during event processing."""
        ...


class LoggingErrorHandler:
    """Default error handler that logs errors."""

    def __init__(self, logger: Any | None = None):
        """Initialize with optional logger."""
        self.logger: Any = logger if logger is not None else get_logger(__name__)

    def handle_error(self, error: Exception, context: dict[str, Any]) -> None:
        """Log the error with context."""
        handler_name = context.get("handler_name", "unknown")
        event_type = context.get("event_type", "unknown")

        if context.get("is_critical", False):
            self.logger.error(
                f"Critical handler {handler_name} failed for {event_type}: {error}", exc_info=True
            )
        else:
            self.logger.warning(f"Handler {handler_name} failed for {event_type}: {error}")


class FunctionObserver:
    """Wrapper to make functions implement the Observer protocol."""

    def __init__(self, func: ObserverFunc | AsyncObserverFunc, executor: ThreadPoolExecutor):
        self._func = func
        self._executor = executor
        self.__name__ = getattr(func, "__name__", "anonymous_observer")

    async def handle(self, event: Event) -> None:
        """Handle the event by calling the wrapped function."""
        if asyncio.iscoroutinefunction(self._func):
            await self._func(event)
        else:
            # Run sync function in thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._func, event)


# Default configuration constants
DEFAULT_MAX_CONCURRENT_OBSERVERS = 10
DEFAULT_OBSERVER_TIMEOUT = 5.0
DEFAULT_MAX_SYNC_WORKERS = 4
DEFAULT_CLEANUP_INTERVAL = 0.1


@adapter(implements_port=ObserverManagerPort, namespace="core")
class LocalObserverManager:
    """Local standalone implementation of observer manager.

    This implementation provides:
    - Weak reference support to prevent memory leaks
    - Event type filtering for efficiency
    - Concurrent observer execution with limits
    - Fault isolation - observer failures don't crash the pipeline
    - Timeout handling for slow observers
    - Thread pool for sync observers to avoid blocking
    """

    def __init__(
        self,
        max_concurrent_observers: int = DEFAULT_MAX_CONCURRENT_OBSERVERS,
        observer_timeout: float = DEFAULT_OBSERVER_TIMEOUT,
        max_sync_workers: int = DEFAULT_MAX_SYNC_WORKERS,
        error_handler: ErrorHandler | None = None,
        use_weak_refs: bool = True,
    ) -> None:
        """Initialize the local observer manager.

        Args
        ----
            max_concurrent_observers: Maximum number of observers to run concurrently
            observer_timeout: Timeout in seconds for each observer
            max_sync_workers: Maximum thread pool workers for sync observers
            error_handler: Optional error handler, defaults to LoggingErrorHandler
            use_weak_refs: If True, use weak references to prevent memory leaks
        """
        self._max_concurrent = max_concurrent_observers
        self._timeout = observer_timeout
        self._error_handler = error_handler or LoggingErrorHandler()
        self._use_weak_refs = use_weak_refs

        self._semaphore = asyncio.Semaphore(max_concurrent_observers)
        self._executor = ThreadPoolExecutor(max_workers=max_sync_workers)
        self._executor_shutdown = False

        self._handlers: dict[str, Any] = {}
        self._event_filters: dict[str, set[type] | None] = {}

        if use_weak_refs:
            self._weak_handlers: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
            self._strong_refs: dict[str, Any] = {}
            self._observer_refs: dict[str, weakref.ref] = {}

    def _on_observer_deleted(self, observer_id: str) -> None:
        """Cleanup callback when observer is garbage collected."""
        self._event_filters.pop(observer_id, None)

    def _store_observer(self, observer_id: str, observer: Any, keep_alive: bool) -> None:
        """Store observer with appropriate reference type."""
        if self._use_weak_refs:
            try:
                self._weak_handlers[observer_id] = observer
                self._observer_refs[observer_id] = weakref.ref(
                    observer, lambda _: self._on_observer_deleted(observer_id)
                )
                if keep_alive:
                    self._strong_refs[observer_id] = observer
            except TypeError:
                self._handlers[observer_id] = observer
                self._strong_refs[observer_id] = observer
        else:
            self._handlers[observer_id] = observer

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

        Raises
        ------
            TypeError: If handler is not callable or doesn't implement Observer protocol
        """
        observer_id = kwargs.get("observer_id", str(uuid.uuid4()))
        event_types = kwargs.get("event_types")

        if hasattr(handler, "handle"):
            observer = handler
            keep_alive = kwargs.get("keep_alive", False)
        elif callable(handler):
            observer = FunctionObserver(handler, self._executor)
            keep_alive = True
        else:
            raise TypeError(
                f"Observer must be callable or implement Observer protocol, got {type(handler)}"
            )

        self._store_observer(observer_id, observer, keep_alive)

        if event_types is not None:
            self._event_filters[observer_id] = set(event_types)
        else:
            self._event_filters[observer_id] = None

        return str(observer_id)

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

        if handler_id in self._event_filters:
            del self._event_filters[handler_id]

        return found

    async def notify(self, event: Event) -> None:
        """Notify all interested observers of an event.

        Only observers registered for this event type will be notified.
        Errors are handled according to the configured error handler
        but don't affect execution.

        Args
        ----
            event: The event to distribute to observers
        """
        if not self._handlers and (not self._use_weak_refs or not self._weak_handlers):
            return

        # Pre-allocate buffer to avoid repeated list resizing (hot path optimization)
        max_tasks = len(self._handlers)
        if self._use_weak_refs:
            max_tasks += len(self._weak_handlers)

        tasks_buffer: list[Any] = [None] * max_tasks
        task_count = 0

        if self._use_weak_refs:
            for obs_id, observer in self._weak_handlers.items():
                if observer is not None and self._should_notify(obs_id, event):
                    tasks_buffer[task_count] = self._limited_invoke(observer, event)
                    task_count += 1

        for obs_id, observer in self._handlers.items():
            if self._should_notify(obs_id, event):
                tasks_buffer[task_count] = self._limited_invoke(observer, event)
                task_count += 1

        tasks = tasks_buffer[:task_count]

        if not tasks:
            return

        total_timeout = self._timeout + 1.0

        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=total_timeout,
            )
        except TimeoutError:
            self._error_handler.handle_error(
                TimeoutError(f"Observer notification timed out after {total_timeout}s"),
                {"event_type": type(event).__name__, "handler_name": "LocalObserverManager"},
            )

    def clear(self) -> None:
        """Remove all registered observers."""
        self._handlers.clear()
        self._event_filters.clear()

        if self._use_weak_refs:
            self._weak_handlers.clear()
            self._strong_refs.clear()
            self._observer_refs.clear()

    async def close(self) -> None:
        """Close the manager and cleanup resources."""
        self.clear()
        if not self._executor_shutdown:
            self._executor.shutdown(wait=False)
            self._executor_shutdown = True

    def __len__(self) -> int:
        """Return number of registered observers.

        Returns
        -------
            int: Count of active observers (including weak refs that are still alive)
        """
        count = len(self._handlers)

        if self._use_weak_refs:
            for handler_id in list(self._weak_handlers.keys()):
                if (
                    handler_id not in self._handlers
                    and self._weak_handlers.get(handler_id) is not None
                ):
                    count += 1

        return count

    def __enter__(self) -> "LocalObserverManager":
        """Context manager entry.

        Returns
        -------
            Self for use in with statements
        """
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Context manager exit with cleanup.

        Args
        ----
            _exc_type: Exception type if an exception occurred
            _exc_val: Exception value if an exception occurred
            _exc_tb: Exception traceback if an exception occurred
        """
        if not self._executor_shutdown:
            self._executor.shutdown(wait=True)
            self._executor_shutdown = True

    async def __aenter__(self) -> "LocalObserverManager":
        """Async context manager entry.

        Returns
        -------
            Self for use in async with statements
        """
        return self

    async def __aexit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Async context manager exit with cleanup.

        Args
        ----
            _exc_type: Exception type if an exception occurred
            _exc_val: Exception value if an exception occurred
            _exc_tb: Exception traceback if an exception occurred
        """
        await self.close()

    def _should_notify(self, observer_id: str, event: Event) -> bool:
        """Check if observer should be notified of this event type."""
        event_filter = self._event_filters.get(observer_id)

        if event_filter is None:
            return True

        return type(event) in event_filter

    async def _limited_invoke(self, observer: Observer, event: Event) -> None:
        """Invoke observer with concurrency limit."""
        async with self._semaphore:
            await self._safe_invoke(observer, event)

    async def _safe_invoke(self, observer: Observer, event: Event) -> None:
        """Safely invoke an observer with timeout."""
        try:
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
            name = getattr(observer, "__name__", observer.__class__.__name__)
            self._error_handler.handle_error(
                e,
                {
                    "handler_name": name,
                    "event_type": type(event).__name__,
                    "is_critical": False,
                },
            )
