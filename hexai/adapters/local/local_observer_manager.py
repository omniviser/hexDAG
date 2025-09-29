"""Local Observer Manager Adapter - Standalone implementation of observer pattern.

This adapter provides a complete, standalone implementation of the ObserverManagerPort
interface with all safety features including weak references, event filtering,
concurrency control, and fault isolation.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import asyncio
import logging
import uuid
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hexai.core.application.events.decorators import (
    EVENT_METADATA_ATTR,
    EventDecoratorMetadata,
    EventType,
    EventTypesInput,
    normalize_event_types,
)
from hexai.core.ports.observer_manager import (
    AsyncObserverFunc,
    Observer,
    ObserverFunc,
    ObserverManagerPort,
)
from hexai.core.registry import adapter

if TYPE_CHECKING:
    from hexai.core.application.events.events import Event


class ErrorHandler(Protocol):
    """Protocol for handling errors in event system."""

    def handle_error(self, error: Exception, context: dict[str, Any]) -> None:
        """Handle an error that occurred during event processing."""
        ...


class LoggingErrorHandler:
    """Default error handler that logs errors."""

    def __init__(self, logger: Any | None = None):
        """Initialize with optional logger."""
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

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


class ObserverRegistrationConfig(BaseModel):
    """Validated configuration for observer registration."""

    model_config = ConfigDict(extra="forbid")

    observer_id: str | None = None
    event_types: set[EventType] | None = None
    timeout: float | None = Field(None, gt=0)
    max_concurrency: int | None = Field(None, ge=1)
    keep_alive: bool = False

    @field_validator("event_types", mode="before")
    @classmethod
    def validate_event_types(cls, value: EventTypesInput) -> set[EventType] | None:
        return normalize_event_types(value)


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

    @staticmethod
    def _get_observer_metadata(handler: Any) -> EventDecoratorMetadata | None:
        """Return observer metadata if present."""
        metadata = getattr(handler, EVENT_METADATA_ATTR, None)
        if isinstance(metadata, EventDecoratorMetadata) and metadata.kind == "observer":
            return metadata
        return None

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

        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent_observers)

        # Thread pool for sync functions
        self._executor = ThreadPoolExecutor(max_workers=max_sync_workers)
        self._executor_shutdown = False

        # Storage for observers (can be Observer or FunctionObserver)
        self._handlers: dict[str, Any] = {}

        # Event type filtering and per-observer config
        self._event_filters: dict[str, set[EventType] | None] = {}
        self._observer_timeouts: dict[str, float | None] = {}
        self._observer_semaphores: dict[str, asyncio.Semaphore] = {}

        # Weak reference support
        if use_weak_refs:
            self._weak_handlers: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
            self._strong_refs: dict[str, Any] = {}  # Keep alive certain observers

    def _store_observer(self, observer_id: str, observer: Any, keep_alive: bool) -> None:
        """Store observer with appropriate reference type."""
        if self._use_weak_refs:
            try:
                # Try to create weak reference
                self._weak_handlers[observer_id] = observer
                # Keep strong ref if requested
                if keep_alive:
                    self._strong_refs[observer_id] = observer
            except TypeError:
                # Some objects can't be weakly referenced - fall back to strong ref
                self._handlers[observer_id] = observer
                self._strong_refs[observer_id] = observer
        else:
            # Normal strong reference when weak refs disabled
            self._handlers[observer_id] = observer

    def register(
        self,
        handler: Observer | ObserverFunc | AsyncObserverFunc,
        *,
        observer_id: str | None = None,
        event_types: EventTypesInput = None,
        timeout: float | None = None,
        max_concurrency: int | None = None,
        keep_alive: bool = False,
    ) -> str:
        """Register an observer with optional event type filtering."""

        metadata = self._get_observer_metadata(handler)

        raw_event_types: EventTypesInput = (
            event_types if event_types is not None else (metadata.event_types if metadata else None)
        )
        normalized_event_types = (
            normalize_event_types(raw_event_types) if raw_event_types is not None else None
        )

        config = ObserverRegistrationConfig(
            observer_id=(
                observer_id
                if observer_id is not None
                else (metadata.id if metadata and metadata.id else None)
            ),
            event_types=normalized_event_types,
            timeout=(timeout if timeout is not None else (metadata.timeout if metadata else None)),
            max_concurrency=(
                max_concurrency
                if max_concurrency is not None
                else (metadata.max_concurrency if metadata else None)
            ),
            keep_alive=keep_alive,
        )

        resolved_id = config.observer_id or str(uuid.uuid4())

        if resolved_id in self._event_filters:
            raise ValueError(f"Observer '{resolved_id}' already registered")

        if resolved_id in self._handlers:
            raise ValueError(f"Observer '{resolved_id}' already registered")

        if (
            self._use_weak_refs
            and hasattr(self, "_weak_handlers")
            and resolved_id in self._weak_handlers
        ):
            raise ValueError(f"Observer '{resolved_id}' already registered")

        keep_alive_flag = config.keep_alive

        if hasattr(handler, "handle"):
            observer = cast("Observer", handler)
        elif callable(handler):
            observer = FunctionObserver(handler, self._executor)
            keep_alive_flag = True
        else:
            raise TypeError(
                f"Observer must be callable or implement Observer protocol, got {type(handler)}"
            )

        self._store_observer(resolved_id, observer, keep_alive_flag)

        self._event_filters[resolved_id] = config.event_types

        if config.timeout is not None:
            self._observer_timeouts[resolved_id] = config.timeout
        else:
            self._observer_timeouts.pop(resolved_id, None)

        if config.max_concurrency is not None:
            self._observer_semaphores[resolved_id] = asyncio.Semaphore(config.max_concurrency)
        else:
            self._observer_semaphores.pop(resolved_id, None)

        return resolved_id

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

        self._observer_timeouts.pop(handler_id, None)
        self._observer_semaphores.pop(handler_id, None)

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
        # Check both strong and weak handlers; exit early when nothing is registered
        if not self._handlers and (
            not self._use_weak_refs or not getattr(self, "_weak_handlers", None)
        ):
            return

        to_check: dict[str, Observer] = {}

        if self._use_weak_refs:
            for obs_id in list(self._weak_handlers.keys()):
                observer = self._weak_handlers.get(obs_id)
                if observer is not None:
                    to_check[obs_id] = observer

        to_check.update(self._handlers)

        interested = [
            (obs_id, observer)
            for obs_id, observer in to_check.items()
            if self._should_notify(obs_id, event)
        ]

        if not interested:
            return

        tasks = [self._limited_invoke(obs_id, observer, event) for obs_id, observer in interested]

        total_timeout = self._timeout + (len(tasks) * DEFAULT_CLEANUP_INTERVAL)

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
        self._observer_timeouts.clear()
        self._observer_semaphores.clear()

        if self._use_weak_refs:
            self._weak_handlers.clear()
            self._strong_refs.clear()

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
            # Count weak handlers that are still alive
            for handler_id in list(self._weak_handlers.keys()):
                if (
                    handler_id not in self._handlers
                    and self._weak_handlers.get(handler_id) is not None
                ):
                    count += 1

        return count

    def __enter__(self) -> LocalObserverManager:
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

    async def __aenter__(self) -> LocalObserverManager:
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

    # Private helper methods

    def _should_notify(self, observer_id: str, event: Event) -> bool:
        """Check if observer should be notified of this event type."""
        event_filter = self._event_filters.get(observer_id)

        # None means accept all events
        if event_filter is None:
            return True

        # Check if event type matches any allowed type (supports subclassing)
        return isinstance(event, tuple(event_filter))

    async def _limited_invoke(self, observer_id: str, observer: Observer, event: Event) -> None:
        """Invoke observer with concurrency limit."""
        per_observer = self._observer_semaphores.get(observer_id)
        if per_observer is None:
            async with self._semaphore:
                await self._safe_invoke(observer_id, observer, event)
        else:
            async with self._semaphore, per_observer:
                await self._safe_invoke(observer_id, observer, event)

    async def _safe_invoke(self, observer_id: str, observer: Observer, event: Event) -> None:
        """Safely invoke an observer with timeout."""
        timeout_value = self._observer_timeouts.get(observer_id, self._timeout)
        try:
            if timeout_value is None:
                await observer.handle(event)
            else:
                await asyncio.wait_for(observer.handle(event), timeout=timeout_value)
        except TimeoutError as exc:
            name = getattr(observer, "__name__", observer.__class__.__name__)
            self._error_handler.handle_error(
                exc,
                {
                    "handler_name": name,
                    "event_type": type(event).__name__,
                    "is_critical": False,
                },
            )
        except Exception as exc:
            name = getattr(observer, "__name__", observer.__class__.__name__)
            self._error_handler.handle_error(
                exc,
                {
                    "handler_name": name,
                    "event_type": type(event).__name__,
                    "is_critical": False,
                },
            )
