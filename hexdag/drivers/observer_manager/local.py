"""Local Observer Manager Adapter - Standalone implementation of observer pattern.

This adapter provides a complete, standalone implementation of the ObserverManagerPort
interface with all safety features including weak references, event filtering,
concurrency control, and fault isolation.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hexdag.kernel.orchestration.events.batching import (
    BatchingConfig,
    BatchingMetrics,
    EventBatchEnvelope,
    EventBatcher,
)
from hexdag.kernel.orchestration.events.decorators import (
    EVENT_METADATA_ATTR,
    EventDecoratorMetadata,
    EventType,
    EventTypesInput,
    normalize_event_types,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence

    from hexdag.kernel.orchestration.events.events import Event
    from hexdag.kernel.ports.observer_manager import (
        AsyncObserverFunc,
        Observer,
        ObserverFunc,
    )

LOGGER = logging.getLogger(__name__)


class ErrorHandler(Protocol):
    """Protocol for handling errors in event system."""

    def handle_error(self, error: Exception, context: dict[str, Any]) -> None:
        """Handle an error that occurred during event processing."""
        ...


class LoggingErrorHandler:
    """Default error handler that logs errors."""

    def __init__(self, logger: Any | None = None):
        """Initialize with optional logger."""
        self.logger: Any = logger if logger is not None else LOGGER

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
        batching_config: BatchingConfig | None = None,
        batching_enabled: bool = True,
    ) -> None:
        """Initialize the local observer manager.

        Args
        ----
            max_concurrent_observers: Maximum number of observers to run concurrently
            observer_timeout: Timeout in seconds for each observer
            max_sync_workers: Maximum thread pool workers for sync observers
            error_handler: Optional error handler, defaults to LoggingErrorHandler
            use_weak_refs: If True, use weak references to prevent memory leaks
            batching_config: Optional batching configuration
            batching_enabled: Toggle to bypass batching (useful for tests/debug)
        """
        self._max_concurrent = max_concurrent_observers
        self._timeout = observer_timeout
        self._error_handler = error_handler or LoggingErrorHandler()
        self._use_weak_refs = use_weak_refs
        self._batching_config = batching_config or BatchingConfig()

        self._semaphore = asyncio.Semaphore(max_concurrent_observers)
        self._executor = ThreadPoolExecutor(max_workers=max_sync_workers)
        self._executor_shutdown = False

        self._handlers: dict[str, Any] = {}

        # Event type filtering and per-observer config
        self._event_filters: dict[str, set[EventType] | None] = {}
        self._observer_timeouts: dict[str, float | None] = {}
        self._observer_semaphores: dict[str, asyncio.Semaphore] = {}

        if use_weak_refs:
            self._weak_handlers: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
            self._strong_refs: dict[str, Any] = {}
            self._observer_refs: dict[str, weakref.ref] = {}

        self._batcher: EventBatcher[Event] | None = None
        if batching_enabled:
            self._batcher = EventBatcher(
                self._flush_envelope,
                self._batching_config,
                logger=LOGGER,
            )

    def _on_observer_deleted(self, observer_id: str) -> None:
        """Cleanup callback when observer is garbage collected."""
        self._event_filters.pop(observer_id, None)
        self._observer_timeouts.pop(observer_id, None)
        self._observer_semaphores.pop(observer_id, None)
        self._observer_refs.pop(observer_id, None)

    def _store_observer(self, observer_id: str, observer: Any, keep_alive: bool) -> None:
        """Store observer with appropriate reference type."""
        if self._use_weak_refs:
            try:
                self._weak_handlers[observer_id] = observer
                manager_ref = weakref.ref(self)

                def _weak_callback(
                    _: Any, oid: str = observer_id, mref: weakref.ref = manager_ref
                ) -> None:
                    manager = mref()
                    if manager is not None:
                        manager._on_observer_deleted(oid)

                self._observer_refs[observer_id] = weakref.ref(observer, _weak_callback)
                if keep_alive:
                    self._strong_refs[observer_id] = observer
            except TypeError:
                self._handlers[observer_id] = observer
                self._strong_refs[observer_id] = observer
        else:
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
        active_observers = self._collect_active_observers()
        if not active_observers:
            return

        if self._batcher is not None:
            await self._batcher.add(event)
            return

        observers = self._collect_interested_observers(event, active_observers)
        if not observers:
            return

        tasks = [
            self._dispatch_events(observer_id, observer, (event,))
            for observer_id, observer in observers.items()
        ]

        await self._run_with_timeout(
            tasks,
            type(event).__name__,
            is_critical=self._is_priority_event(event),
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
            self._observer_refs.clear()

    async def close(self) -> None:
        """Close the manager and cleanup resources."""
        if self._batcher is not None:
            await self._batcher.close()

        self.clear()
        if not self._executor_shutdown:
            self._executor.shutdown(wait=True)
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

    @property
    def batching_metrics(self) -> BatchingMetrics | None:
        """Expose batching metrics when batching is enabled."""

        if self._batcher is None:
            return None
        return self._batcher.metrics

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

    async def _flush_envelope(self, envelope: EventBatchEnvelope) -> None:
        """Flush a prepared event envelope to interested observers."""

        active_observers = self._collect_active_observers()
        if not active_observers:
            return

        tasks = []
        batch_contains_priority = False

        for observer_id, observer in active_observers.items():
            events = self._filter_events_for_observer(observer_id, envelope.events)
            if not events:
                continue

            if any(self._is_priority_event(event) for event in events):
                batch_contains_priority = True

            batch_envelope = None
            if self._supports_batch(observer):
                batch_envelope = self._make_filtered_envelope(envelope, events)

            tasks.append(self._dispatch_events(observer_id, observer, events, batch_envelope))

        if not tasks:
            return

        await self._run_with_timeout(
            tasks,
            f"batch:{envelope.flush_reason.value}",
            is_critical=batch_contains_priority,
        )

    def _collect_active_observers(self) -> dict[str, Observer]:
        """Collect currently active observers, pruning dead weak references."""

        observers: dict[str, Observer] = dict(self._handlers)

        if self._use_weak_refs:
            for observer_id in list(self._weak_handlers.keys()):
                observer = self._weak_handlers.get(observer_id)
                if observer is None:
                    self._weak_handlers.pop(observer_id, None)
                    self._event_filters.pop(observer_id, None)
                    self._strong_refs.pop(observer_id, None)
                    continue
                observers.setdefault(observer_id, observer)

        return observers

    def _collect_interested_observers(
        self, event: Event, observers: dict[str, Observer]
    ) -> dict[str, Observer]:
        """Filter observers down to those interested in the given event."""

        return {
            observer_id: observer
            for observer_id, observer in observers.items()
            if self._should_notify(observer_id, event)
        }

    async def _dispatch_events(
        self,
        observer_id: str,
        observer: Observer,
        events: Sequence[Event],
        envelope: EventBatchEnvelope | None = None,
    ) -> None:
        """Dispatch one or more events to an observer under concurrency control."""

        per_observer_semaphore = self._observer_semaphores.get(observer_id)

        async def _execute() -> None:
            if envelope is not None and self._supports_batch(observer):
                await self._safe_invoke_batch(observer_id, observer, envelope, events)
            else:
                for event in events:
                    await self._safe_invoke(observer_id, observer, event)

        if per_observer_semaphore is None:
            async with self._semaphore:
                await _execute()
            return

        async with self._semaphore, per_observer_semaphore:
            await _execute()

    def _filter_events_for_observer(
        self, observer_id: str, events: Sequence[Event]
    ) -> tuple[Event, ...]:
        """Return events from the batch that match the observer's filter."""

        event_filter = self._event_filters.get(observer_id)
        if event_filter is None:
            return tuple(events)

        return tuple(event for event in events if type(event) in event_filter)

    def _make_filtered_envelope(
        self, envelope: EventBatchEnvelope, events: Sequence[Event]
    ) -> EventBatchEnvelope:
        """Create envelope tailored to an observer's filtered events."""

        if len(events) == len(envelope.events):
            return envelope

        return EventBatchEnvelope(
            batch_id=envelope.batch_id,
            sequence_no=envelope.sequence_no,
            created_at=envelope.created_at,
            events=tuple(events),
            flush_reason=envelope.flush_reason,
        )

    def _supports_batch(self, observer: Observer) -> bool:
        """Check whether observer exposes a batch handler."""

        handler = getattr(observer, "handle_batch", None)
        return callable(handler)

    async def _run_with_timeout(
        self,
        tasks: Sequence[Awaitable[Any]],
        context_label: str,
        *,
        is_critical: bool = False,
    ) -> None:
        """Run observer tasks enforcing a global timeout."""

        if not tasks:
            return

        total_timeout = self._timeout + (len(tasks) * DEFAULT_CLEANUP_INTERVAL)

        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=total_timeout,
            )
        except TimeoutError:
            self._error_handler.handle_error(
                TimeoutError(f"Observer notification timed out after {total_timeout}s"),
                {
                    "event_type": context_label,
                    "handler_name": "LocalObserverManager",
                    "is_critical": is_critical,
                },
            )

    def _should_notify(self, observer_id: str, event: Event) -> bool:
        """Check if observer should be notified of this event type."""
        event_filter = self._event_filters.get(observer_id)

        if event_filter is None:
            return True

        # Check if event type matches any allowed type (supports subclassing)
        return isinstance(event, tuple(event_filter))

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
                    "is_critical": self._is_priority_event(event),
                },
            )
        except Exception as exc:
            name = getattr(observer, "__name__", observer.__class__.__name__)
            self._error_handler.handle_error(
                exc,
                {
                    "handler_name": name,
                    "event_type": type(event).__name__,
                    "is_critical": self._is_priority_event(event),
                },
            )

    async def _safe_invoke_batch(
        self,
        observer_id: str,
        observer: Observer,
        envelope: EventBatchEnvelope,
        events: Sequence[Event],
    ) -> None:
        """Safely invoke an observer's batch handler."""

        handler = getattr(observer, "handle_batch", None)
        if handler is None:
            for event in events:
                await self._safe_invoke(observer_id, observer, event)
            return

        timeout_value = self._observer_timeouts.get(observer_id, self._timeout)

        loop = asyncio.get_running_loop()

        try:
            if inspect.iscoroutinefunction(handler):
                coroutine = handler(envelope)
                if timeout_value is None:
                    await coroutine
                else:
                    await asyncio.wait_for(coroutine, timeout=timeout_value)
            else:
                task = loop.run_in_executor(self._executor, handler, envelope)
                if timeout_value is None:
                    await task
                else:
                    await asyncio.wait_for(task, timeout=timeout_value)
        except TimeoutError as e:
            name = getattr(observer, "__name__", observer.__class__.__name__)
            self._error_handler.handle_error(
                e,
                {
                    "handler_name": name,
                    "event_type": f"batch:{envelope.flush_reason.value}",
                    "is_critical": any(self._is_priority_event(event) for event in events),
                },
            )
        except Exception as e:
            name = getattr(observer, "__name__", observer.__class__.__name__)
            self._error_handler.handle_error(
                e,
                {
                    "handler_name": name,
                    "event_type": f"batch:{envelope.flush_reason.value}",
                    "is_critical": any(self._is_priority_event(event) for event in events),
                },
            )

    def _is_priority_event(self, event: Event) -> bool:
        """Check if event is considered priority/critical."""

        return isinstance(event, self._batching_config.priority_event_types)
