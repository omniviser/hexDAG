"""Observer Manager: Manages event distribution to observers.

This is a hexagonal architecture PORT that allows external systems
to observe pipeline execution without affecting it.
"""

from __future__ import annotations

import asyncio
import uuid
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .decorators import (
    EVENT_METADATA_ATTR,
    EventDecoratorMetadata,
    EventType,
    EventTypesInput,
    normalize_event_types,
)
from .models import (
    AsyncObserverFunc,
    BaseEventManager,
    ErrorHandler,
    EventFilterMixin,
    LoggingErrorHandler,
    Observer,
    ObserverFunc,
)

if TYPE_CHECKING:
    from .events import Event


type ObserverId = str
type EventFilter = set[EventType] | None


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

    @staticmethod
    def _get_observer_metadata(handler: Any) -> EventDecoratorMetadata | None:
        """Return observer metadata if present."""
        metadata = getattr(handler, EVENT_METADATA_ATTR, None)
        if isinstance(metadata, EventDecoratorMetadata) and metadata.kind == "observer":
            return metadata
        return None

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

        # Track which event types each observer wants (for quick filtering)
        self._event_filters: dict[ObserverId, EventFilter] = {}
        self._observer_timeouts: dict[ObserverId, float | None] = {}
        self._observer_semaphores: dict[ObserverId, asyncio.Semaphore] = {}
        if use_weak_refs:
            self._weak_handlers: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
            self._strong_refs: dict[ObserverId, Observer] = {}

    def register(
        self,
        handler: Observer | ObserverFunc | AsyncObserverFunc,
        *,
        observer_id: str | None = None,
        event_types: EventTypesInput = None,
        timeout: float | None = None,
        max_concurrency: int | None = None,
        keep_alive: bool = False,
        **extra: Any,
    ) -> str:
        """Register an observer with optional event type filtering."""
        if extra:
            unexpected = ", ".join(sorted(extra))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        metadata = self._get_observer_metadata(handler)

        resolved_observer_id = (
            observer_id
            if observer_id is not None
            else (metadata.id if metadata and metadata.id else None)
        )
        raw_event_types: EventTypesInput = (
            event_types if event_types is not None else (metadata.event_types if metadata else None)
        )
        normalized_event_types = (
            normalize_event_types(raw_event_types) if raw_event_types is not None else None
        )
        resolved_timeout = (
            timeout if timeout is not None else (metadata.timeout if metadata else None)
        )
        resolved_concurrency = (
            max_concurrency
            if max_concurrency is not None
            else (metadata.max_concurrency if metadata else None)
        )

        config = ObserverRegistrationConfig(
            observer_id=resolved_observer_id,
            event_types=normalized_event_types,
            timeout=resolved_timeout,
            max_concurrency=resolved_concurrency,
            keep_alive=keep_alive,
        )

        resolved_id = config.observer_id or str(uuid.uuid4())

        if resolved_id in self._event_filters:
            raise ValueError(f"Observer '{resolved_id}' already registered")

        event_filter = config.event_types
        timeout_value = config.timeout
        per_observer_concurrency = config.max_concurrency
        keep_alive_flag = config.keep_alive

        if hasattr(handler, "handle"):
            observer: Observer = handler  # pyright: ignore[reportAssignmentType]
        elif callable(handler):
            observer = FunctionObserver(handler, self._executor)  # pyright: ignore[reportArgumentType]
            keep_alive_flag = True
        else:
            raise TypeError(
                f"Observer must be callable or implement Observer protocol, got {type(handler)}"
            )

        self._event_filters[resolved_id] = event_filter

        if timeout_value is not None:
            self._observer_timeouts[resolved_id] = timeout_value
        elif resolved_id in self._observer_timeouts:
            del self._observer_timeouts[resolved_id]

        if per_observer_concurrency is not None:
            self._observer_semaphores[resolved_id] = asyncio.Semaphore(per_observer_concurrency)
        elif resolved_id in self._observer_semaphores:
            del self._observer_semaphores[resolved_id]

        if self._use_weak_refs:
            try:
                self._weak_handlers[resolved_id] = observer
                if keep_alive_flag:
                    self._strong_refs[resolved_id] = observer
            except TypeError:
                self._handlers[resolved_id] = observer
                self._strong_refs[resolved_id] = observer
        else:
            self._handlers[resolved_id] = observer

        return str(resolved_id)

    def _should_notify(self, observer_id: str, event: Event) -> bool:
        """Check if observer should be notified of this event type.

        Returns
        -------
        bool
            True if observer should be notified of this event
        """
        event_filter = self._event_filters.get(observer_id)
        return self._should_process_event(event_filter, event)

    async def notify(self, event: Event) -> None:
        """Notify observers interested in ``event``.

        Args
        ----
            event: Event instance to broadcast to observers.

        Observers are pulled from both strong and weak-reference stores,
        filtered by event type, and executed with the configured concurrency
        and timeout limits. Errors are logged and do not affect pipeline flow.
        """
        # Check both strong and weak handlers; exit early when nothing is registered
        if not self._handlers and (
            not self._use_weak_refs or not getattr(self, "_weak_handlers", None)
        ):
            return

        # Gather handlers from both storage collections
        all_observers: dict[str, Observer] = {}

        if self._use_weak_refs:
            # Get observers from weak references (auto-cleaned)
            for obs_id in list(self._weak_handlers.keys()):
                observer = self._weak_handlers.get(obs_id)
                if observer is not None:
                    all_observers[obs_id] = observer

        # Add strong references
        all_observers.update(self._handlers)

        # Filter by event type before scheduling work
        interested = [
            (obs_id, observer)
            for obs_id, observer in all_observers.items()
            if self._should_notify(obs_id, event)
        ]

        if not interested:
            return

        # Schedule interested observers with per-manager/per-observer concurrency limits
        tasks = [self._limited_invoke(obs_id, observer, event) for obs_id, observer in interested]

        # Total timeout = base timeout + small buffer for concurrency
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

    async def _limited_invoke(self, observer_id: str, observer: Observer, event: Event) -> None:
        """Invoke an observer respecting global and per-observer limits."""
        per_observer = self._observer_semaphores.get(observer_id)
        if per_observer is None:
            async with self._semaphore:
                await self._safe_invoke(observer_id, observer, event)
        else:
            async with self._semaphore, per_observer:
                await self._safe_invoke(observer_id, observer, event)

    async def _safe_invoke(self, observer_id: str, observer: Observer, event: Event) -> None:
        """Safely invoke an observer while enforcing configured timeouts."""
        timeout = self._observer_timeouts.get(observer_id, self._timeout)
        try:
            if timeout is None:
                await observer.handle(event)
            else:
                await asyncio.wait_for(observer.handle(event), timeout=timeout)
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
        bool
            True if observer was found and removed, False otherwise
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

        if handler_id in self._observer_timeouts:
            del self._observer_timeouts[handler_id]

        if handler_id in self._observer_semaphores:
            del self._observer_semaphores[handler_id]

        return found

    def clear(self) -> None:
        """Remove all registered observers and reset runtime caches."""
        super().clear()
        self._event_filters.clear()
        self._observer_timeouts.clear()
        self._observer_semaphores.clear()
        if self._use_weak_refs:
            self._weak_handlers.clear()
            self._strong_refs.clear()

    async def close(self) -> None:
        """Close the manager and cleanup resources."""
        await super().close()
        if not self._executor_shutdown:
            self._executor.shutdown(wait=False)
            self._executor_shutdown = True

    def __enter__(self) -> ObserverManager:
        """Context manager entry.

        Returns
        -------
        ObserverManager
            This manager instance
        """
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        if not self._executor_shutdown:
            self._executor.shutdown(wait=True)
            self._executor_shutdown = True

    async def __aenter__(self) -> ObserverManager:
        """Async context manager entry.

        Returns
        -------
        ObserverManager
            This manager instance
        """
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

    async def handle(self, event: Event) -> None:
        """Handle the event by calling the wrapped function."""
        if asyncio.iscoroutinefunction(self._func):
            await self._func(event)
        else:
            # Run sync function in thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._func, event)
