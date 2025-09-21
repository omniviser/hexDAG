"""Local Observer Manager Adapter - Delegates to the concrete implementation.

This adapter wraps the concrete ObserverManager implementation to satisfy
the ObserverManagerPort interface, maintaining all safety features.
"""

from typing import Any

from hexai.core.application.events.events import Event
from hexai.core.application.events.models import (
    AsyncObserverFunc,
    ErrorHandler,
    Observer,
    ObserverFunc,
)
from hexai.core.application.events.observer_manager import ObserverManager
from hexai.core.ports.observer_manager import ObserverManagerPort
from hexai.core.registry import adapter


@adapter(implements_port=ObserverManagerPort, namespace="core")
class LocalObserverManager:
    """Local adapter that delegates to the concrete ObserverManager.

    This adapter maintains all safety features and configuration options
    from the original implementation while conforming to the port interface.
    """

    def __init__(
        self,
        max_concurrent_observers: int = 10,
        observer_timeout: float = 5.0,
        max_sync_workers: int = 4,
        error_handler: ErrorHandler | None = None,
        use_weak_refs: bool = True,
    ) -> None:
        """Initialize the local observer manager adapter.

        Args
        ----
            max_concurrent_observers: Maximum number of observers to run concurrently
            observer_timeout: Timeout in seconds for each observer
            max_sync_workers: Maximum thread pool workers for sync observers
            error_handler: Optional error handler, defaults to LoggingErrorHandler
            use_weak_refs: If True, use weak references to prevent memory leaks
        """
        # Create the internal ObserverManager with all passed parameters
        self._inner = ObserverManager(
            max_concurrent_observers=max_concurrent_observers,
            observer_timeout=observer_timeout,
            max_sync_workers=max_sync_workers,
            error_handler=error_handler,
            use_weak_refs=use_weak_refs,
        )

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
        return self._inner.register(handler, **kwargs)

    def unregister(self, handler_id: str) -> bool:
        """Unregister an observer by ID.

        Args
        ----
            handler_id: The ID of the observer to unregister

        Returns
        -------
            bool: True if observer was found and removed, False otherwise
        """
        return self._inner.unregister(handler_id)

    async def notify(self, event: Event) -> None:
        """Notify all interested observers of an event.

        Only observers registered for this event type will be notified.
        Errors are handled according to the configured error handler
        but don't affect execution.

        Args
        ----
            event: The event to distribute to observers
        """
        await self._inner.notify(event)

    def clear(self) -> None:
        """Remove all registered observers."""
        self._inner.clear()

    async def close(self) -> None:
        """Close the manager and cleanup resources."""
        await self._inner.close()

    def __len__(self) -> int:
        """Return number of registered observers.

        Returns
        -------
            int: Count of active observers (including weak refs that are still alive)
        """
        return len(self._inner)

    def __enter__(self) -> "LocalObserverManager":
        """Context manager entry.

        Returns
        -------
            Self after entering the inner manager's context
        """
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with cleanup.

        Args
        ----
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        self._inner.__exit__(exc_type, exc_val, exc_tb)

    async def __aenter__(self) -> "LocalObserverManager":
        """Async context manager entry.

        Returns
        -------
            Self after entering the inner manager's async context
        """
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit with cleanup.

        Args
        ----
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        await self._inner.__aexit__(exc_type, exc_val, exc_tb)
