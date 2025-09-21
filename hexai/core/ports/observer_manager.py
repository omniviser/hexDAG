"""Observer Manager Port - Clean interface for event observation systems.

This port defines the complete contract for observer managers, including
all safety features and configuration options from the concrete implementation.
"""

from abc import abstractmethod
from typing import Any, Protocol

from hexai.core.application.events.events import Event
from hexai.core.application.events.models import (
    AsyncObserverFunc,
    Observer,
    ObserverFunc,
)
from hexai.core.registry import port


@port(name="observer_manager", namespace="core")
class ObserverManagerPort(Protocol):
    """Port interface for event observation systems.

    Key Safety Guarantees:
    - Observers must be READ-ONLY and cannot affect execution
    - Observer failures must not crash the pipeline (fault isolation)
    - Fire-and-forget pattern with async, non-blocking execution
    - Event type filtering for performance optimization
    - Configurable concurrency control and timeouts
    - Optional weak reference support for memory management
    """

    @abstractmethod
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
        ...

    @abstractmethod
    def unregister(self, handler_id: str) -> bool:
        """Unregister an observer by ID.

        Args
        ----
            handler_id: The ID of the observer to unregister

        Returns
        -------
            bool: True if observer was found and removed, False otherwise
        """
        ...

    @abstractmethod
    async def notify(self, event: Event) -> None:
        """Notify all interested observers of an event.

        Only observers registered for this event type will be notified.
        Errors are handled according to the configured error handler
        but don't affect execution.

        Args
        ----
            event: The event to distribute to observers
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all registered observers."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the manager and cleanup resources."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return number of registered observers."""
        ...

    @abstractmethod
    def __enter__(self) -> "ObserverManagerPort":
        """Context manager entry."""
        ...

    @abstractmethod
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        ...

    @abstractmethod
    async def __aenter__(self) -> "ObserverManagerPort":
        """Async context manager entry."""
        ...

    @abstractmethod
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit with cleanup."""
        ...
