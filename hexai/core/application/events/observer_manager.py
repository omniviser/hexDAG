"""Observer Manager: Manages event distribution to observers.

This is a hexagonal architecture PORT that allows external systems
to observe pipeline execution without affecting it.
"""

import asyncio
import logging
import uuid
from typing import Any

from .models import AsyncObserverFunc, BaseEventManager, Observer, ObserverFunc

logger = logging.getLogger(__name__)


class ObserverManager(BaseEventManager):
    """Observer Manager for distributing events to observers.

    Key principles:
    - Observers are READ-ONLY - they cannot affect execution
    - Failures in observers don't crash the pipeline (fault isolation)
    - Fire-and-forget pattern (async, non-blocking)
    - Async-first design
    """

    def __init__(
        self,
        max_concurrent_observers: int = 10,
        observer_timeout: float = 5.0,
    ) -> None:
        """Initialize the observer manager.

        Args
        ----
            max_concurrent_observers: Maximum number of observers to run concurrently
            observer_timeout: Timeout in seconds for each observer
        """
        super().__init__()
        self._max_concurrent = max_concurrent_observers
        self._timeout = observer_timeout

    def register(self, handler: Any, **kwargs: Any) -> str:
        """Register an observer with optional ID.

        Args
        ----
            handler: Either an Observer protocol implementation or
                    a function (sync/async) that takes an event
            **kwargs: Can include 'observer_id' for the optional ID

        Returns
        -------
            str: The ID of the registered observer
        """
        observer_id = kwargs.get("observer_id", str(uuid.uuid4()))

        # Wrap function if needed
        if hasattr(handler, "handle"):
            # Already implements Observer protocol
            self._handlers[observer_id] = handler
        elif callable(handler):
            # Wrap function to implement the protocol
            self._handlers[observer_id] = FunctionObserver(handler)
        else:
            raise TypeError(
                f"Observer must be callable or implement Observer protocol, got {type(handler)}"
            )

        return str(observer_id)

    async def notify(self, event: Any) -> None:
        """Notify all observers of an event.

        Errors are logged but don't affect execution.
        """
        if not self._handlers:
            return

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self._max_concurrent)

        async def limited_invoke(observer: Observer) -> None:
            async with semaphore:
                await self._safe_invoke(observer, event)

        # Fire all observers with limited concurrency
        tasks = [limited_invoke(observer) for observer in self._handlers.values()]

        # Wait for all with timeout
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self._timeout * len(tasks) / self._max_concurrent + self._timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("Observer notification timed out")

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


class FunctionObserver:
    """Wrapper to make functions implement the Observer protocol."""

    def __init__(self, func: ObserverFunc | AsyncObserverFunc):
        self._func = func
        self.__name__ = getattr(func, "__name__", "anonymous_observer")

    async def handle(self, event: Any) -> None:
        """Handle the event by calling the wrapped function."""
        if asyncio.iscoroutinefunction(self._func):
            await self._func(event)
        else:
            # For sync functions, call directly
            # User is responsible for ensuring non-blocking code
            self._func(event)
