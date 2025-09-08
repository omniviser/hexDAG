"""Observability Port: Manages monitoring and telemetry.

This is a hexagonal architecture PORT that allows external systems
to observe pipeline execution without affecting it. Observers can:
- Log events
- Collect metrics
- Send telemetry
- Update dashboards
- Record audit trails

This port defines the protocol for observability but contains no
implementations (those belong in Tier 2 or external systems).
"""

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Union

from .models import AsyncObserverFunc, Observer, ObserverFunc, ObserverLike

logger = logging.getLogger(__name__)


def _make_observer(func: Union[ObserverFunc, AsyncObserverFunc]) -> Observer:
    """Convert an observer function into an Observer protocol.

    Args
    ----
        func: A function that takes an event and returns None (no return value)

    Returns
    -------
        An object implementing the Observer protocol
    """

    class FunctionObserver:
        def __init__(self, fn: Union[ObserverFunc, AsyncObserverFunc]):
            self._func = fn
            self.__name__ = getattr(fn, "__name__", "anonymous_observer")

        async def handle(self, event: Any) -> None:
            if asyncio.iscoroutinefunction(self._func):
                await self._func(event)
            else:
                # Run sync function in executor to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._func, event)

    return FunctionObserver(func)


class ObserverManager:
    """Observability Port implementation for monitoring.

    This is the primary port for observability in the hexagonal architecture.
    It defines how external systems can monitor pipeline execution.

    Key principles:
    - Observers are READ-ONLY - they cannot affect execution
    - Failures in observers don't crash the pipeline (fault isolation)
    - Fire-and-forget pattern (async, non-blocking)
    - No observer implementations here (Tier 1 = plumbing only)
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
        self._observers: dict[str, Observer] = {}
        self._max_concurrent = max_concurrent_observers
        self._timeout = observer_timeout
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_observers)

    def register(self, observer: ObserverLike, observer_id: Optional[str] = None) -> str:
        """Register an observer with optional ID.

        Args
        ----
            observer: Either an Observer protocol implementation or
                     a function (sync/async) that takes an event
            observer_id: Optional ID for the observer (generated if not provided)

        Returns
        -------
            str: The ID of the registered observer
        """
        # Generate ID if not provided
        if observer_id is None:
            observer_id = str(uuid.uuid4())

        # Wrap if needed
        if hasattr(observer, "handle"):
            # Already implements protocol
            self._observers[observer_id] = observer
        elif callable(observer):
            # Wrap function to implement the protocol
            self._observers[observer_id] = _make_observer(observer)
        else:
            raise TypeError(
                f"Observer must be callable or implement Observer protocol, got {type(observer)}"
            )

        return observer_id

    def unregister(self, observer_id: str) -> bool:
        """Unregister an observer by ID.

        Args
        ----
            observer_id: The ID of the observer to unregister

        Returns
        -------
            bool: True if observer was found and removed, False otherwise
        """
        if observer_id in self._observers:
            del self._observers[observer_id]
            return True
        return False

    async def notify(self, event: Any) -> None:
        """Notify all observers of an event.

        Errors are logged but don't affect execution.
        """
        if not self._observers:
            return

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self._max_concurrent)

        async def limited_invoke(obs: Observer) -> None:
            async with semaphore:
                await self._safe_invoke(obs, event)

        # Fire all observers with limited concurrency
        tasks = [limited_invoke(observer) for observer in self._observers.values()]

        # Wait for all with timeout
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self._timeout * 2,  # Total timeout for all observers
                )
            except asyncio.TimeoutError:
                logger.warning(f"Observer notification timed out after {self._timeout * 2}s")

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

    def clear(self) -> None:
        """Remove all observers."""
        self._observers.clear()

    def __len__(self) -> int:
        """Return number of attached observers."""
        return len(self._observers)

    def __del__(self) -> None:
        """Cleanup executor on deletion."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
