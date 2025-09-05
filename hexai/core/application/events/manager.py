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
from typing import Any, Callable, Protocol, Union

logger = logging.getLogger(__name__)


class Observer(Protocol):
    """Protocol for observers."""

    async def handle(self, event: Any) -> None:
        """Handle an event."""
        ...


# Types that can be observers
ObserverFunc = Callable[[Any], None]
AsyncObserverFunc = Callable[[Any], Any]  # Returns awaitable
ObserverLike = Union[Observer, ObserverFunc, AsyncObserverFunc]


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

    def __init__(self) -> None:
        self._observers: list[Observer] = []

    def register(self, observer: ObserverLike) -> None:
        """Register an observer.

        Args
        ----
            observer: Either an Observer protocol implementation or
                     a function (sync/async) that takes an event
        """
        if hasattr(observer, "handle"):
            # Already implements protocol
            self._observers.append(observer)
        elif callable(observer):
            # Wrap function to implement the protocol
            self._observers.append(_make_observer(observer))
        else:
            raise TypeError(
                f"Observer must be callable or implement Observer protocol, got {type(observer)}"
            )

    def unregister(self, observer: Observer) -> None:
        """Unregister an observer.

        Warning: This only works for observers that were registered as objects
        implementing the Observer protocol. Functions that were auto-wrapped
        cannot be unregistered this way since a new wrapper is created each time.

        If you need to unregister functions, consider using a class that
        implements the Observer protocol instead.
        """
        # Only works for observers registered as-is (not wrapped functions)
        if observer in self._observers:
            self._observers.remove(observer)

    async def notify(self, event: Any) -> None:
        """Notify all observers of an event.

        Errors are logged but don't affect execution.
        """
        if not self._observers:
            return

        # Fire all observers concurrently
        tasks = [self._safe_invoke(observer, event) for observer in self._observers]

        # Wait for all, but don't fail on exceptions
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_invoke(self, observer: Observer, event: Any) -> None:
        """Safely invoke an observer."""
        try:
            await observer.handle(event)
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
