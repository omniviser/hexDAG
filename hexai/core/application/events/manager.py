"""Simple observer manager for logging and metrics - read only."""

import asyncio
import logging
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)


class Observer(Protocol):
    """Protocol for observers - just need to handle events."""

    async def handle(self, event: Any) -> None:
        """Handle an event (async)."""
        ...


class ObserverManager:
    """Manages observers for logging, metrics, telemetry.

    Key principles:
    - Observers are READ-ONLY - they cannot affect execution
    - Failures in observers don't crash the pipeline
    - Fire-and-forget pattern
    """

    def __init__(self) -> None:
        self._observers: list[Observer | Callable] = []

    def attach(self, observer: Observer | Callable) -> None:
        """Attach an observer."""
        self._observers.append(observer)

    def detach(self, observer: Observer | Callable) -> None:
        """Detach an observer."""
        if observer in self._observers:
            self._observers.remove(observer)

    async def notify(self, event: Any) -> None:
        """Notify all observers of an event.

        Errors are logged but don't affect execution.
        """
        if not self._observers:
            return

        # Fire all observers concurrently
        tasks = []
        for observer in self._observers:
            if hasattr(observer, "handle"):
                # Observer protocol
                tasks.append(self._safe_handle(observer, event))
            elif asyncio.iscoroutinefunction(observer):
                # Async function
                tasks.append(self._safe_call(observer, event))
            else:
                # Sync function - run in executor
                tasks.append(self._safe_call_sync(observer, event))

        # Wait for all, but don't fail on exceptions
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_handle(self, observer: Observer, event: Any) -> None:
        """Safely call observer.handle()."""
        try:
            await observer.handle(event)
        except Exception as e:
            logger.error(f"Observer {observer.__class__.__name__} failed: {e}")

    async def _safe_call(self, func: Callable, event: Any) -> None:
        """Safely call an async function."""
        try:
            await func(event)
        except Exception as e:
            logger.error(f"Observer function {func.__name__} failed: {e}")

    async def _safe_call_sync(self, func: Callable, event: Any) -> None:
        """Safely call a sync function in executor."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, func, event)
        except Exception as e:
            logger.error(f"Observer function {func.__name__} failed: {e}")

    def clear(self) -> None:
        """Remove all observers."""
        self._observers.clear()

    def __len__(self) -> int:
        """Return number of attached observers."""
        return len(self._observers)
