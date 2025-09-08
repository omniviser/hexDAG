"""Base manager class for event system components.

This module provides a common base class for both ObserverManager and ControlManager,
defining their shared API while allowing for different implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

# Type variable for handler types
THandler = TypeVar("THandler")


class BaseEventManager(ABC, Generic[THandler]):
    """Abstract base class for event system managers.

    Provides a common API for registering handlers and clearing them.
    Subclasses must implement the specific registration and processing logic.
    """

    def __init__(self) -> None:
        """Initialize the base manager."""
        self._handlers: dict[str, THandler] = {}

    @abstractmethod
    def register(self, handler: Any, **kwargs: Any) -> Any:
        """Register a handler with the manager.

        Args
        ----
            handler: The handler to register
            **kwargs: Additional registration parameters

        Returns
        -------
            Registration identifier or None
        """
        ...

    def unregister(self, handler_id: str) -> bool:
        """Unregister a handler by ID.

        Args
        ----
            handler_id: The ID of the handler to unregister

        Returns
        -------
            bool: True if handler was found and removed, False otherwise
        """
        if handler_id in self._handlers:
            del self._handlers[handler_id]
            return True
        return False

    def clear(self) -> None:
        """Remove all registered handlers."""
        self._handlers.clear()

    def __len__(self) -> int:
        """Return number of registered handlers."""
        return len(self._handlers)

    async def close(self) -> None:
        """Close the manager and cleanup resources."""
        self.clear()

    def __enter__(self) -> "BaseEventManager[THandler]":
        """Support context manager protocol."""
        return self

    async def __aenter__(self) -> "BaseEventManager[THandler]":
        """Support async context manager protocol."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:  # noqa: ARG002
        """Exit context manager."""
        import asyncio

        asyncio.create_task(self.close())

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:  # noqa: ARG002
        """Exit async context manager."""
        await self.close()
