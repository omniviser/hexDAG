"""Configuration and defaults for the event system.

Provides null implementations and factory functions to eliminate
if/else checks in client code.
"""

from typing import Any

from .models import ControlResponse, ControlSignal, ExecutionContext

# Default configuration values
DEFAULT_MAX_CONCURRENT_OBSERVERS = 10
DEFAULT_OBSERVER_TIMEOUT = 5.0
DEFAULT_HANDLER_PRIORITY = 100
CRITICAL_HANDLER_PRIORITY_THRESHOLD = 50


class NullControlManager:
    """Null implementation of ControlManager.

    Always returns PROCEED, allowing execution to continue.
    Used when no control management is needed.
    """

    async def check(self, event: Any, context: ExecutionContext) -> ControlResponse:
        """Return PROCEED to allow execution.

        Returns
        -------
        ControlResponse
            Response with PROCEED signal to allow execution
        """
        return ControlResponse(signal=ControlSignal.PROCEED)

    def register(self, handler: Any, **kwargs: Any) -> str:
        """No-op registration, returns dummy ID.

        Returns
        -------
        str
            Dummy handler ID
        """
        return "null-control-handler"

    def unregister(self, handler_id: str) -> bool:
        """No-op unregistration, always returns False.

        Returns
        -------
        bool
            Always False for null implementation
        """
        return False

    def clear(self) -> None:
        """No-op clear."""
        pass

    def __len__(self) -> int:
        """Return 0 for empty manager.

        Returns
        -------
        int
            Always 0 for empty null manager
        """
        return 0

    async def close(self) -> None:
        """No-op close."""
        pass


class NullObserverManager:
    """Null implementation of ObserverManager.

    Silently ignores all notifications.
    Used when no observation is needed.
    """

    async def notify(self, event: Any) -> None:
        """Silently ignore notification."""
        pass

    def register(self, observer: Any, **kwargs: Any) -> str:
        """No-op registration, returns dummy ID.

        Returns
        -------
        str
            Dummy observer ID
        """
        return "null-observer-id"

    def unregister(self, observer_id: str) -> bool:
        """No-op unregistration, always returns False.

        Returns
        -------
        bool
            Always False for null implementation
        """
        return False

    def clear(self) -> None:
        """No-op clear."""
        pass

    def __len__(self) -> int:
        """Return 0 for empty manager.

        Returns
        -------
        int
            Always 0 for empty null manager
        """
        return 0

    async def close(self) -> None:
        """No-op close."""
        pass


# Singleton instances for use in tests and as defaults
NULL_CONTROL_MANAGER = NullControlManager()
NULL_OBSERVER_MANAGER = NullObserverManager()


def get_control_manager(ports: dict[str, Any]) -> Any:
    """Get control manager from ports, returning null manager if not found.

    This eliminates the need for 'if control_manager:' checks everywhere.

    Args
    ----
        ports: Dictionary of ports that may contain control_manager

    Returns
    -------
    Any
        Control manager or null manager that always allows execution
    """
    return ports.get("control_manager", NULL_CONTROL_MANAGER)


def get_observer_manager(ports: dict[str, Any]) -> Any:
    """Get observer manager from ports, returning null manager if not found.

    This eliminates the need for 'if observer_manager:' checks everywhere.

    Args
    ----
        ports: Dictionary of ports that may contain observer_manager

    Returns
    -------
    Any
        Observer manager or null manager that silently ignores notifications
    """
    return ports.get("observer_manager", NULL_OBSERVER_MANAGER)
