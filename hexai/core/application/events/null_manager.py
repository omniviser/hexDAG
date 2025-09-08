"""Null object pattern implementations for event system.

Provides null implementations to eliminate if/else checks in client code.
Clear separation between control plane (check) and observability plane (notify).
"""

from typing import Any, Optional

from .models import ControlResponse, ControlSignal


class NullEventBus:
    """Null implementation of EventBus for control plane.

    Always returns PROCEED, allowing execution to continue.
    """

    async def check(self, event: Any, context: Any) -> ControlResponse:
        """Return PROCEED to allow execution."""
        return ControlResponse(signal=ControlSignal.PROCEED)

    def register(self, handler: Any, **kwargs: Any) -> None:
        """No-op registration."""
        pass

    def clear(self) -> None:
        """No-op clear."""
        pass


class NullObserverManager:
    """Null implementation of ObserverManager for observability plane.

    Silently ignores all notifications.
    """

    async def notify(self, event: Any) -> None:
        """Silently ignore notification."""
        pass

    def register(self, observer: Any, observer_id: Optional[str] = None) -> str:
        """No-op registration, returns dummy ID."""
        return "null-observer-id"

    def unregister(self, observer_id: str) -> bool:
        """No-op unregistration, always returns False."""
        return False

    def clear(self) -> None:
        """No-op clear."""
        pass

    def __len__(self) -> int:
        """Return 0 for empty manager."""
        return 0


# Singleton instances for use in tests and as defaults
NULL_EVENT_BUS = NullEventBus()
NULL_OBSERVER_MANAGER = NullObserverManager()


def get_event_bus(ports: dict[str, Any]) -> Any:
    """Get event bus from ports, returning null bus if not found.

    This eliminates the need for 'if event_bus:' checks everywhere.

    Parameters
    ----------
    ports : dict[str, Any]
        Dictionary of ports that may contain event_bus

    Returns
    -------
    Any
        Event bus or null bus that always allows execution
    """
    return ports.get("event_bus", NULL_EVENT_BUS)


def get_observer_manager(ports: dict[str, Any]) -> Any:
    """Get observer manager from ports, returning null manager if not found.

    This eliminates the need for 'if observer_manager:' checks everywhere.

    Parameters
    ----------
    ports : dict[str, Any]
        Dictionary of ports that may contain observer_manager

    Returns
    -------
    Any
        Observer manager or null manager that silently ignores notifications
    """
    return ports.get("observer_manager", NULL_OBSERVER_MANAGER)
