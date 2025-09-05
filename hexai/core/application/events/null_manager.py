"""Null object pattern for event manager to eliminate boilerplate."""

from typing import Any


class NullEventManager:
    """Event manager that does nothing - eliminates if event_manager checks."""

    async def emit(self, event: Any) -> None:
        """Do nothing - null object pattern."""
        pass

    async def notify(self, event: Any) -> None:
        """Do nothing - null object pattern."""
        pass


# Singleton instance
NULL_EVENT_MANAGER = NullEventManager()


def get_event_manager(ports: dict[str, Any]) -> Any:
    """Get event manager from ports, returning null manager if not found.

    This eliminates the need for 'if event_manager:' checks everywhere.

    Parameters
    ----------
    ports : dict[str, Any]
        Dictionary of ports that may contain event_manager

    Returns
    -------
    Any
        Event manager or null manager that does nothing
    """
    return ports.get("event_manager", NULL_EVENT_MANAGER)
