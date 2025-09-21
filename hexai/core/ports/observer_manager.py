"""Observer Manager Port - Clean interface for event observation systems."""

from abc import abstractmethod
from typing import Any, Protocol

from hexai.core.application.events.models import Event, Observer
from hexai.core.registry import port


@port(name="observer_manager", namespace="core")
class ObserverManagerPort(Protocol):
    """Port interface for event observation systems."""

    @abstractmethod
    async def notify(self, event: Event) -> None:
        """Send event to all registered observers."""
        ...

    @abstractmethod
    def register(self, observer: Observer, event_types: list[type] | None = None) -> str:
        """Register observer for specific event types. Returns observer ID."""
        ...

    @abstractmethod
    def unregister(self, observer_id: str) -> bool:
        """Unregister observer by ID."""
        ...

    @abstractmethod
    def list_observers(self) -> list[dict[str, Any]]:
        """List all registered observers with metadata."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean shutdown of observation system."""
        ...
