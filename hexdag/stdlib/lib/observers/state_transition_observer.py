"""State transition observer — accumulates entity state transitions.

Provides a queryable audit trail of ``StateTransitionEvent`` occurrences
during pipeline execution.  Follows the same pattern as
``PortCallStoreObserver``.

Example
-------
    >>> from hexdag.stdlib.lib.observers import StateTransitionObserver
    >>> observer = StateTransitionObserver()
    >>> # Register with observer_manager for StateTransitionEvent
    >>> # ... run pipeline ...
    >>> transitions = observer.get_transitions()  # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hexdag.kernel.logging import get_logger
from hexdag.kernel.orchestration.events.events import Event, StateTransitionEvent

logger = get_logger(__name__)


@dataclass(slots=True)
class StoredTransition:
    """Serialisable record of a single state transition.

    Attributes
    ----------
    entity_type : str
        Entity kind (e.g. ``"order"``, ``"ticket"``).
    entity_id : str
        Unique identifier for the entity instance.
    from_state : str
        State before transition.
    to_state : str
        State after transition.
    reason : str | None
        Optional reason for the transition.
    node_name : str
        DAG node that triggered the transition.
    timestamp : float
        Unix timestamp when the event was created.
    metadata : dict[str, Any]
        Additional event metadata.
    """

    entity_type: str
    entity_id: str
    from_state: str
    to_state: str
    reason: str | None
    node_name: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


class StateTransitionObserver:
    """Observer that accumulates state transition events for querying.

    Listens to ``StateTransitionEvent`` occurrences and stores them
    in memory.  Provides filtering by entity type, entity ID, and
    state for audit trails and debugging.

    Example
    -------
        >>> observer = StateTransitionObserver()
        >>> # After pipeline run:
        >>> order_transitions = observer.get_transitions(  # doctest: +SKIP
        ...     entity_type="order",
        ... )
    """

    def __init__(self) -> None:
        self._transitions: list[StoredTransition] = []

    async def handle(self, event: Event) -> None:
        """Handle state transition events by storing them.

        Parameters
        ----------
        event : Event
            The event to process.  Only ``StateTransitionEvent`` is stored.
        """
        if not isinstance(event, StateTransitionEvent):
            return

        record = StoredTransition(
            entity_type=event.entity_type,
            entity_id=event.entity_id,
            from_state=event.from_state,
            to_state=event.to_state,
            reason=event.reason,
            node_name=event.node_name,
            timestamp=event.timestamp.timestamp(),
        )
        self._transitions.append(record)

    def get_transitions(
        self,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        from_state: str | None = None,
        to_state: str | None = None,
    ) -> list[StoredTransition]:
        """Query accumulated transitions with optional filters.

        Parameters
        ----------
        entity_type : str | None
            Filter by entity type.
        entity_id : str | None
            Filter by entity ID.
        from_state : str | None
            Filter by source state.
        to_state : str | None
            Filter by target state.

        Returns
        -------
        list[StoredTransition]
            Matching transitions in chronological order.
        """
        result = self._transitions
        if entity_type is not None:
            result = [t for t in result if t.entity_type == entity_type]
        if entity_id is not None:
            result = [t for t in result if t.entity_id == entity_id]
        if from_state is not None:
            result = [t for t in result if t.from_state == from_state]
        if to_state is not None:
            result = [t for t in result if t.to_state == to_state]
        return result

    def clear(self) -> None:
        """Clear all accumulated transitions."""
        self._transitions.clear()

    @property
    def count(self) -> int:
        """Total number of stored transitions."""
        return len(self._transitions)
