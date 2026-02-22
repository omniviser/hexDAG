"""EntityState lib — declarative state machines for business entities.

Tracks entity lifecycle (e.g. order: new → processing → shipped → delivered)
with validated transitions and full audit history.

Usage in YAML::

    spec:
      libs:
        entity_state:
          class: hexdag.stdlib.lib.EntityState

Programmatic::

    from hexdag.kernel.domain.entity_state import StateMachineConfig
    from hexdag.stdlib.lib.entity_state import EntityState

    state_lib = EntityState()
    state_lib.register_machine(StateMachineConfig(
        entity_type="order",
        states={"new", "processing", "shipped"},
        initial_state="new",
        transitions={"new": {"processing"}, "processing": {"shipped"}},
    ))
    await state_lib.aregister_entity("order", "order-123")
    await state_lib.atransition("order", "order-123", "processing")
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from hexdag.kernel.exceptions import HexDAGError
from hexdag.stdlib.lib_base import HexDAGLib

if TYPE_CHECKING:
    from hexdag.kernel.domain.entity_state import StateMachineConfig, StateTransition


class InvalidTransitionError(HexDAGError):
    """Raised when a state transition violates the machine config."""


class EntityState(HexDAGLib):
    """In-memory entity state tracker with validated transitions.

    Exposed tools
    -------------
    - ``aget_state(entity_type, entity_id)`` — current state
    - ``atransition(entity_type, entity_id, to_state)`` — validated transition
    - ``aget_history(entity_type, entity_id)`` — full audit trail
    - ``aregister_entity(entity_type, entity_id)`` — create new entity
    """

    def __init__(self) -> None:
        """Initialise state machines, entity states, and history stores."""
        # entity_type → StateMachineConfig
        self._machines: dict[str, StateMachineConfig] = {}
        # (entity_type, entity_id) → current state string
        self._states: dict[tuple[str, str], str] = {}
        # (entity_type, entity_id) → ordered list of transitions
        self._history: dict[tuple[str, str], list[StateTransition]] = {}

    # ------------------------------------------------------------------
    # Setup API (not tools — called before pipeline runs)
    # ------------------------------------------------------------------

    def register_machine(self, config: StateMachineConfig) -> None:
        """Register a state machine config for an entity type."""
        self._machines[config.entity_type] = config

    # ------------------------------------------------------------------
    # Agent-callable tools
    # ------------------------------------------------------------------

    async def aregister_entity(
        self,
        entity_type: str,
        entity_id: str,
        initial_state: str | None = None,
    ) -> dict[str, Any]:
        """Register a new entity and set its initial state.

        Args
        ----
            entity_type: The type of entity (e.g. "order", "ticket").
            entity_id: Unique identifier for the entity.
            initial_state: Override the default initial state.

        Returns
        -------
            Dict with entity_type, entity_id, and state.
        """
        from hexdag.kernel.domain.entity_state import StateTransition

        config = self._machines.get(entity_type)
        state = initial_state or (config.initial_state if config else "created")

        if config and state not in config.states:
            msg = f"State {state!r} not valid for {entity_type!r}"
            raise InvalidTransitionError(msg)

        key = (entity_type, entity_id)
        self._states[key] = state
        self._history.setdefault(key, []).append(
            StateTransition(
                entity_type=entity_type,
                entity_id=entity_id,
                from_state=None,
                to_state=state,
                timestamp=time.time(),
            )
        )
        return {"entity_type": entity_type, "entity_id": entity_id, "state": state}

    async def atransition(
        self,
        entity_type: str,
        entity_id: str,
        to_state: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Transition an entity to a new state.

        Validates the transition against the registered state machine.

        Args
        ----
            entity_type: The type of entity.
            entity_id: Unique identifier for the entity.
            to_state: Target state.
            reason: Optional reason for the transition.

        Returns
        -------
            Dict with entity details and old/new state.
        """
        from hexdag.kernel.domain.entity_state import StateTransition

        key = (entity_type, entity_id)
        current = self._states.get(key)
        if current is None:
            msg = f"Entity {entity_type!r}/{entity_id!r} not registered"
            raise InvalidTransitionError(msg)

        config = self._machines.get(entity_type)
        if config and not config.is_valid_transition(current, to_state):
            msg = (
                f"Cannot transition {entity_type!r}/{entity_id!r} from {current!r} to {to_state!r}"
            )
            raise InvalidTransitionError(msg)

        if config and to_state not in config.states:
            msg = f"State {to_state!r} not valid for {entity_type!r}"
            raise InvalidTransitionError(msg)

        self._states[key] = to_state
        metadata: dict[str, Any] = {}
        if reason:
            metadata["reason"] = reason
        self._history.setdefault(key, []).append(
            StateTransition(
                entity_type=entity_type,
                entity_id=entity_id,
                from_state=current,
                to_state=to_state,
                timestamp=time.time(),
                metadata=metadata,
            )
        )
        return {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "from_state": current,
            "to_state": to_state,
        }

    async def aget_state(self, entity_type: str, entity_id: str) -> dict[str, Any] | None:
        """Get the current state of an entity.

        Args
        ----
            entity_type: The type of entity.
            entity_id: Unique identifier for the entity.

        Returns
        -------
            Dict with entity_type, entity_id, and current state, or None.
        """
        key = (entity_type, entity_id)
        current = self._states.get(key)
        if current is None:
            return None
        return {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "state": current,
        }

    async def aget_history(self, entity_type: str, entity_id: str) -> list[dict[str, Any]]:
        """Get the full state transition history of an entity.

        Args
        ----
            entity_type: The type of entity.
            entity_id: Unique identifier for the entity.

        Returns
        -------
            List of transition records, oldest first.
        """
        key = (entity_type, entity_id)
        transitions = self._history.get(key, [])
        return [
            {
                "from_state": t.from_state,
                "to_state": t.to_state,
                "timestamp": t.timestamp,
                "metadata": t.metadata,
            }
            for t in transitions
        ]
