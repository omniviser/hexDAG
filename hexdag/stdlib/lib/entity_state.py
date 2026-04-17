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

from hexdag.kernel.exceptions import InvalidTransitionError  # noqa: F401
from hexdag.kernel.service import Service, tool

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from hexdag.kernel.domain.entity_state import StateMachineConfig, StateTransition
    from hexdag.kernel.orchestration.events.events import TransitionContext
    from hexdag.kernel.ports.data_store import SupportsCollectionStorage

_STATES_COLLECTION = "entity_states"
_HISTORY_COLLECTION = "state_history"


def _entity_key(entity_type: str, entity_id: str) -> str:
    """Build a storage key from entity type and ID."""
    return f"{entity_type}:{entity_id}"


class EntityState(Service):
    """Entity state tracker with validated transitions and optional persistence.

    Exposed tools
    -------------
    - ``aget_state(entity_type, entity_id)`` — current state
    - ``atransition(entity_type, entity_id, to_state)`` — validated transition
    - ``aget_history(entity_type, entity_id)`` — full audit trail
    - ``aregister_entity(entity_type, entity_id)`` — create new entity
    """

    def __init__(self, storage: SupportsCollectionStorage | None = None) -> None:
        """Initialise state machines, entity states, and history stores.

        Args
        ----
            storage: Optional persistent backend.  When ``None`` (default),
                all data lives only in memory.
        """
        self._storage = storage
        # entity_type → StateMachineConfig
        self._machines: dict[str, StateMachineConfig] = {}
        # (entity_type, entity_id) → current state string
        self._states: dict[tuple[str, str], str] = {}
        # (entity_type, entity_id) → ordered list of transitions
        self._history: dict[tuple[str, str], list[StateTransition]] = {}
        # entity_type → async handler callable (transactional: failure = rollback)
        self._transition_handlers: dict[str, Callable[..., Any]] = {}

    # ------------------------------------------------------------------
    # Setup API (not tools — called before pipeline runs)
    # ------------------------------------------------------------------

    def register_machine(self, config: StateMachineConfig) -> None:
        """Register a state machine config for an entity type."""
        self._machines[config.entity_type] = config

    def register_handler(
        self,
        entity_type: str,
        handler: Callable[..., Any],
    ) -> None:
        """Register a transition handler for an entity type.

        Handlers are called asynchronously after a successful transition.
        **Handler failure = transition failure** — the in-memory state is
        rolled back and the error propagates to the caller.

        The handler receives keyword arguments::

            async def on_transition(
                entity_type: str,
                entity_id: str,
                from_state: str,
                to_state: str,
                reason: str | None,
                context: TransitionContext | None,
            ) -> None: ...

        Args
        ----
            entity_type: Entity type to attach the handler to.
            handler: Async callable invoked on every transition.
        """
        self._transition_handlers[entity_type] = handler

    # ------------------------------------------------------------------
    # Collection interface
    # ------------------------------------------------------------------

    def __contains__(self, entity_type: str) -> bool:
        """Check if an entity type has a registered state machine."""
        return entity_type in self._machines

    def __iter__(self) -> Iterator[str]:
        """Iterate over registered entity type names."""
        return iter(self._machines)

    def __len__(self) -> int:
        """Number of registered entity types."""
        return len(self._machines)

    def entity_ids(self, entity_type: str) -> list[str]:
        """Get all entity IDs for a given type."""
        return sorted(eid for (etype, eid) in self._states if etype == entity_type)

    # ------------------------------------------------------------------
    # Agent-callable tools
    # ------------------------------------------------------------------

    @tool
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
        from hexdag.kernel.domain.entity_state import (
            StateTransition,  # lazy: avoid import cycle with kernel
            state_transition_to_storage,
        )

        config = self._machines.get(entity_type)
        state = initial_state or (config.initial_state if config else "created")

        if config and state not in config.states:
            msg = f"State {state!r} not valid for {entity_type!r}"
            raise InvalidTransitionError(msg)

        key = (entity_type, entity_id)
        self._states[key] = state
        transition = StateTransition(
            entity_type=entity_type,
            entity_id=entity_id,
            from_state=None,
            to_state=state,
            timestamp=time.time(),
        )
        self._history.setdefault(key, []).append(transition)

        if self._storage is not None:
            storage_key = _entity_key(entity_type, entity_id)
            await self._storage.asave(
                _STATES_COLLECTION,
                storage_key,
                {"entity_type": entity_type, "entity_id": entity_id, "state": state},
            )
            await self._storage.asave(
                _HISTORY_COLLECTION,
                storage_key,
                {
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "transitions": [state_transition_to_storage(transition)],
                },
            )

        return {"entity_type": entity_type, "entity_id": entity_id, "state": state}

    @tool
    async def atransition(
        self,
        entity_type: str,
        entity_id: str,
        to_state: str,
        reason: str | None = None,
        *,
        _context: TransitionContext | None = None,
    ) -> dict[str, Any]:
        """Transition an entity to a new state.

        Validates the transition against the registered state machine.
        If a transition handler is registered, it is called after persistence
        but **before** the in-memory state is updated.  Handler failure rolls
        back the transition (the entity stays in its previous state).

        After a successful transition the method auto-emits a
        ``StateTransitionEvent`` through the observer manager (if one is
        available in the execution context).

        Args
        ----
            entity_type: The type of entity.
            entity_id: Unique identifier for the entity.
            to_state: Target state.
            reason: Optional reason for the transition.
            _context: Internal — transition context injected by framework.

        Returns
        -------
            Dict with entity details and old/new state.
        """
        from hexdag.kernel.domain.entity_state import (
            StateTransition,  # lazy: avoid import cycle with kernel
            state_transition_to_storage,
        )

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

        # Build transition record
        metadata: dict[str, Any] = {}
        if reason:
            metadata["reason"] = reason
        transition = StateTransition(
            entity_type=entity_type,
            entity_id=entity_id,
            from_state=current,
            to_state=to_state,
            timestamp=time.time(),
            metadata=metadata,
        )

        # Write-ahead: persist BEFORE updating in-memory state (P2)
        if self._storage is not None:
            storage_key = _entity_key(entity_type, entity_id)
            await self._storage.asave(
                _STATES_COLLECTION,
                storage_key,
                {"entity_type": entity_type, "entity_id": entity_id, "state": to_state},
            )
            history_doc = await self._storage.aload(_HISTORY_COLLECTION, storage_key)
            transitions_list = history_doc.get("transitions", []) if history_doc else []
            transitions_list.append(state_transition_to_storage(transition))
            await self._storage.asave(
                _HISTORY_COLLECTION,
                storage_key,
                {
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "transitions": transitions_list,
                },
            )

        # Call transition handler (transactional: failure = rollback)
        handler = self._transition_handlers.get(entity_type)
        if handler is not None:
            try:
                await handler(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    from_state=current,
                    to_state=to_state,
                    reason=reason,
                    context=_context,
                )
            except Exception:
                # Rollback: revert storage to previous state AND history
                if self._storage is not None:
                    storage_key = _entity_key(entity_type, entity_id)
                    await self._storage.asave(
                        _STATES_COLLECTION,
                        storage_key,
                        {
                            "entity_type": entity_type,
                            "entity_id": entity_id,
                            "state": current,
                        },
                    )
                    history_doc = await self._storage.aload(_HISTORY_COLLECTION, storage_key)
                    if history_doc:
                        rollback_transitions = history_doc.get("transitions", [])
                        if rollback_transitions:
                            rollback_transitions.pop()
                        await self._storage.asave(
                            _HISTORY_COLLECTION,
                            storage_key,
                            {
                                "entity_type": entity_type,
                                "entity_id": entity_id,
                                "transitions": rollback_transitions,
                            },
                        )
                raise

        # Commit in-memory state (after storage + handler succeeded)
        self._states[key] = to_state
        self._history.setdefault(key, []).append(transition)

        # Auto-emit StateTransitionEvent (fire-and-forget via observer)
        await self._emit_transition_event(
            entity_type,
            entity_id,
            current,
            to_state,
            reason,
            _context,
        )

        return {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "from_state": current,
            "to_state": to_state,
        }

    async def _emit_transition_event(
        self,
        entity_type: str,
        entity_id: str,
        from_state: str,
        to_state: str,
        reason: str | None,
        context: TransitionContext | None,
    ) -> None:
        """Emit StateTransitionEvent if an observer manager is available."""
        try:
            from hexdag.kernel.context.execution_context import (
                get_observer_manager,  # lazy: avoid circular import with kernel
            )
            from hexdag.kernel.orchestration.events.events import (
                StateTransitionEvent,  # lazy: avoid circular import with kernel
            )

            observer = get_observer_manager()
            if observer is not None:
                node_name = context.node_name if context else ""
                event = StateTransitionEvent(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    from_state=from_state,
                    to_state=to_state,
                    reason=reason,
                    node_name=node_name,
                )
                await observer.notify(event)
        except Exception:  # noqa: BLE001
            pass  # Observer failures are best-effort, never block transitions

    @tool
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
        if current is not None:
            return {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "state": current,
            }
        if self._storage is not None:
            data = await self._storage.aload(
                _STATES_COLLECTION, _entity_key(entity_type, entity_id)
            )
            if data is not None:
                return {
                    "entity_type": data.get("entity_type"),
                    "entity_id": data.get("entity_id"),
                    "state": data.get("state"),
                }
        return None

    @tool
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
        transitions = self._history.get(key)
        if transitions is not None:
            return [
                {
                    "from_state": t.from_state,
                    "to_state": t.to_state,
                    "timestamp": t.timestamp,
                    "metadata": t.metadata,
                }
                for t in transitions
            ]
        if self._storage is not None:
            data = await self._storage.aload(
                _HISTORY_COLLECTION, _entity_key(entity_type, entity_id)
            )
            if data is not None:
                return [
                    {
                        "from_state": t.get("from_state"),
                        "to_state": t.get("to_state"),
                        "timestamp": t.get("timestamp"),
                        "metadata": t.get("metadata", {}),
                    }
                    for t in data.get("transitions", [])
                ]
        return []

    # ------------------------------------------------------------------
    # Schema enrichment — enrich tool descriptions from state machines
    # ------------------------------------------------------------------

    def _build_transition_map_description(self) -> str:
        """Build a human-readable transition map from registered machines.

        Used to enrich the ``atransition`` tool description so agents can
        reason about valid transitions without trial-and-error.
        """
        if not self._machines:
            return ""

        lines = ["\n\nAvailable state machines:"]
        for entity_type, config in sorted(self._machines.items()):
            parts = []
            for from_state, targets in sorted(config.transitions.items()):
                sorted_targets = sorted(targets)
                parts.append(f"{from_state} -> [{', '.join(sorted_targets)}]")
            lines.append(f"- {entity_type}: {' | '.join(parts)}")
        return "\n".join(lines)

    def get_tools(self) -> dict[str, Any]:
        """Return tools with enriched descriptions from state machines.

        When state machines are registered, the ``atransition`` tool
        description is extended with the full transition map and the
        ``entity_type`` parameter gains an ``enum`` constraint listing
        valid entity types.
        """
        tools = super().get_tools()

        if "atransition" in tools:
            original = tools["atransition"]
            fn = getattr(original, "__func__", None)
            if fn is not None:
                # Capture the first line of the original docstring once
                if not hasattr(fn, "_original_first_line"):
                    raw_doc = fn.__doc__ or ""
                    fn._original_first_line = raw_doc.strip().split("\n")[0]

                if self._machines:
                    transition_map = self._build_transition_map_description()
                    fn.__doc__ = fn._original_first_line + transition_map
                else:
                    fn.__doc__ = fn._original_first_line

        return tools
