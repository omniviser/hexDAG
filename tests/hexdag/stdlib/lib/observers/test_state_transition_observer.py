"""Tests for StateTransitionObserver."""

import pytest

from hexdag.kernel.orchestration.events.events import (
    NodeCompleted,
    StateTransitionEvent,
)
from hexdag.stdlib.lib.observers.state_transition_observer import (
    StateTransitionObserver,
)


def _make_event(
    entity_type: str = "order",
    entity_id: str = "ORD-1",
    from_state: str = "new",
    to_state: str = "processing",
    reason: str | None = None,
    node_name: str = "transition_node",
) -> StateTransitionEvent:
    return StateTransitionEvent(
        entity_type=entity_type,
        entity_id=entity_id,
        from_state=from_state,
        to_state=to_state,
        reason=reason,
        node_name=node_name,
    )


class TestStateTransitionObserver:
    """Tests for state transition event accumulation and querying."""

    @pytest.mark.asyncio()
    async def test_stores_transition_event(self) -> None:
        observer = StateTransitionObserver()
        event = _make_event()
        await observer.handle(event)
        assert observer.count == 1
        transitions = observer.get_transitions()
        assert len(transitions) == 1
        assert transitions[0].entity_type == "order"
        assert transitions[0].to_state == "processing"

    @pytest.mark.asyncio()
    async def test_ignores_non_transition_events(self) -> None:
        observer = StateTransitionObserver()
        other_event = NodeCompleted(
            name="some_node",
            wave_index=0,
            duration_ms=10.0,
            result={"ok": True},
        )
        await observer.handle(other_event)
        assert observer.count == 0

    @pytest.mark.asyncio()
    async def test_filter_by_entity_type(self) -> None:
        observer = StateTransitionObserver()
        await observer.handle(_make_event(entity_type="order"))
        await observer.handle(_make_event(entity_type="ticket"))
        await observer.handle(_make_event(entity_type="order"))

        orders = observer.get_transitions(entity_type="order")
        assert len(orders) == 2
        tickets = observer.get_transitions(entity_type="ticket")
        assert len(tickets) == 1

    @pytest.mark.asyncio()
    async def test_filter_by_entity_id(self) -> None:
        observer = StateTransitionObserver()
        await observer.handle(_make_event(entity_id="ORD-1"))
        await observer.handle(_make_event(entity_id="ORD-2"))

        result = observer.get_transitions(entity_id="ORD-1")
        assert len(result) == 1
        assert result[0].entity_id == "ORD-1"

    @pytest.mark.asyncio()
    async def test_filter_by_to_state(self) -> None:
        observer = StateTransitionObserver()
        await observer.handle(_make_event(to_state="processing"))
        await observer.handle(_make_event(to_state="shipped"))

        result = observer.get_transitions(to_state="shipped")
        assert len(result) == 1

    @pytest.mark.asyncio()
    async def test_filter_by_from_state(self) -> None:
        observer = StateTransitionObserver()
        await observer.handle(_make_event(from_state="new", to_state="processing"))
        await observer.handle(_make_event(from_state="processing", to_state="shipped"))

        result = observer.get_transitions(from_state="processing")
        assert len(result) == 1
        assert result[0].to_state == "shipped"

    @pytest.mark.asyncio()
    async def test_combined_filters(self) -> None:
        observer = StateTransitionObserver()
        await observer.handle(
            _make_event(
                entity_type="order",
                entity_id="ORD-1",
                to_state="processing",
            )
        )
        await observer.handle(
            _make_event(
                entity_type="order",
                entity_id="ORD-2",
                to_state="processing",
            )
        )
        await observer.handle(
            _make_event(
                entity_type="ticket",
                entity_id="T-1",
                to_state="investigating",
            )
        )

        result = observer.get_transitions(entity_type="order", entity_id="ORD-1")
        assert len(result) == 1

    @pytest.mark.asyncio()
    async def test_clear(self) -> None:
        observer = StateTransitionObserver()
        await observer.handle(_make_event())
        assert observer.count == 1
        observer.clear()
        assert observer.count == 0

    @pytest.mark.asyncio()
    async def test_stored_transition_fields(self) -> None:
        observer = StateTransitionObserver()
        await observer.handle(
            _make_event(
                entity_type="order",
                entity_id="ORD-99",
                from_state="new",
                to_state="cancelled",
                reason="customer request",
                node_name="cancel_node",
            )
        )
        t = observer.get_transitions()[0]
        assert t.entity_type == "order"
        assert t.entity_id == "ORD-99"
        assert t.from_state == "new"
        assert t.to_state == "cancelled"
        assert t.reason == "customer request"
        assert t.node_name == "cancel_node"
        assert t.timestamp > 0
