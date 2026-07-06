"""Tests for EntityState transition handlers + auto-emit (Phase 1)."""

import pytest

from hexdag.kernel.domain.entity_state import StateMachineConfig
from hexdag.stdlib.lib.entity_state import EntityState


def _make_ticket_machine():
    return StateMachineConfig(
        entity_type="ticket",
        states={"OPEN", "INVESTIGATING", "RESOLVED", "CLOSED"},
        initial_state="OPEN",
        transitions={
            "OPEN": {"INVESTIGATING"},
            "INVESTIGATING": {"RESOLVED"},
            "RESOLVED": {"CLOSED"},
        },
    )


class TestTransitionHandlers:
    @pytest.mark.asyncio()
    async def test_handler_called_on_transition(self):
        es = EntityState()
        es.register_machine(_make_ticket_machine())
        await es.aregister_entity("ticket", "T-1")

        calls = []

        async def handler(**kwargs):
            calls.append(kwargs)

        es.register_handler("ticket", handler)
        await es.atransition("ticket", "T-1", "INVESTIGATING", reason="test")

        assert len(calls) == 1
        assert calls[0]["from_state"] == "OPEN"
        assert calls[0]["to_state"] == "INVESTIGATING"
        assert calls[0]["reason"] == "test"

    @pytest.mark.asyncio()
    async def test_handler_failure_rolls_back(self):
        es = EntityState()
        es.register_machine(_make_ticket_machine())
        await es.aregister_entity("ticket", "T-1")

        async def failing_handler(**kwargs):
            msg = "DB write failed"
            raise RuntimeError(msg)

        es.register_handler("ticket", failing_handler)

        with pytest.raises(RuntimeError, match="DB write failed"):
            await es.atransition("ticket", "T-1", "INVESTIGATING")

        # State should NOT have changed (rollback)
        state = await es.aget_state("ticket", "T-1")
        assert state["state"] == "OPEN"

    @pytest.mark.asyncio()
    async def test_no_handler_still_works(self):
        es = EntityState()
        es.register_machine(_make_ticket_machine())
        await es.aregister_entity("ticket", "T-1")

        result = await es.atransition("ticket", "T-1", "INVESTIGATING")
        assert result["to_state"] == "INVESTIGATING"

    @pytest.mark.asyncio()
    async def test_handler_only_called_for_matching_entity_type(self):
        es = EntityState()
        es.register_machine(_make_ticket_machine())
        es.register_machine(
            StateMachineConfig(
                entity_type="load",
                states={"PENDING", "POSTED"},
                initial_state="PENDING",
                transitions={"PENDING": {"POSTED"}},
            )
        )
        await es.aregister_entity("ticket", "T-1")
        await es.aregister_entity("load", "L-1")

        ticket_calls = []

        async def ticket_handler(**kwargs):
            ticket_calls.append(kwargs)

        es.register_handler("ticket", ticket_handler)

        # Load transition should not trigger ticket handler
        await es.atransition("load", "L-1", "POSTED")
        assert len(ticket_calls) == 0

        # Ticket transition should trigger it
        await es.atransition("ticket", "T-1", "INVESTIGATING")
        assert len(ticket_calls) == 1


class TestHandlerPayload:
    @pytest.mark.asyncio()
    async def test_handler_with_payload_param_receives_it(self):
        es = EntityState()
        es.register_machine(_make_ticket_machine())
        await es.aregister_entity("ticket", "T-1")

        received = []

        async def handler(entity_type, entity_id, from_state, to_state, reason, context, payload):
            received.append(payload)

        es.register_handler("ticket", handler)
        await es.atransition(
            "ticket",
            "T-1",
            "INVESTIGATING",
            payload={"resolved_by": "user-1", "priority": 2},
        )

        assert received == [{"resolved_by": "user-1", "priority": 2}]

    @pytest.mark.asyncio()
    async def test_handler_with_kwargs_receives_payload(self):
        es = EntityState()
        es.register_machine(_make_ticket_machine())
        await es.aregister_entity("ticket", "T-1")

        received = []

        async def handler(**kwargs):
            received.append(kwargs.get("payload"))

        es.register_handler("ticket", handler)
        await es.atransition("ticket", "T-1", "INVESTIGATING", payload={"a": 1})

        assert received == [{"a": 1}]

    @pytest.mark.asyncio()
    async def test_legacy_handler_without_payload_param_unchanged(self):
        """Handlers with the original six-kwarg signature never see payload."""
        es = EntityState()
        es.register_machine(_make_ticket_machine())
        await es.aregister_entity("ticket", "T-1")

        calls = []

        async def legacy_handler(entity_type, entity_id, from_state, to_state, reason, context):
            calls.append(entity_id)

        es.register_handler("ticket", legacy_handler)
        # Passing payload must not break a handler that can't accept it.
        await es.atransition("ticket", "T-1", "INVESTIGATING", payload={"ignored": True})

        assert calls == ["T-1"]

    @pytest.mark.asyncio()
    async def test_payload_defaults_to_none(self):
        es = EntityState()
        es.register_machine(_make_ticket_machine())
        await es.aregister_entity("ticket", "T-1")

        received = []

        async def handler(payload=None, **kwargs):
            received.append(payload)

        es.register_handler("ticket", handler)
        await es.atransition("ticket", "T-1", "INVESTIGATING")

        assert received == [None]

    @pytest.mark.asyncio()
    async def test_payload_handler_failure_still_rolls_back(self):
        es = EntityState()
        es.register_machine(_make_ticket_machine())
        await es.aregister_entity("ticket", "T-1")

        async def failing_handler(payload=None, **kwargs):
            msg = f"rejected: {payload}"
            raise RuntimeError(msg)

        es.register_handler("ticket", failing_handler)

        with pytest.raises(RuntimeError, match="rejected"):
            await es.atransition("ticket", "T-1", "INVESTIGATING", payload={"x": 1})

        state = await es.aget_state("ticket", "T-1")
        assert state["state"] == "OPEN"


class TestSchemaEnrichment:
    def test_get_tools_enriches_description(self):
        es = EntityState()
        es.register_machine(_make_ticket_machine())

        tools = es.get_tools()
        assert "atransition" in tools

        # The docstring should mention the state machine
        doc = tools["atransition"].__doc__ or ""
        assert "ticket" in doc
        assert "OPEN" in doc
        assert "INVESTIGATING" in doc

    def test_get_tools_no_enrichment_without_machines(self):
        es = EntityState()
        tools = es.get_tools()
        assert "atransition" in tools

        doc = tools["atransition"].__doc__ or ""
        assert "Available state machines" not in doc

    def test_enrichment_updates_on_new_machine(self):
        es = EntityState()

        # First call — no machines
        tools1 = es.get_tools()
        doc1 = tools1["atransition"].__doc__ or ""

        # Register machine
        es.register_machine(_make_ticket_machine())

        # Second call — should be enriched
        tools2 = es.get_tools()
        doc2 = tools2["atransition"].__doc__ or ""
        assert "ticket" in doc2
        assert len(doc2) > len(doc1)


class TestWriteAhead:
    @pytest.mark.asyncio()
    async def test_handler_failure_does_not_update_inmemory(self):
        """Write-ahead: if handler fails, in-memory state stays at previous."""
        es = EntityState()
        es.register_machine(_make_ticket_machine())
        await es.aregister_entity("ticket", "T-1")

        async def fail(**kwargs):
            raise ValueError("boom")

        es.register_handler("ticket", fail)

        with pytest.raises(ValueError, match="boom"):
            await es.atransition("ticket", "T-1", "INVESTIGATING")

        # In-memory state should be unchanged
        assert es._states[("ticket", "T-1")] == "OPEN"
        # History should NOT contain the failed transition
        history = es._history.get(("ticket", "T-1"), [])
        assert all(t.to_state != "INVESTIGATING" for t in history)


class TestComposableHandlerLists:
    """Multiple handlers per entity type: ordered dispatch + rollback."""

    @pytest.mark.asyncio()
    async def test_handlers_fire_in_registration_order(self):
        es = EntityState()
        es.register_machine(_make_ticket_machine())
        await es.aregister_entity("ticket", "T-1")

        order = []

        async def first(**kwargs):
            order.append("first")

        async def second(**kwargs):
            order.append("second")

        es.register_handler("ticket", first)
        es.register_handler("ticket", second)
        await es.atransition("ticket", "T-1", "INVESTIGATING")

        assert order == ["first", "second"]

    @pytest.mark.asyncio()
    async def test_second_handler_failure_rolls_back_state(self):
        """Failure rolls state back, but earlier handlers are not compensated."""
        es = EntityState()
        es.register_machine(_make_ticket_machine())
        await es.aregister_entity("ticket", "T-1")

        side_effects = []

        async def first(**kwargs):
            side_effects.append("first-ran")

        async def second(**kwargs):
            raise RuntimeError("second handler failed")

        es.register_handler("ticket", first)
        es.register_handler("ticket", second)

        with pytest.raises(RuntimeError, match="second handler failed"):
            await es.atransition("ticket", "T-1", "INVESTIGATING")

        # State rolled back...
        state = await es.aget_state("ticket", "T-1")
        assert state["state"] == "OPEN"
        # ...but the first handler's side effect is NOT compensated
        assert side_effects == ["first-ran"]

    @pytest.mark.asyncio()
    async def test_mixed_payload_signatures(self):
        """Payload is delivered only to handlers that declare it."""
        es = EntityState()
        es.register_machine(_make_ticket_machine())
        await es.aregister_entity("ticket", "T-1")

        seen = {}

        async def payload_aware(
            entity_type, entity_id, from_state, to_state, reason, context, payload=None
        ):
            seen["payload_aware"] = payload

        async def legacy(entity_type, entity_id, from_state, to_state, reason, context):
            seen["legacy"] = "called"

        es.register_handler("ticket", payload_aware)
        es.register_handler("ticket", legacy)
        await es.atransition("ticket", "T-1", "INVESTIGATING", payload={"assignee": "alice"})

        assert seen["payload_aware"] == {"assignee": "alice"}
        assert seen["legacy"] == "called"

    @pytest.mark.asyncio()
    async def test_re_registration_appends_not_replaces(self):
        es = EntityState()
        es.register_machine(_make_ticket_machine())
        await es.aregister_entity("ticket", "T-1")

        calls = []

        async def handler_a(**kwargs):
            calls.append("a")

        async def handler_b(**kwargs):
            calls.append("b")

        es.register_handler("ticket", handler_a)
        es.register_handler("ticket", handler_b)

        assert len(es._transition_handlers["ticket"]) == 2

        await es.atransition("ticket", "T-1", "INVESTIGATING")
        assert calls == ["a", "b"]
