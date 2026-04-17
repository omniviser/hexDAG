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
