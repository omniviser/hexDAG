"""Tests for TransitionNode — entity transitions with payload passthrough."""

import pytest

from hexdag.kernel.context.execution_context import set_services
from hexdag.kernel.domain.entity_state import StateMachineConfig
from hexdag.stdlib.lib.entity_state import EntityState
from hexdag.stdlib.nodes.transition_node import TransitionNode


@pytest.fixture(autouse=True)
def _clear_services():
    yield
    set_services(None)


def _make_entity_state() -> EntityState:
    es = EntityState()
    es.register_machine(
        StateMachineConfig(
            entity_type="ticket",
            states={"OPEN", "RESOLVED"},
            initial_state="OPEN",
            transitions={"OPEN": {"RESOLVED"}},
        )
    )
    return es


class TestTransitionNodeFactory:
    def test_dynamic_refs_go_to_input_mapping(self):
        spec = TransitionNode()(
            "resolve",
            entity="ticket",
            entity_id="$input.ticket_id",
            to_state="RESOLVED",
            payload={"resolved_by": "$input.user_id", "source": "api"},
        )
        mapping = spec.params["input_mapping"]
        assert mapping["entity_id"] == "$input.ticket_id"
        assert mapping["payload__resolved_by"] == "$input.user_id"
        # Literal payload values stay out of input_mapping
        assert "payload__source" not in mapping

    def test_payload_in_factory_params(self):
        payload = {"a": 1}
        spec = TransitionNode()(
            "resolve",
            entity="ticket",
            entity_id="T-1",
            to_state="RESOLVED",
            payload=payload,
        )
        assert spec.factory_params["payload"] == payload


class TestTransitionNodeExecution:
    @pytest.mark.asyncio()
    async def test_literal_payload_reaches_handler(self):
        es = _make_entity_state()
        await es.aregister_entity("ticket", "T-1")

        received = []

        async def handler(payload=None, **kwargs):
            received.append(payload)

        es.register_handler("ticket", handler)
        set_services({"entity_state": es})

        spec = TransitionNode()(
            "resolve",
            entity="ticket",
            entity_id="T-1",
            to_state="RESOLVED",
            payload={"source": "api", "attempt": 1},
        )
        result = await spec.fn({})

        assert result["to_state"] == "RESOLVED"
        assert received == [{"source": "api", "attempt": 1}]

    @pytest.mark.asyncio()
    async def test_dynamic_payload_resolved_from_inputs(self):
        es = _make_entity_state()
        await es.aregister_entity("ticket", "T-1")

        received = []

        async def handler(payload=None, **kwargs):
            received.append(payload)

        es.register_handler("ticket", handler)
        set_services({"entity_state": es})

        spec = TransitionNode()(
            "resolve",
            entity="ticket",
            entity_id="$input.ticket_id",
            to_state="RESOLVED",
            payload={"resolved_by": "$input.user_id", "source": "api"},
        )
        # Simulate ExecutionCoordinator having resolved input_mapping
        result = await spec.fn({"entity_id": "T-1", "payload__resolved_by": "user-9"})

        assert result["to_state"] == "RESOLVED"
        assert received == [{"resolved_by": "user-9", "source": "api"}]

    @pytest.mark.asyncio()
    async def test_no_payload_passes_none(self):
        es = _make_entity_state()
        await es.aregister_entity("ticket", "T-1")

        received = []

        async def handler(payload=None, **kwargs):
            received.append(payload)

        es.register_handler("ticket", handler)
        set_services({"entity_state": es})

        spec = TransitionNode()(
            "resolve",
            entity="ticket",
            entity_id="T-1",
            to_state="RESOLVED",
        )
        await spec.fn({})

        assert received == [None]
