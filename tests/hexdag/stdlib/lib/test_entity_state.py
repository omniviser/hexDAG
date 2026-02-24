"""Tests for EntityState lib and domain models."""

from __future__ import annotations

import pytest

from hexdag.kernel.domain.entity_state import StateMachineConfig, StateTransition
from hexdag.kernel.exceptions import ValidationError
from hexdag.stdlib.adapters.memory.collection_memory import InMemoryCollectionStorage
from hexdag.stdlib.lib.entity_state import EntityState, InvalidTransitionError
from hexdag.stdlib.lib_base import HexDAGLib

# ---------------------------------------------------------------------------
# Domain — StateMachineConfig
# ---------------------------------------------------------------------------


def _order_config() -> StateMachineConfig:
    return StateMachineConfig(
        entity_type="order",
        states={"new", "processing", "shipped", "delivered", "cancelled"},
        initial_state="new",
        transitions={
            "new": {"processing", "cancelled"},
            "processing": {"shipped", "cancelled"},
            "shipped": {"delivered"},
        },
    )


class TestStateMachineConfig:
    def test_valid_config(self) -> None:
        config = _order_config()
        assert config.entity_type == "order"
        assert config.initial_state == "new"

    def test_initial_state_not_in_states(self) -> None:
        with pytest.raises(ValidationError, match="initial_state"):
            StateMachineConfig(
                entity_type="x",
                states={"a", "b"},
                initial_state="c",
                transitions={},
            )

    def test_transition_source_not_in_states(self) -> None:
        with pytest.raises(ValidationError, match="source"):
            StateMachineConfig(
                entity_type="x",
                states={"a", "b"},
                initial_state="a",
                transitions={"c": {"b"}},
            )

    def test_transition_target_not_in_states(self) -> None:
        with pytest.raises(ValidationError, match="targets"):
            StateMachineConfig(
                entity_type="x",
                states={"a", "b"},
                initial_state="a",
                transitions={"a": {"z"}},
            )

    def test_is_valid_transition(self) -> None:
        config = _order_config()
        assert config.is_valid_transition("new", "processing") is True
        assert config.is_valid_transition("new", "cancelled") is True
        assert config.is_valid_transition("new", "delivered") is False
        assert config.is_valid_transition("shipped", "delivered") is True

    def test_is_valid_transition_unknown_source(self) -> None:
        config = _order_config()
        assert config.is_valid_transition("unknown", "processing") is False


class TestStateTransition:
    def test_defaults(self) -> None:
        t = StateTransition(
            entity_type="order",
            entity_id="123",
            from_state="new",
            to_state="processing",
        )
        assert t.entity_type == "order"
        assert t.from_state == "new"
        assert t.to_state == "processing"
        assert t.timestamp > 0
        assert t.metadata == {}


# ---------------------------------------------------------------------------
# EntityState lib — tool surface
# ---------------------------------------------------------------------------


class TestEntityStateIsLib:
    def test_is_hexdag_lib(self) -> None:
        assert isinstance(EntityState(), HexDAGLib)

    def test_exposed_tools(self) -> None:
        lib = EntityState()
        tools = lib.get_tools()
        assert "aget_state" in tools
        assert "atransition" in tools
        assert "aget_history" in tools
        assert "aregister_entity" in tools
        assert len(tools) == 4


class TestRegisterEntity:
    @pytest.mark.asyncio()
    async def test_register_with_machine(self) -> None:
        lib = EntityState()
        lib.register_machine(_order_config())
        result = await lib.aregister_entity("order", "o-1")
        assert result["entity_type"] == "order"
        assert result["entity_id"] == "o-1"
        assert result["state"] == "new"  # uses machine's initial_state

    @pytest.mark.asyncio()
    async def test_register_with_override_initial(self) -> None:
        lib = EntityState()
        lib.register_machine(_order_config())
        result = await lib.aregister_entity("order", "o-1", initial_state="processing")
        assert result["state"] == "processing"

    @pytest.mark.asyncio()
    async def test_register_invalid_initial_state(self) -> None:
        lib = EntityState()
        lib.register_machine(_order_config())
        with pytest.raises(InvalidTransitionError, match="not valid"):
            await lib.aregister_entity("order", "o-1", initial_state="nonexistent")

    @pytest.mark.asyncio()
    async def test_register_without_machine_uses_created(self) -> None:
        lib = EntityState()
        result = await lib.aregister_entity("custom", "c-1")
        assert result["state"] == "created"


class TestTransition:
    @pytest.mark.asyncio()
    async def test_valid_transition(self) -> None:
        lib = EntityState()
        lib.register_machine(_order_config())
        await lib.aregister_entity("order", "o-1")
        result = await lib.atransition("order", "o-1", "processing")
        assert result["from_state"] == "new"
        assert result["to_state"] == "processing"

    @pytest.mark.asyncio()
    async def test_invalid_transition_raises(self) -> None:
        lib = EntityState()
        lib.register_machine(_order_config())
        await lib.aregister_entity("order", "o-1")
        with pytest.raises(InvalidTransitionError, match="Cannot transition"):
            await lib.atransition("order", "o-1", "delivered")

    @pytest.mark.asyncio()
    async def test_transition_unknown_entity_raises(self) -> None:
        lib = EntityState()
        with pytest.raises(InvalidTransitionError, match="not registered"):
            await lib.atransition("order", "missing", "processing")

    @pytest.mark.asyncio()
    async def test_transition_to_invalid_state_raises(self) -> None:
        lib = EntityState()
        lib.register_machine(_order_config())
        await lib.aregister_entity("order", "o-1")
        with pytest.raises(InvalidTransitionError, match="Cannot transition"):
            await lib.atransition("order", "o-1", "nonexistent")

    @pytest.mark.asyncio()
    async def test_multi_step_transition(self) -> None:
        lib = EntityState()
        lib.register_machine(_order_config())
        await lib.aregister_entity("order", "o-1")
        await lib.atransition("order", "o-1", "processing")
        await lib.atransition("order", "o-1", "shipped")
        result = await lib.atransition("order", "o-1", "delivered")
        assert result["from_state"] == "shipped"
        assert result["to_state"] == "delivered"

    @pytest.mark.asyncio()
    async def test_transition_without_machine_is_freeform(self) -> None:
        """Without a registered machine, any transition is allowed."""
        lib = EntityState()
        await lib.aregister_entity("custom", "c-1")
        result = await lib.atransition("custom", "c-1", "anything")
        assert result["to_state"] == "anything"

    @pytest.mark.asyncio()
    async def test_transition_with_reason(self) -> None:
        lib = EntityState()
        lib.register_machine(_order_config())
        await lib.aregister_entity("order", "o-1")
        await lib.atransition("order", "o-1", "cancelled", reason="customer request")
        history = await lib.aget_history("order", "o-1")
        last = history[-1]
        assert last["metadata"]["reason"] == "customer request"


class TestGetState:
    @pytest.mark.asyncio()
    async def test_get_existing(self) -> None:
        lib = EntityState()
        await lib.aregister_entity("order", "o-1")
        result = await lib.aget_state("order", "o-1")
        assert result is not None
        assert result["entity_type"] == "order"
        assert result["entity_id"] == "o-1"
        assert result["state"] == "created"

    @pytest.mark.asyncio()
    async def test_get_missing_returns_none(self) -> None:
        lib = EntityState()
        assert await lib.aget_state("order", "missing") is None

    @pytest.mark.asyncio()
    async def test_get_after_transition(self) -> None:
        lib = EntityState()
        lib.register_machine(_order_config())
        await lib.aregister_entity("order", "o-1")
        await lib.atransition("order", "o-1", "processing")
        result = await lib.aget_state("order", "o-1")
        assert result is not None
        assert result["state"] == "processing"


class TestGetHistory:
    @pytest.mark.asyncio()
    async def test_history_after_transitions(self) -> None:
        lib = EntityState()
        lib.register_machine(_order_config())
        await lib.aregister_entity("order", "o-1")
        await lib.atransition("order", "o-1", "processing")
        await lib.atransition("order", "o-1", "shipped")
        history = await lib.aget_history("order", "o-1")
        assert len(history) == 3  # initial + 2 transitions
        assert history[0]["from_state"] is None
        assert history[0]["to_state"] == "new"
        assert history[1]["from_state"] == "new"
        assert history[1]["to_state"] == "processing"
        assert history[2]["from_state"] == "processing"
        assert history[2]["to_state"] == "shipped"

    @pytest.mark.asyncio()
    async def test_history_empty_for_unknown(self) -> None:
        lib = EntityState()
        assert await lib.aget_history("order", "missing") == []

    @pytest.mark.asyncio()
    async def test_history_timestamps_ordered(self) -> None:
        lib = EntityState()
        lib.register_machine(_order_config())
        await lib.aregister_entity("order", "o-1")
        await lib.atransition("order", "o-1", "processing")
        history = await lib.aget_history("order", "o-1")
        assert history[0]["timestamp"] <= history[1]["timestamp"]


# ---------------------------------------------------------------------------
# EntityState — with SupportsCollectionStorage
# ---------------------------------------------------------------------------


class TestEntityStateWithStorage:
    @pytest.mark.asyncio()
    async def test_register_persists_state(self) -> None:
        storage = InMemoryCollectionStorage()
        lib = EntityState(storage=storage)
        lib.register_machine(_order_config())
        await lib.aregister_entity("order", "o-1")
        doc = await storage.aload("entity_states", "order:o-1")
        assert doc is not None
        assert doc["state"] == "new"
        assert doc["entity_type"] == "order"

    @pytest.mark.asyncio()
    async def test_register_persists_history(self) -> None:
        storage = InMemoryCollectionStorage()
        lib = EntityState(storage=storage)
        lib.register_machine(_order_config())
        await lib.aregister_entity("order", "o-1")
        doc = await storage.aload("state_history", "order:o-1")
        assert doc is not None
        assert len(doc["transitions"]) == 1
        assert doc["transitions"][0]["from_state"] is None
        assert doc["transitions"][0]["to_state"] == "new"

    @pytest.mark.asyncio()
    async def test_transition_updates_storage(self) -> None:
        storage = InMemoryCollectionStorage()
        lib = EntityState(storage=storage)
        lib.register_machine(_order_config())
        await lib.aregister_entity("order", "o-1")
        await lib.atransition("order", "o-1", "processing")
        state_doc = await storage.aload("entity_states", "order:o-1")
        assert state_doc is not None
        assert state_doc["state"] == "processing"
        history_doc = await storage.aload("state_history", "order:o-1")
        assert history_doc is not None
        assert len(history_doc["transitions"]) == 2
        assert history_doc["transitions"][1]["from_state"] == "new"
        assert history_doc["transitions"][1]["to_state"] == "processing"

    @pytest.mark.asyncio()
    async def test_aget_state_falls_back_to_storage(self) -> None:
        storage = InMemoryCollectionStorage()
        await storage.asave(
            "entity_states",
            "ticket:t-1",
            {"entity_type": "ticket", "entity_id": "t-1", "state": "open"},
        )
        lib = EntityState(storage=storage)
        result = await lib.aget_state("ticket", "t-1")
        assert result is not None
        assert result["state"] == "open"

    @pytest.mark.asyncio()
    async def test_aget_history_falls_back_to_storage(self) -> None:
        storage = InMemoryCollectionStorage()
        await storage.asave(
            "state_history",
            "ticket:t-1",
            {
                "entity_type": "ticket",
                "entity_id": "t-1",
                "transitions": [
                    {
                        "entity_type": "ticket",
                        "entity_id": "t-1",
                        "from_state": None,
                        "to_state": "open",
                        "timestamp": 1000.0,
                        "metadata": {},
                    }
                ],
            },
        )
        lib = EntityState(storage=storage)
        history = await lib.aget_history("ticket", "t-1")
        assert len(history) == 1
        assert history[0]["to_state"] == "open"

    @pytest.mark.asyncio()
    async def test_backward_compat_no_storage(self) -> None:
        """EntityState without storage works exactly as before."""
        lib = EntityState()
        lib.register_machine(_order_config())
        await lib.aregister_entity("order", "o-1")
        await lib.atransition("order", "o-1", "processing")
        result = await lib.aget_state("order", "o-1")
        assert result is not None
        assert result["state"] == "processing"
        history = await lib.aget_history("order", "o-1")
        assert len(history) == 2
