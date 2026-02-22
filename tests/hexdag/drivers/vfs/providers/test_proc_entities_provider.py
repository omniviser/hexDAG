"""Tests for ProcEntitiesProvider (/proc/entities/ namespace)."""

from __future__ import annotations

import json

import pytest

from hexdag.drivers.vfs.providers.proc_entities_provider import ProcEntitiesProvider
from hexdag.kernel.domain.entity_state import StateMachineConfig
from hexdag.kernel.domain.vfs import EntryType
from hexdag.kernel.exceptions import VFSError
from hexdag.stdlib.lib.entity_state import EntityState


def _make_entity_state() -> EntityState:
    """Create an EntityState pre-populated with a machine and entity."""
    es = EntityState()
    es.register_machine(
        StateMachineConfig(
            entity_type="order",
            states={"pending", "confirmed", "shipped", "delivered"},
            transitions={
                "pending": {"confirmed"},
                "confirmed": {"shipped"},
                "shipped": {"delivered"},
            },
            initial_state="pending",
        )
    )
    return es


class TestReaddir:
    @pytest.mark.asyncio()
    async def test_root_lists_entity_types(self) -> None:
        es = _make_entity_state()
        provider = ProcEntitiesProvider(es)
        entries = await provider.readdir("")
        names = [e.name for e in entries]
        assert "order" in names
        for entry in entries:
            assert entry.entry_type == EntryType.DIRECTORY

    @pytest.mark.asyncio()
    async def test_type_lists_entities(self) -> None:
        es = _make_entity_state()
        await es.aregister_entity("order", "ORD-001")
        await es.aregister_entity("order", "ORD-002")
        provider = ProcEntitiesProvider(es)
        entries = await provider.readdir("order")
        names = [e.name for e in entries]
        assert "ORD-001" in names
        assert "ORD-002" in names


class TestRead:
    @pytest.mark.asyncio()
    async def test_read_entity_state(self) -> None:
        es = _make_entity_state()
        await es.aregister_entity("order", "ORD-001")
        provider = ProcEntitiesProvider(es)
        result = await provider.read("order/ORD-001")
        data = json.loads(result)
        assert data["state"] == "pending"
        assert data["entity_type"] == "order"
        assert data["entity_id"] == "ORD-001"

    @pytest.mark.asyncio()
    async def test_read_entity_history(self) -> None:
        es = _make_entity_state()
        await es.aregister_entity("order", "ORD-001")
        await es.atransition("order", "ORD-001", "confirmed")
        provider = ProcEntitiesProvider(es)
        result = await provider.read("order/ORD-001/history")
        history = json.loads(result)
        assert isinstance(history, list)
        assert len(history) >= 1

    @pytest.mark.asyncio()
    async def test_read_unknown_entity_raises(self) -> None:
        es = _make_entity_state()
        provider = ProcEntitiesProvider(es)
        with pytest.raises(VFSError, match="not found"):
            await provider.read("order/NONEXISTENT")

    @pytest.mark.asyncio()
    async def test_read_missing_type_format_raises(self) -> None:
        es = _make_entity_state()
        provider = ProcEntitiesProvider(es)
        with pytest.raises(VFSError, match="expected format"):
            await provider.read("order")


class TestStat:
    @pytest.mark.asyncio()
    async def test_stat_root(self) -> None:
        es = _make_entity_state()
        provider = ProcEntitiesProvider(es)
        stat = await provider.stat("")
        assert stat.path == "/proc/entities"
        assert stat.entry_type == EntryType.DIRECTORY

    @pytest.mark.asyncio()
    async def test_stat_entity_type(self) -> None:
        es = _make_entity_state()
        await es.aregister_entity("order", "ORD-001")
        provider = ProcEntitiesProvider(es)
        stat = await provider.stat("order")
        assert stat.entry_type == EntryType.DIRECTORY
        assert stat.child_count == 1

    @pytest.mark.asyncio()
    async def test_stat_specific_entity(self) -> None:
        es = _make_entity_state()
        await es.aregister_entity("order", "ORD-001")
        provider = ProcEntitiesProvider(es)
        stat = await provider.stat("order/ORD-001")
        assert stat.entry_type == EntryType.FILE
        assert stat.entity_type == "entity"
        assert stat.status == "pending"
        assert stat.tags["entity_type"] == "order"
