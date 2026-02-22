"""Tests for InMemoryMemory SupportsKeyValue methods (adelete, aexists, alist_keys)."""

from __future__ import annotations

import pytest

from hexdag.kernel.ports.data_store import SupportsKeyValue
from hexdag.stdlib.adapters.memory import InMemoryMemory


class TestSupportsKeyValueProtocol:
    """InMemoryMemory satisfies SupportsKeyValue at runtime."""

    def test_isinstance_check(self) -> None:
        mem = InMemoryMemory()
        assert isinstance(mem, SupportsKeyValue)


class TestAdelete:
    """Tests for InMemoryMemory.adelete()."""

    @pytest.mark.asyncio()
    async def test_delete_existing_key(self) -> None:
        mem = InMemoryMemory()
        await mem.aset("k", "v")
        assert await mem.adelete("k") is True
        assert await mem.aget("k") is None

    @pytest.mark.asyncio()
    async def test_delete_missing_key(self) -> None:
        mem = InMemoryMemory()
        assert await mem.adelete("missing") is False

    @pytest.mark.asyncio()
    async def test_delete_logs_history(self) -> None:
        mem = InMemoryMemory()
        await mem.aset("k", "v")
        await mem.adelete("k")
        ops = [h["operation"] for h in mem.get_access_history()]
        assert "delete" in ops

    @pytest.mark.asyncio()
    async def test_delete_with_delay(self) -> None:
        mem = InMemoryMemory(delay_seconds=0.001)
        await mem.aset("k", "v")
        assert await mem.adelete("k") is True


class TestAexists:
    """Tests for InMemoryMemory.aexists()."""

    @pytest.mark.asyncio()
    async def test_exists_true(self) -> None:
        mem = InMemoryMemory()
        await mem.aset("k", "v")
        assert await mem.aexists("k") is True

    @pytest.mark.asyncio()
    async def test_exists_false(self) -> None:
        mem = InMemoryMemory()
        assert await mem.aexists("k") is False

    @pytest.mark.asyncio()
    async def test_exists_after_delete(self) -> None:
        mem = InMemoryMemory()
        await mem.aset("k", "v")
        await mem.adelete("k")
        assert await mem.aexists("k") is False


class TestAlistKeys:
    """Tests for InMemoryMemory.alist_keys()."""

    @pytest.mark.asyncio()
    async def test_empty_store(self) -> None:
        mem = InMemoryMemory()
        assert await mem.alist_keys() == []

    @pytest.mark.asyncio()
    async def test_all_keys(self) -> None:
        mem = InMemoryMemory()
        await mem.aset("a", 1)
        await mem.aset("b", 2)
        keys = await mem.alist_keys()
        assert sorted(keys) == ["a", "b"]

    @pytest.mark.asyncio()
    async def test_prefix_filter(self) -> None:
        mem = InMemoryMemory()
        await mem.aset("user:1", "alice")
        await mem.aset("user:2", "bob")
        await mem.aset("order:1", "pizza")
        keys = await mem.alist_keys(prefix="user:")
        assert sorted(keys) == ["user:1", "user:2"]

    @pytest.mark.asyncio()
    async def test_prefix_no_match(self) -> None:
        mem = InMemoryMemory()
        await mem.aset("key", "val")
        assert await mem.alist_keys(prefix="nope:") == []

    @pytest.mark.asyncio()
    async def test_empty_prefix_returns_all(self) -> None:
        mem = InMemoryMemory()
        await mem.aset("x", 1)
        assert await mem.alist_keys(prefix="") == ["x"]
