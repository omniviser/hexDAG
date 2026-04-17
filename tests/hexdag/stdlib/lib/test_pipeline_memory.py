"""Tests for PipelineMemory service (Phase 1)."""

import pytest

from hexdag.kernel.service import Service
from hexdag.stdlib.lib.pipeline_memory import PipelineMemory


class TestPipelineMemoryIsService:
    def test_is_service(self):
        assert issubclass(PipelineMemory, Service)

    def test_has_tools(self):
        mem = PipelineMemory()
        tools = mem.get_tools()
        assert "get" in tools
        assert "set" in tools

    def test_has_steps(self):
        mem = PipelineMemory()
        steps = mem.get_steps()
        assert "get" in steps
        assert "set" in steps
        assert "update" in steps
        assert "snapshot" in steps


class TestPipelineMemoryOperations:
    @pytest.fixture()
    def mem(self):
        return PipelineMemory()

    @pytest.mark.asyncio()
    async def test_get_default(self, mem):
        result = await mem.get("missing")
        assert result is None

    @pytest.mark.asyncio()
    async def test_get_with_default(self, mem):
        result = await mem.get("missing", default="fallback")
        assert result == "fallback"

    @pytest.mark.asyncio()
    async def test_set_and_get(self, mem):
        await mem.set("key1", "value1")
        result = await mem.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio()
    async def test_set_returns_confirmation(self, mem):
        result = await mem.set("k", "v")
        assert result == {"key": "k", "stored": True}

    @pytest.mark.asyncio()
    async def test_update(self, mem):
        await mem.set("existing", "old")
        result = await mem.update({"existing": "new", "added": "val"})
        assert result == {"keys_updated": 2}
        assert await mem.get("existing") == "new"
        assert await mem.get("added") == "val"

    @pytest.mark.asyncio()
    async def test_snapshot(self, mem):
        await mem.set("a", 1)
        await mem.set("b", 2)
        snap = await mem.snapshot()
        assert snap == {"a": 1, "b": 2}

    @pytest.mark.asyncio()
    async def test_snapshot_is_copy(self, mem):
        await mem.set("a", 1)
        snap = await mem.snapshot()
        snap["a"] = 999
        assert await mem.get("a") == 1  # Original unchanged

    @pytest.mark.asyncio()
    async def test_overwrite(self, mem):
        await mem.set("k", "v1")
        await mem.set("k", "v2")
        assert await mem.get("k") == "v2"

    @pytest.mark.asyncio()
    async def test_complex_values(self, mem):
        await mem.set("list", [1, 2, 3])
        await mem.set("dict", {"nested": True})
        assert await mem.get("list") == [1, 2, 3]
        assert await mem.get("dict") == {"nested": True}
