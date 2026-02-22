"""Tests for LibProvider (/lib/ namespace)."""

from __future__ import annotations

import json

import pytest

from hexdag.drivers.vfs.providers.lib_provider import LibProvider
from hexdag.kernel.domain.vfs import EntryType
from hexdag.kernel.exceptions import VFSError


class TestReaddir:
    @pytest.mark.asyncio()
    async def test_root_lists_entity_types(self) -> None:
        provider = LibProvider()
        entries = await provider.readdir("")
        names = [e.name for e in entries]
        assert "nodes" in names
        assert "adapters" in names
        assert "macros" in names
        assert "tools" in names
        assert "tags" in names
        for entry in entries:
            assert entry.entry_type == EntryType.DIRECTORY

    @pytest.mark.asyncio()
    async def test_nodes_lists_builtin_nodes(self) -> None:
        provider = LibProvider()
        entries = await provider.readdir("nodes")
        names = [e.name for e in entries]
        # LLMNode should always be available
        assert "llm_node" in names
        for entry in entries:
            assert entry.entry_type == EntryType.FILE
            assert entry.path.startswith("/lib/nodes/")

    @pytest.mark.asyncio()
    async def test_adapters_lists_adapters(self) -> None:
        provider = LibProvider()
        entries = await provider.readdir("adapters")
        assert len(entries) > 0
        for entry in entries:
            assert entry.entry_type == EntryType.FILE
            assert entry.path.startswith("/lib/adapters/")

    @pytest.mark.asyncio()
    async def test_unknown_type_raises(self) -> None:
        provider = LibProvider()
        with pytest.raises(VFSError, match="not a directory"):
            await provider.readdir("unknown_type")


class TestRead:
    @pytest.mark.asyncio()
    async def test_read_node(self) -> None:
        provider = LibProvider()
        result = await provider.read("nodes/llm_node")
        data = json.loads(result)
        assert data["kind"] == "llm_node"
        assert "module_path" in data

    @pytest.mark.asyncio()
    async def test_read_node_schema(self) -> None:
        provider = LibProvider()
        result = await provider.read("nodes/llm_node/schema")
        data = json.loads(result)
        assert isinstance(data, dict)

    @pytest.mark.asyncio()
    async def test_read_unknown_node_raises(self) -> None:
        provider = LibProvider()
        with pytest.raises(VFSError, match="not found"):
            await provider.read("nodes/nonexistent_node_xyz")

    @pytest.mark.asyncio()
    async def test_read_unknown_type_raises(self) -> None:
        provider = LibProvider()
        with pytest.raises(VFSError, match="unknown entity type"):
            await provider.read("unknown_type/foo")

    @pytest.mark.asyncio()
    async def test_read_directory_raises(self) -> None:
        provider = LibProvider()
        with pytest.raises(VFSError, match="cannot read directory"):
            await provider.read("")


class TestStat:
    @pytest.mark.asyncio()
    async def test_stat_root(self) -> None:
        provider = LibProvider()
        stat = await provider.stat("")
        assert stat.path == "/lib"
        assert stat.entry_type == EntryType.DIRECTORY
        assert stat.child_count is not None
        assert stat.child_count > 0

    @pytest.mark.asyncio()
    async def test_stat_entity_type_directory(self) -> None:
        provider = LibProvider()
        stat = await provider.stat("nodes")
        assert stat.path == "/lib/nodes"
        assert stat.entry_type == EntryType.DIRECTORY
        assert stat.child_count is not None
        assert stat.child_count > 0

    @pytest.mark.asyncio()
    async def test_stat_specific_node(self) -> None:
        provider = LibProvider()
        stat = await provider.stat("nodes/llm_node")
        assert stat.path == "/lib/nodes/llm_node"
        assert stat.entry_type == EntryType.FILE
        assert stat.entity_type == "node"
        assert stat.module_path is not None
        assert stat.description is not None
        assert "read" in stat.capabilities

    @pytest.mark.asyncio()
    async def test_stat_node_has_builtin_tag(self) -> None:
        provider = LibProvider()
        stat = await provider.stat("nodes/llm_node")
        assert stat.tags.get("is_builtin") == "true"
