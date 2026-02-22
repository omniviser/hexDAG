"""Tests for VFSTools lib."""

from __future__ import annotations

import pytest

from hexdag.kernel.domain.vfs import DirEntry, EntryType, StatResult
from hexdag.kernel.lib_base import HexDAGLib
from hexdag.stdlib.lib.vfs_tools import VFSTools


class MockVFS:
    """Mock VFS for testing VFSTools."""

    async def aread(self, path: str) -> str:
        return f'{{"path": "{path}"}}'

    async def alist(self, path: str) -> list[DirEntry]:
        return [DirEntry(name="test_entry", entry_type=EntryType.FILE, path=f"{path}/test_entry")]

    async def astat(self, path: str) -> StatResult:
        return StatResult(
            path=path,
            entry_type=EntryType.FILE,
            entity_type="node",
            description="Test node",
            capabilities=["read"],
        )

    def mount(self, prefix: str, provider: object) -> None:
        pass

    def mounts(self) -> dict:
        return {}


class TestIsLib:
    def test_is_hexdag_lib(self) -> None:
        tools = VFSTools(vfs=MockVFS())  # type: ignore[arg-type]
        assert isinstance(tools, HexDAGLib)


class TestExposedTools:
    def test_exposed_tools(self) -> None:
        tools = VFSTools(vfs=MockVFS())  # type: ignore[arg-type]
        tool_map = tools.get_tools()
        assert "aread_path" in tool_map
        assert "alist_directory" in tool_map
        assert "astat_path" in tool_map
        assert len(tool_map) == 3


class TestToolExecution:
    @pytest.mark.asyncio()
    async def test_aread_path(self) -> None:
        tools = VFSTools(vfs=MockVFS())  # type: ignore[arg-type]
        result = await tools.aread_path("/lib/nodes/llm_node")
        assert "/lib/nodes/llm_node" in result

    @pytest.mark.asyncio()
    async def test_alist_directory(self) -> None:
        tools = VFSTools(vfs=MockVFS())  # type: ignore[arg-type]
        result = await tools.alist_directory("/lib/nodes")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["name"] == "test_entry"
        assert result[0]["entry_type"] == "file"

    @pytest.mark.asyncio()
    async def test_astat_path(self) -> None:
        tools = VFSTools(vfs=MockVFS())  # type: ignore[arg-type]
        result = await tools.astat_path("/lib/nodes/llm_node")
        assert isinstance(result, dict)
        assert result["entity_type"] == "node"
        assert result["description"] == "Test node"
        assert result["capabilities"] == ["read"]
        # None fields excluded
        assert "module_path" not in result
        assert "status" not in result
