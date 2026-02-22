"""Tests for VFS API module."""

from __future__ import annotations

import pytest

from hexdag.api import vfs as vfs_api
from hexdag.kernel.domain.vfs import DirEntry, EntryType, StatResult


class MockVFS:
    """Mock VFS for testing API functions."""

    async def aread(self, path: str) -> str:
        return f'{{"path": "{path}"}}'

    async def alist(self, path: str) -> list[DirEntry]:
        return [
            DirEntry(name="nodes", entry_type=EntryType.DIRECTORY, path="/lib/nodes"),
            DirEntry(name="adapters", entry_type=EntryType.DIRECTORY, path="/lib/adapters"),
        ]

    async def astat(self, path: str) -> StatResult:
        return StatResult(
            path=path,
            entry_type=EntryType.DIRECTORY,
            description="Test directory",
            child_count=5,
            capabilities=["read"],
        )

    def mount(self, prefix: str, provider: object) -> None:
        pass

    def mounts(self) -> dict:
        return {}


class TestReadPath:
    @pytest.mark.asyncio()
    async def test_returns_string(self) -> None:
        result = await vfs_api.read_path(MockVFS(), "/lib/nodes/llm_node")  # type: ignore[arg-type]
        assert isinstance(result, str)
        assert "/lib/nodes/llm_node" in result


class TestListPath:
    @pytest.mark.asyncio()
    async def test_returns_list_of_dicts(self) -> None:
        result = await vfs_api.list_path(MockVFS(), "/lib/")  # type: ignore[arg-type]
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "nodes"
        assert result[0]["entry_type"] == "directory"
        assert result[0]["path"] == "/lib/nodes"


class TestStatPath:
    @pytest.mark.asyncio()
    async def test_returns_dict(self) -> None:
        result = await vfs_api.stat_path(MockVFS(), "/lib/")  # type: ignore[arg-type]
        assert isinstance(result, dict)
        assert result["entry_type"] == "directory"
        assert result["child_count"] == 5
        assert result["capabilities"] == ["read"]

    @pytest.mark.asyncio()
    async def test_excludes_none_fields(self) -> None:
        result = await vfs_api.stat_path(MockVFS(), "/lib/")  # type: ignore[arg-type]
        # Fields that are None should not appear
        assert "entity_type" not in result
        assert "module_path" not in result
        assert "status" not in result
