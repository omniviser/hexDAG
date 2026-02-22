"""Tests for SysProvider (/sys/ namespace)."""

from __future__ import annotations

import json

import pytest

from hexdag.drivers.vfs.providers.sys_provider import SysProvider
from hexdag.kernel.domain.vfs import EntryType
from hexdag.kernel.exceptions import VFSError


class TestReaddir:
    @pytest.mark.asyncio()
    async def test_root_lists_entries(self) -> None:
        provider = SysProvider()
        entries = await provider.readdir("")
        names = [e.name for e in entries]
        assert "version" in names
        assert "config" in names

    @pytest.mark.asyncio()
    async def test_non_root_raises(self) -> None:
        provider = SysProvider()
        with pytest.raises(VFSError, match="not a directory"):
            await provider.readdir("version")


class TestRead:
    @pytest.mark.asyncio()
    async def test_read_version(self) -> None:
        provider = SysProvider()
        result = await provider.read("version")
        version = json.loads(result)
        assert isinstance(version, str)

    @pytest.mark.asyncio()
    async def test_read_config_empty(self) -> None:
        provider = SysProvider()
        result = await provider.read("config")
        data = json.loads(result)
        assert isinstance(data, dict)

    @pytest.mark.asyncio()
    async def test_read_config_with_data(self) -> None:
        provider = SysProvider(config={"env": "test", "debug": "true"})
        result = await provider.read("config")
        data = json.loads(result)
        assert data["env"] == "test"
        assert data["debug"] == "true"

    @pytest.mark.asyncio()
    async def test_read_unknown_raises(self) -> None:
        provider = SysProvider()
        with pytest.raises(VFSError, match="path not found"):
            await provider.read("unknown")

    @pytest.mark.asyncio()
    async def test_read_directory_raises(self) -> None:
        provider = SysProvider()
        with pytest.raises(VFSError, match="cannot read directory"):
            await provider.read("")


class TestStat:
    @pytest.mark.asyncio()
    async def test_stat_root(self) -> None:
        provider = SysProvider()
        stat = await provider.stat("")
        assert stat.path == "/sys"
        assert stat.entry_type == EntryType.DIRECTORY
        assert stat.child_count == 2

    @pytest.mark.asyncio()
    async def test_stat_version(self) -> None:
        provider = SysProvider()
        stat = await provider.stat("version")
        assert stat.path == "/sys/version"
        assert stat.entry_type == EntryType.FILE
        assert stat.description is not None

    @pytest.mark.asyncio()
    async def test_stat_config(self) -> None:
        provider = SysProvider()
        stat = await provider.stat("config")
        assert stat.path == "/sys/config"
        assert stat.entry_type == EntryType.FILE

    @pytest.mark.asyncio()
    async def test_stat_unknown_raises(self) -> None:
        provider = SysProvider()
        with pytest.raises(VFSError, match="path not found"):
            await provider.stat("unknown")
