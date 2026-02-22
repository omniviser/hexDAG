"""Tests for VFS port protocol."""

from __future__ import annotations

import pytest

from hexdag.kernel.domain.vfs import DirEntry, EntryType, StatResult
from hexdag.kernel.ports.vfs import VFS, VFSProvider


class MockVFSProvider:
    """Mock VFS provider for testing protocol conformance."""

    async def read(self, relative_path: str) -> str:
        return '{"mock": true}'

    async def readdir(self, relative_path: str) -> list[DirEntry]:
        return [DirEntry(name="test", entry_type=EntryType.FILE, path="/test")]

    async def stat(self, relative_path: str) -> StatResult:
        return StatResult(path=f"/{relative_path}", entry_type=EntryType.FILE)


class MockVFS:
    """Mock VFS for testing protocol conformance."""

    def __init__(self) -> None:
        self._mounts: dict[str, VFSProvider] = {}

    async def aread(self, path: str) -> str:
        return '{"mock": true}'

    async def alist(self, path: str) -> list[DirEntry]:
        return [DirEntry(name="root", entry_type=EntryType.DIRECTORY, path="/")]

    async def astat(self, path: str) -> StatResult:
        return StatResult(path=path, entry_type=EntryType.FILE)

    def mount(self, prefix: str, provider: VFSProvider) -> None:
        self._mounts[prefix] = provider

    def mounts(self) -> dict[str, VFSProvider]:
        return dict(self._mounts)


class TestVFSProviderProtocol:
    def test_mock_satisfies_protocol(self) -> None:
        provider = MockVFSProvider()
        assert isinstance(provider, VFSProvider)

    @pytest.mark.asyncio()
    async def test_read(self) -> None:
        provider = MockVFSProvider()
        result = await provider.read("test")
        assert isinstance(result, str)

    @pytest.mark.asyncio()
    async def test_readdir(self) -> None:
        provider = MockVFSProvider()
        result = await provider.readdir("")
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], DirEntry)

    @pytest.mark.asyncio()
    async def test_stat(self) -> None:
        provider = MockVFSProvider()
        result = await provider.stat("test")
        assert isinstance(result, StatResult)


class TestVFSProtocol:
    def test_mock_satisfies_protocol(self) -> None:
        vfs = MockVFS()
        assert isinstance(vfs, VFS)

    @pytest.mark.asyncio()
    async def test_aread(self) -> None:
        vfs = MockVFS()
        result = await vfs.aread("/test")
        assert isinstance(result, str)

    @pytest.mark.asyncio()
    async def test_alist(self) -> None:
        vfs = MockVFS()
        result = await vfs.alist("/")
        assert isinstance(result, list)

    @pytest.mark.asyncio()
    async def test_astat(self) -> None:
        vfs = MockVFS()
        result = await vfs.astat("/test")
        assert isinstance(result, StatResult)

    def test_mount_and_mounts(self) -> None:
        vfs = MockVFS()
        provider = MockVFSProvider()
        vfs.mount("/lib/", provider)
        assert "/lib/" in vfs.mounts()
