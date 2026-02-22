"""Tests for LocalVFS driver."""

from __future__ import annotations

import pytest

from hexdag.drivers.vfs.local import LocalVFS
from hexdag.kernel.domain.vfs import DirEntry, EntryType, StatResult
from hexdag.kernel.exceptions import VFSError
from hexdag.kernel.ports.vfs import VFS


class StubProvider:
    """Stub VFS provider for testing LocalVFS routing."""

    def __init__(self, name: str = "stub") -> None:
        self._name = name
        self.read_calls: list[str] = []
        self.readdir_calls: list[str] = []
        self.stat_calls: list[str] = []

    async def read(self, relative_path: str) -> str:
        self.read_calls.append(relative_path)
        return f'{{"provider": "{self._name}", "path": "{relative_path}"}}'

    async def readdir(self, relative_path: str) -> list[DirEntry]:
        self.readdir_calls.append(relative_path)
        return [
            DirEntry(
                name=f"entry_{self._name}",
                entry_type=EntryType.FILE,
                path=f"/stub/{relative_path}",
            )
        ]

    async def stat(self, relative_path: str) -> StatResult:
        self.stat_calls.append(relative_path)
        return StatResult(
            path=f"/stub/{relative_path}",
            entry_type=EntryType.FILE,
            description=f"Stub {self._name}",
        )


class TestProtocol:
    def test_local_vfs_satisfies_protocol(self) -> None:
        vfs = LocalVFS()
        assert isinstance(vfs, VFS)


class TestMount:
    def test_mount_provider(self) -> None:
        vfs = LocalVFS()
        provider = StubProvider()
        vfs.mount("/lib/", provider)
        assert "/lib/" in vfs.mounts()

    def test_mount_auto_adds_trailing_slash(self) -> None:
        vfs = LocalVFS()
        provider = StubProvider()
        vfs.mount("/lib", provider)
        assert "/lib/" in vfs.mounts()

    def test_mount_rejects_non_absolute(self) -> None:
        vfs = LocalVFS()
        provider = StubProvider()
        with pytest.raises(ValueError, match="must start with '/'"):
            vfs.mount("lib/", provider)

    def test_mounts_returns_copy(self) -> None:
        vfs = LocalVFS()
        provider = StubProvider()
        vfs.mount("/lib/", provider)
        mounts = vfs.mounts()
        mounts.clear()
        assert len(vfs.mounts()) == 1

    def test_multiple_mounts(self) -> None:
        vfs = LocalVFS()
        vfs.mount("/lib/", StubProvider("lib"))
        vfs.mount("/proc/runs/", StubProvider("runs"))
        vfs.mount("/sys/", StubProvider("sys"))
        assert len(vfs.mounts()) == 3


class TestResolve:
    def test_longest_prefix_match(self) -> None:
        vfs = LocalVFS()
        lib_provider = StubProvider("lib")
        nodes_provider = StubProvider("nodes")
        vfs.mount("/lib/", lib_provider)
        vfs.mount("/lib/nodes/", nodes_provider)

        # /lib/nodes/llm_node should match /lib/nodes/ (longer prefix)
        provider, relative = vfs._resolve("/lib/nodes/llm_node")
        assert provider is nodes_provider
        assert relative == "llm_node"

    def test_strips_prefix(self) -> None:
        vfs = LocalVFS()
        provider = StubProvider()
        vfs.mount("/proc/runs/", provider)

        _, relative = vfs._resolve("/proc/runs/abc123")
        assert relative == "abc123"

    def test_raises_for_unmounted_path(self) -> None:
        vfs = LocalVFS()
        vfs.mount("/lib/", StubProvider())
        with pytest.raises(VFSError, match="no provider mounted"):
            vfs._resolve("/unknown/path")

    def test_raises_for_relative_path(self) -> None:
        vfs = LocalVFS()
        with pytest.raises(VFSError, match="must be absolute"):
            vfs._resolve("relative/path")


class TestAread:
    @pytest.mark.asyncio()
    async def test_delegates_to_provider(self) -> None:
        vfs = LocalVFS()
        provider = StubProvider("lib")
        vfs.mount("/lib/", provider)

        result = await vfs.aread("/lib/nodes/llm_node")
        assert '"lib"' in result
        assert provider.read_calls == ["nodes/llm_node"]

    @pytest.mark.asyncio()
    async def test_raises_for_root(self) -> None:
        vfs = LocalVFS()
        with pytest.raises(VFSError, match="cannot read root"):
            await vfs.aread("/")

    @pytest.mark.asyncio()
    async def test_raises_for_unmounted(self) -> None:
        vfs = LocalVFS()
        with pytest.raises(VFSError, match="no provider mounted"):
            await vfs.aread("/unknown")


class TestAlist:
    @pytest.mark.asyncio()
    async def test_root_listing(self) -> None:
        vfs = LocalVFS()
        vfs.mount("/lib/", StubProvider())
        vfs.mount("/proc/runs/", StubProvider())
        vfs.mount("/sys/", StubProvider())

        entries = await vfs.alist("/")
        names = [e.name for e in entries]
        assert "lib" in names
        assert "proc" in names
        assert "sys" in names

    @pytest.mark.asyncio()
    async def test_root_deduplicates_prefixes(self) -> None:
        vfs = LocalVFS()
        vfs.mount("/proc/runs/", StubProvider())
        vfs.mount("/proc/scheduled/", StubProvider())
        vfs.mount("/proc/entities/", StubProvider())

        entries = await vfs.alist("/")
        names = [e.name for e in entries]
        # "proc" should appear only once
        assert names.count("proc") == 1

    @pytest.mark.asyncio()
    async def test_delegates_to_provider(self) -> None:
        vfs = LocalVFS()
        provider = StubProvider()
        vfs.mount("/lib/", provider)

        entries = await vfs.alist("/lib/nodes")
        assert len(entries) == 1
        assert provider.readdir_calls == ["nodes"]

    @pytest.mark.asyncio()
    async def test_root_entries_are_directories(self) -> None:
        vfs = LocalVFS()
        vfs.mount("/lib/", StubProvider())

        entries = await vfs.alist("/")
        for entry in entries:
            assert entry.entry_type == EntryType.DIRECTORY


class TestAstat:
    @pytest.mark.asyncio()
    async def test_root_stat(self) -> None:
        vfs = LocalVFS()
        vfs.mount("/lib/", StubProvider())
        vfs.mount("/sys/", StubProvider())

        stat = await vfs.astat("/")
        assert stat.path == "/"
        assert stat.entry_type == EntryType.DIRECTORY
        assert stat.child_count == 2

    @pytest.mark.asyncio()
    async def test_delegates_to_provider(self) -> None:
        vfs = LocalVFS()
        provider = StubProvider()
        vfs.mount("/sys/", provider)

        await vfs.astat("/sys/version")
        assert provider.stat_calls == ["version"]
