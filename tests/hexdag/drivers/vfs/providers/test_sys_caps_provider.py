"""Tests for SysCapsProvider (/sys/caps/)."""

from __future__ import annotations

import json

import pytest

from hexdag.drivers.vfs.providers.sys_caps_provider import SysCapsProvider
from hexdag.kernel.domain.caps import ALL_CAPABILITIES, CapSet
from hexdag.kernel.domain.vfs import EntryType
from hexdag.kernel.exceptions import VFSError


@pytest.fixture
def provider() -> SysCapsProvider:
    """Provider with a restricted CapSet and sample profiles."""
    caps = CapSet(
        _allowed=frozenset({"vas.read", "port.llm"}),
        _denied=frozenset({"proc.spawn"}),
    )
    profiles = {
        "read-only": {"allow": ["vas.read", "vas.list"], "deny": ["vas.exec"]},
        "admin": {"allow": ["*"]},
    }
    return SysCapsProvider(cap_set=caps, profiles=profiles)


class TestSysCapsProviderRead:
    """Tests for read()."""

    async def test_read_current(self, provider: SysCapsProvider) -> None:
        result = json.loads(await provider.read("current"))
        assert "vas.read" in result["allowed"]
        assert "port.llm" in result["allowed"]
        assert "proc.spawn" in result["denied"]

    async def test_read_taxonomy(self, provider: SysCapsProvider) -> None:
        result = json.loads(await provider.read("taxonomy"))
        cap_names = {entry["capability"] for entry in result}
        assert cap_names == ALL_CAPABILITIES

    async def test_read_profile(self, provider: SysCapsProvider) -> None:
        result = json.loads(await provider.read("profiles/read-only"))
        assert result["allow"] == ["vas.read", "vas.list"]
        assert result["deny"] == ["vas.exec"]

    async def test_read_profile_not_found(self, provider: SysCapsProvider) -> None:
        with pytest.raises(VFSError, match="profile not found"):
            await provider.read("profiles/nonexistent")

    async def test_read_unknown_path(self, provider: SysCapsProvider) -> None:
        with pytest.raises(VFSError, match="path not found"):
            await provider.read("unknown")


class TestSysCapsProviderReaddir:
    """Tests for readdir()."""

    async def test_readdir_root(self, provider: SysCapsProvider) -> None:
        entries = await provider.readdir("")
        names = {e.name for e in entries}
        assert names == {"current", "taxonomy", "profiles"}

    async def test_readdir_profiles(self, provider: SysCapsProvider) -> None:
        entries = await provider.readdir("profiles")
        names = {e.name for e in entries}
        assert names == {"admin", "read-only"}

    async def test_readdir_unknown(self, provider: SysCapsProvider) -> None:
        with pytest.raises(VFSError, match="not a directory"):
            await provider.readdir("current")


class TestSysCapsProviderStat:
    """Tests for stat()."""

    async def test_stat_root(self, provider: SysCapsProvider) -> None:
        result = await provider.stat("")
        assert result.entry_type == EntryType.DIRECTORY
        assert result.child_count == 3

    async def test_stat_current(self, provider: SysCapsProvider) -> None:
        result = await provider.stat("current")
        assert result.entry_type == EntryType.FILE

    async def test_stat_taxonomy(self, provider: SysCapsProvider) -> None:
        result = await provider.stat("taxonomy")
        assert result.entry_type == EntryType.FILE

    async def test_stat_profiles_dir(self, provider: SysCapsProvider) -> None:
        result = await provider.stat("profiles")
        assert result.entry_type == EntryType.DIRECTORY
        assert result.child_count == 2

    async def test_stat_profile_entry(self, provider: SysCapsProvider) -> None:
        result = await provider.stat("profiles/read-only")
        assert result.entry_type == EntryType.FILE

    async def test_stat_unknown(self, provider: SysCapsProvider) -> None:
        with pytest.raises(VFSError, match="path not found"):
            await provider.stat("nonexistent")


class TestSysCapsProviderMutation:
    """Tests for set_caps() and set_profiles()."""

    async def test_set_caps(self) -> None:
        provider = SysCapsProvider()
        new_caps = CapSet(_allowed=frozenset({"mem.read"}))
        provider.set_caps(new_caps)
        result = json.loads(await provider.read("current"))
        assert result["allowed"] == ["mem.read"]

    async def test_set_profiles(self) -> None:
        provider = SysCapsProvider()
        provider.set_profiles({"test": {"allow": ["vas.read"]}})
        entries = await provider.readdir("profiles")
        assert len(entries) == 1
        assert entries[0].name == "test"
