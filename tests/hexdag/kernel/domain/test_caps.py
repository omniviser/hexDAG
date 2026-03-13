"""Tests for CapSet domain model."""

from __future__ import annotations

import pytest

from hexdag.kernel.config.models import DefaultCaps
from hexdag.kernel.domain.caps import ALL_CAPABILITIES, CapSet
from hexdag.kernel.exceptions import CapDeniedError


class TestCapSetAllows:
    """Tests for CapSet.allows()."""

    def test_allows_granted_cap(self) -> None:
        caps = CapSet(_allowed=frozenset({"port.llm", "vas.read"}))
        assert caps.allows("port.llm") is True
        assert caps.allows("vas.read") is True

    def test_denies_unganted_cap(self) -> None:
        caps = CapSet(_allowed=frozenset({"port.llm"}))
        assert caps.allows("port.data_store") is False

    def test_denied_overrides_allowed(self) -> None:
        caps = CapSet(
            _allowed=frozenset({"port.llm", "vas.read"}),
            _denied=frozenset({"port.llm"}),
        )
        assert caps.allows("port.llm") is False
        assert caps.allows("vas.read") is True


class TestCapSetIntersect:
    """Tests for CapSet.intersect()."""

    def test_intersect_narrows_allowed(self) -> None:
        parent = CapSet(_allowed=frozenset({"port.llm", "vas.read", "mem.read"}))
        child = CapSet(_allowed=frozenset({"port.llm", "vas.read", "vas.write"}))

        result = parent.intersect(child)
        assert result.allows("port.llm") is True
        assert result.allows("vas.read") is True
        assert result.allows("mem.read") is False  # not in child
        assert result.allows("vas.write") is False  # not in parent

    def test_intersect_denied_union(self) -> None:
        parent = CapSet(
            _allowed=ALL_CAPABILITIES,
            _denied=frozenset({"proc.spawn"}),
        )
        child = CapSet(
            _allowed=ALL_CAPABILITIES,
            _denied=frozenset({"vas.write"}),
        )

        result = parent.intersect(child)
        assert result.allows("proc.spawn") is False
        assert result.allows("vas.write") is False
        assert result.allows("port.llm") is True


class TestCapSetGrant:
    """Tests for CapSet.grant()."""

    def test_grant_adds_capability(self) -> None:
        caps = CapSet(_allowed=frozenset({"vas.read"}))
        new_caps = caps.grant("port.llm")
        assert new_caps.allows("port.llm") is True
        # Original unchanged (frozen)
        assert caps.allows("port.llm") is False

    def test_grant_denied_raises(self) -> None:
        caps = CapSet(
            _allowed=frozenset({"vas.read"}),
            _denied=frozenset({"port.llm"}),
        )
        with pytest.raises(CapDeniedError, match="port.llm"):
            caps.grant("port.llm")


class TestCapSetRevoke:
    """Tests for CapSet.revoke()."""

    def test_revoke_removes_capability(self) -> None:
        caps = CapSet(_allowed=frozenset({"vas.read", "port.llm"}))
        new_caps = caps.revoke("port.llm")
        assert new_caps.allows("port.llm") is False
        assert new_caps.allows("vas.read") is True

    def test_revoke_nonexistent_is_noop(self) -> None:
        caps = CapSet(_allowed=frozenset({"vas.read"}))
        new_caps = caps.revoke("port.llm")
        assert new_caps.allows("vas.read") is True


class TestCapSetFactories:
    """Tests for CapSet factory methods."""

    def test_unrestricted(self) -> None:
        caps = CapSet.unrestricted()
        for cap in ALL_CAPABILITIES:
            assert caps.allows(cap) is True

    def test_from_config_none_is_unrestricted(self) -> None:
        config = DefaultCaps()
        caps = CapSet.from_config(config)
        for cap in ALL_CAPABILITIES:
            assert caps.allows(cap) is True

    def test_from_config_with_default_set(self) -> None:
        config = DefaultCaps(default_set=["port.llm", "vas.read"])
        caps = CapSet.from_config(config)
        assert caps.allows("port.llm") is True
        assert caps.allows("vas.read") is True
        assert caps.allows("proc.spawn") is False

    def test_from_config_with_deny(self) -> None:
        config = DefaultCaps(deny=["proc.spawn", "vas.write"])
        caps = CapSet.from_config(config)
        assert caps.allows("port.llm") is True
        assert caps.allows("proc.spawn") is False
        assert caps.allows("vas.write") is False

    def test_from_profile(self) -> None:
        profile = {
            "allow": ["vas.read", "vas.list", "vas.stat"],
            "deny": ["vas.exec"],
        }
        caps = CapSet.from_profile(profile)
        assert caps.allows("vas.read") is True
        assert caps.allows("vas.list") is True
        assert caps.allows("vas.exec") is False
        assert caps.allows("port.llm") is False

    def test_from_profile_wildcard(self) -> None:
        profile = {"allow": ["*"], "deny": ["proc.spawn"]}
        caps = CapSet.from_profile(profile)
        assert caps.allows("port.llm") is True
        assert caps.allows("proc.spawn") is False


class TestCapSetProperties:
    """Tests for CapSet properties."""

    def test_allowed_property(self) -> None:
        caps = CapSet(
            _allowed=frozenset({"port.llm", "vas.read"}),
            _denied=frozenset({"port.llm"}),
        )
        assert caps.allowed == frozenset({"vas.read"})

    def test_denied_property(self) -> None:
        caps = CapSet(_denied=frozenset({"proc.spawn"}))
        assert caps.denied == frozenset({"proc.spawn"})

    def test_frozen(self) -> None:
        caps = CapSet(_allowed=frozenset({"vas.read"}))
        with pytest.raises(AttributeError):
            caps._allowed = frozenset()  # type: ignore[misc]


class TestAllCapabilities:
    """Tests for capability taxonomy constants."""

    def test_all_capabilities_is_frozenset(self) -> None:
        assert isinstance(ALL_CAPABILITIES, frozenset)

    def test_known_capabilities_present(self) -> None:
        expected = {
            "vas.read",
            "vas.write",
            "vas.exec",
            "vas.list",
            "vas.stat",
            "port.llm",
            "port.tool_router",
            "port.data_store",
            "proc.spawn",
            "entity.transition",
            "mem.read",
            "mem.write",
        }
        assert expected == ALL_CAPABILITIES
