"""Tests for the Notification port and its stdlib adapters."""

from __future__ import annotations

import pytest

from hexdag.kernel.ports.notification import Notification
from hexdag.stdlib.adapters.mock import MockNotification
from hexdag.stdlib.adapters.notification import ConsoleNotificationAdapter


class TestProtocolConformance:
    def test_console_adapter_satisfies_port(self) -> None:
        assert isinstance(ConsoleNotificationAdapter(), Notification)

    def test_mock_adapter_satisfies_port(self) -> None:
        assert isinstance(MockNotification(), Notification)


class TestConsoleNotificationAdapter:
    @pytest.mark.asyncio()
    async def test_asend_records_notification(self) -> None:
        adapter = ConsoleNotificationAdapter()
        await adapter.asend(
            "Order 42 needs approval",
            title="Approval required",
            channel="#approvals",
            metadata={"event_key": "approval:42"},
        )
        assert adapter.sent == [
            {
                "message": "Order 42 needs approval",
                "title": "Approval required",
                "channel": "#approvals",
                "metadata": {"event_key": "approval:42"},
            }
        ]

    @pytest.mark.asyncio()
    async def test_asend_minimal(self) -> None:
        adapter = ConsoleNotificationAdapter()
        await adapter.asend("hello")
        assert adapter.sent[0]["message"] == "hello"
        assert adapter.sent[0]["title"] is None
        assert adapter.sent[0]["metadata"] == {}


class TestMockNotification:
    @pytest.mark.asyncio()
    async def test_records_and_resets(self) -> None:
        mock = MockNotification()
        await mock.asend("one")
        await mock.asend("two", channel="email")
        assert [s["message"] for s in mock.sent] == ["one", "two"]
        mock.reset()
        assert mock.sent == []

    @pytest.mark.asyncio()
    async def test_simulated_failure(self) -> None:
        mock = MockNotification(should_raise=True)
        with pytest.raises(RuntimeError, match="delivery failure"):
            await mock.asend("boom")


class TestAliasRegistration:
    def test_adapters_resolvable_by_alias(self) -> None:
        from hexdag.stdlib.adapters._discovery import discover_adapter_aliases

        aliases = discover_adapter_aliases()
        assert (
            aliases["notification:console"]
            == "hexdag.stdlib.adapters.notification.ConsoleNotificationAdapter"
        )
        assert aliases["notification:mock"] == "hexdag.stdlib.adapters.mock.MockNotification"
        assert "ConsoleNotificationAdapter" in aliases
        assert "MockNotification" in aliases
