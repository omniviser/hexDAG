"""Tests for MockEmailAdapter."""

from datetime import UTC, datetime, timedelta

import pytest

from hexdag.kernel.ports.messaging import SendEmailRequest, SupportsEmail
from hexdag.stdlib.adapters.mock import MockEmailAdapter


@pytest.fixture
def adapter():
    return MockEmailAdapter()


def _request(**overrides) -> SendEmailRequest:
    base = {
        "to_addresses": ["carrier@example.com"],
        "subject": "Rate confirmation",
        "body_text": "Please confirm.",
    }
    base.update(overrides)
    return SendEmailRequest(**base)


class TestSend:
    @pytest.mark.asyncio
    async def test_records_and_returns_deterministic_ids(self, adapter):
        r1 = await adapter.send_email(_request())
        r2 = await adapter.send_email(_request(subject="Follow-up"))

        assert r1.message_id == "mock-message-1"
        assert r2.message_id == "mock-message-2"
        assert [r.subject for r in adapter.sent_requests] == [
            "Rate confirmation",
            "Follow-up",
        ]

    @pytest.mark.asyncio
    async def test_preserves_conversation_id(self, adapter):
        result = await adapter.send_email(_request(conversation_id="conv-7"))
        assert result.conversation_id == "conv-7"

    @pytest.mark.asyncio
    async def test_should_raise(self):
        adapter = MockEmailAdapter(should_raise=True)
        with pytest.raises(RuntimeError, match="delivery failure"):
            await adapter.send_email(_request())

    @pytest.mark.asyncio
    async def test_send_from_dict(self, adapter):
        result = await adapter.send_email_from_dict({
            "to_addresses": ["a@example.com"],
            "subject": "Hi",
            "body_text": "Hello",
        })
        assert result["message_id"] == "mock-message-1"
        assert result["success"] is True


class TestInbox:
    @pytest.mark.asyncio
    async def test_scripted_inbox(self, adapter):
        adapter.queue_inbound({"message_id": "in-1", "subject": "Reply"})
        emails = await adapter.fetch_emails()
        assert emails == [{"message_id": "in-1", "subject": "Reply"}]

    @pytest.mark.asyncio
    async def test_since_filter(self, adapter):
        old = datetime.now(UTC) - timedelta(days=2)
        new = datetime.now(UTC)
        adapter.queue_inbound({"message_id": "old", "received_at": old})
        adapter.queue_inbound({"message_id": "new", "received_at": new})

        emails = await adapter.fetch_emails(since=datetime.now(UTC) - timedelta(days=1))
        assert [e["message_id"] for e in emails] == ["new"]

    @pytest.mark.asyncio
    async def test_reset(self, adapter):
        adapter.queue_inbound({"message_id": "x"})
        await adapter.send_email(_request())
        adapter.reset()

        assert adapter.sent_requests == []
        assert await adapter.fetch_emails() == []


class TestProtocolAndDiscovery:
    def test_implements_supports_email(self, adapter):
        assert isinstance(adapter, SupportsEmail)

    def test_discovery_alias(self):
        from hexdag.stdlib.adapters._discovery import discover_adapter_aliases

        aliases = discover_adapter_aliases()
        assert aliases["email:mock"] == "hexdag.stdlib.adapters.mock.MockEmailAdapter"
