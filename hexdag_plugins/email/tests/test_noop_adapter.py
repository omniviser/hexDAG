"""Tests for Noop email adapter."""

import pytest

from hexdag_plugins._ports import SendEmailRequest
from hexdag_plugins.email.adapters.noop import NoopEmailAdapter


@pytest.fixture
def noop_adapter():
    return NoopEmailAdapter()


@pytest.mark.asyncio
class TestSendEmail:
    async def test_returns_success(self, noop_adapter):
        request = SendEmailRequest(
            to_addresses=["test@example.com"],
            subject="Test Subject",
            body_text="Hello, World!",
        )
        result = await noop_adapter.send_email(request)

        assert result.success is True
        assert result.message_id.startswith("noop-")
        assert result.sent_at is not None
        assert result.error_message is None

    async def test_preserves_conversation_id(self, noop_adapter):
        request = SendEmailRequest(
            to_addresses=["test@example.com"],
            subject="Test",
            body_text="Hello",
            conversation_id="conv-123",
        )
        result = await noop_adapter.send_email(request)
        assert result.conversation_id == "conv-123"


@pytest.mark.asyncio
class TestSendEmailFromDict:
    async def test_dict_request(self, noop_adapter):
        result = await noop_adapter.send_email_from_dict(
            {
                "to_addresses": ["test@example.com"],
                "subject": "Test",
                "body_text": "Hello",
            }
        )
        assert result["success"] is True
        assert result["message_id"].startswith("noop-")
        assert result["sent_at"] is not None
        assert result["error_message"] is None


@pytest.mark.asyncio
async def test_fetch_emails_returns_empty(noop_adapter):
    result = await noop_adapter.fetch_emails()
    assert result == []


@pytest.mark.asyncio
async def test_fetch_emails_with_params(noop_adapter):
    from datetime import UTC, datetime

    result = await noop_adapter.fetch_emails(since=datetime.now(UTC), limit=10)
    assert result == []
