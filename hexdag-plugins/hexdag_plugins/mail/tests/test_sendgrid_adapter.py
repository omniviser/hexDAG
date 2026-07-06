"""Tests for SendGridEmailAdapter."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from hexdag.kernel.ports.messaging import SendEmailRequest, SupportsEmail

from hexdag_plugins.mail.adapters.sendgrid import SendGridEmailAdapter


@pytest.fixture
def adapter():
    return SendGridEmailAdapter(
        api_key="sg-test-key",
        from_address="dispatch@example.com",
        from_name="Dispatch",
    )


def _mock_client(status_code: int = 202, headers: dict | None = None) -> AsyncMock:
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.headers = headers or {"X-Message-Id": "sg-abc123"}
    response.text = "error body" if status_code != 202 else ""
    client = AsyncMock()
    client.post = AsyncMock(return_value=response)
    client.is_closed = False
    return client


def _request(**overrides) -> SendEmailRequest:
    base = {
        "to_addresses": ["carrier@example.com"],
        "subject": "Rate confirmation",
        "body_text": "Please confirm.",
    }
    base.update(overrides)
    return SendEmailRequest(**base)


class TestInitialization:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("SENDGRID_API_KEY", raising=False)
        with pytest.raises(ValueError, match="SENDGRID_API_KEY"):
            SendGridEmailAdapter(api_key="", from_address="x@example.com")

    def test_missing_from_address_raises(self, monkeypatch):
        monkeypatch.delenv("EMAIL_FROM_ADDRESS", raising=False)
        monkeypatch.delenv("EMAIL_USER_ID", raising=False)
        with pytest.raises(ValueError, match="from_address"):
            SendGridEmailAdapter(api_key="key", from_address="")

    def test_does_not_auto_quote_thread(self, adapter):
        assert adapter.auto_quotes_thread is False

    def test_implements_supports_email(self, adapter):
        assert isinstance(adapter, SupportsEmail)


class TestSend:
    async def test_success_returns_sendgrid_message_id(self, adapter):
        adapter._client = _mock_client()

        result = await adapter.send_email(_request())

        assert result.success is True
        assert result.message_id == "sendgrid-sg-abc123"

    async def test_threading_headers(self, adapter):
        adapter._client = _mock_client()

        await adapter.send_email(
            _request(in_reply_to="<msg-0@example.com>", conversation_id="conv-9")
        )

        payload = adapter._client.post.call_args.kwargs["json"]
        assert payload["headers"]["In-Reply-To"] == "<msg-0@example.com>"
        assert payload["headers"]["References"] == "<msg-0@example.com>"
        assert payload["headers"]["X-Entity-Ref-ID"] == "conv-9"

    async def test_no_threading_headers_for_new_email(self, adapter):
        adapter._client = _mock_client()

        await adapter.send_email(_request())

        payload = adapter._client.post.call_args.kwargs["json"]
        assert "headers" not in payload

    async def test_payload_shape(self, adapter):
        adapter._client = _mock_client()

        await adapter.send_email(_request(cc_addresses=["ops@example.com"], body_html="<p>Hi</p>"))

        payload = adapter._client.post.call_args.kwargs["json"]
        assert payload["from"] == {"email": "dispatch@example.com", "name": "Dispatch"}
        assert payload["personalizations"][0]["to"] == [{"email": "carrier@example.com"}]
        assert payload["personalizations"][0]["cc"] == [{"email": "ops@example.com"}]
        types = [c["type"] for c in payload["content"]]
        assert types == ["text/plain", "text/html"]

    async def test_api_error_returns_failed_result(self, adapter):
        adapter._client = _mock_client(status_code=401)

        result = await adapter.send_email(_request())

        assert result.success is False
        assert "401" in (result.error_message or "")

    async def test_exception_returns_failed_result(self, adapter):
        client = AsyncMock()
        client.post = AsyncMock(side_effect=httpx.ConnectError("boom"))
        client.is_closed = False
        adapter._client = client

        result = await adapter.send_email(_request())

        assert result.success is False
        assert "boom" in (result.error_message or "")


class TestFetch:
    async def test_fetch_returns_empty(self, adapter):
        assert await adapter.fetch_emails() == []


async def test_aclose(adapter):
    client = AsyncMock()
    client.is_closed = False
    adapter._client = client

    await adapter.aclose()

    client.aclose.assert_called_once()
    assert adapter._client is None
