"""Tests for the unified Messaging port family."""

from datetime import UTC, datetime

import pytest

from hexdag.kernel.ports.messaging import (
    Messaging,
    SendEmailRequest,
    SendEmailResult,
    SupportsEmail,
    SupportsNotification,
    html_to_plain_text,
)


class TestNotificationAlias:
    def test_notification_is_supports_notification(self):
        from hexdag.kernel.ports.notification import Notification

        assert Notification is SupportsNotification

    def test_kernel_ports_package_reexports(self):
        from hexdag.kernel import ports

        assert ports.Notification is SupportsNotification
        assert ports.SupportsEmail is SupportsEmail
        assert ports.Messaging is Messaging


class TestProtocolChecks:
    def test_notification_structural_isinstance(self):
        class Adapter:
            async def asend(self, message, *, title=None, channel=None, metadata=None):
                return None

        assert isinstance(Adapter(), SupportsNotification)

    def test_email_structural_isinstance(self):
        class Adapter:
            async def send_email(self, request):
                return None

            async def send_email_from_dict(self, request):
                return {}

            async def fetch_emails(self, since=None, limit=50):
                return []

        assert isinstance(Adapter(), SupportsEmail)

    def test_notification_adapter_is_not_email(self):
        class Adapter:
            async def asend(self, message, *, title=None, channel=None, metadata=None):
                return None

        assert not isinstance(Adapter(), SupportsEmail)


class TestSendEmailDictBridge:
    @pytest.mark.asyncio
    async def test_send_email_from_dict_round_trip(self):
        class Adapter(SupportsEmail):
            def __init__(self):
                self.last_request = None

            async def send_email(self, request: SendEmailRequest) -> SendEmailResult:
                self.last_request = request
                return SendEmailResult(
                    message_id="m-1",
                    conversation_id=request.conversation_id,
                    sent_at=datetime.now(UTC),
                )

            async def fetch_emails(self, since=None, limit=50):
                return []

        adapter = Adapter()
        result = await adapter.send_email_from_dict({
            "to_addresses": ["a@example.com"],
            "subject": "Hi",
            "body_text": "Hello",
            "in_reply_to": "<msg-0>",
            "conversation_id": "conv-1",
        })

        assert adapter.last_request is not None
        assert adapter.last_request.in_reply_to == "<msg-0>"
        assert result["message_id"] == "m-1"
        assert result["conversation_id"] == "conv-1"
        assert result["success"] is True


class TestHtmlToPlainText:
    def test_strips_tags_and_decodes_entities(self):
        html = "<div>Hello&nbsp;<b>world</b></div><p>Second&amp;line</p>"
        text = html_to_plain_text(html)
        assert "Hello" in text
        assert "world" in text
        assert "Second&line" in text
        assert "<" not in text

    def test_skips_script_and_style(self):
        html = "<style>.x{}</style><script>alert(1)</script><p>visible</p>"
        assert html_to_plain_text(html).strip() == "visible"

    def test_empty_input(self):
        assert html_to_plain_text("") == ""


class TestPortDetection:
    def test_email_adapter_detected_as_messaging(self):
        from hexdag.kernel.ports.detection import detect_port_type
        from hexdag.stdlib.adapters.mock import MockEmailAdapter

        assert detect_port_type(MockEmailAdapter) == "messaging"
