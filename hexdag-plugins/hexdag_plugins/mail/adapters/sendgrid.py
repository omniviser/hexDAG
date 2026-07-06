"""SendGrid email adapter for hexDAG.

Uses the SendGrid v3 REST API via httpx to send transactional emails.
Email threading is preserved via ``In-Reply-To`` / ``References`` SMTP
headers — SendGrid does NOT auto-append the quoted conversation
(``auto_quotes_thread = False``), so callers must build threaded bodies
themselves.

Examples
--------
YAML configuration::

    spec:
      ports:
        email:
          adapter: hexdag_plugins.mail.SendGridEmailAdapter
          config:
            api_key: ${SENDGRID_API_KEY}
            from_address: ${EMAIL_FROM_ADDRESS}
"""

import os
import uuid
from datetime import UTC, datetime

import httpx
from hexdag.kernel.logging import get_logger
from hexdag.kernel.ports.messaging import (
    SendEmailRequest,
    SendEmailResult,
    SupportsEmail,
)

logger = get_logger(__name__)


class SendGridEmailAdapter(SupportsEmail):
    """SendGrid v3 email adapter for high-throughput sending.

    Send-only: ``fetch_emails`` returns an empty list with a warning.
    Wire a ``GraphEmailAdapter`` (or another inbox-capable adapter) on a
    separate port when inbound mail is needed.

    Parameters
    ----------
    api_key : str | None
        SendGrid API key. Falls back to ``SENDGRID_API_KEY`` env var.
    from_address : str | None
        Sender address. Falls back to ``EMAIL_FROM_ADDRESS`` env var,
        then ``EMAIL_USER_ID``.
    from_name : str | None
        Optional sender display name. Falls back to ``SENDGRID_FROM_NAME``.
    timeout : float
        Request timeout in seconds.
    """

    auto_quotes_thread: bool = False

    SENDGRID_API_URL = "https://api.sendgrid.com/v3/mail/send"

    def __init__(
        self,
        api_key: str | None = None,
        from_address: str | None = None,
        from_name: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("SENDGRID_API_KEY", "")
        self.from_address = (
            from_address
            or os.environ.get("EMAIL_FROM_ADDRESS", "")
            or os.environ.get("EMAIL_USER_ID", "")
        )
        self.from_name = from_name or os.environ.get("SENDGRID_FROM_NAME", "")
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("SendGrid adapter requires SENDGRID_API_KEY")
        if not self.from_address:
            raise ValueError("SendGrid adapter requires a from_address (EMAIL_FROM_ADDRESS)")

        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Return a reusable HTTP client, creating one if needed."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _headers(self) -> dict[str, str]:
        """Return authorization headers (static Bearer token)."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def send_email(self, request: SendEmailRequest) -> SendEmailResult:
        """Send email via the SendGrid v3 API."""
        try:
            client = self._get_client()

            personalization: dict = {
                "to": [{"email": addr} for addr in request.to_addresses],
            }
            if request.cc_addresses:
                personalization["cc"] = [{"email": addr} for addr in request.cc_addresses]

            content: list[dict] = []
            if request.body_text:
                content.append({"type": "text/plain", "value": request.body_text})
            if request.body_html:
                content.append({"type": "text/html", "value": request.body_html})
            if not content:
                content.append({"type": "text/plain", "value": ""})

            from_field: dict = {"email": self.from_address}
            if self.from_name:
                from_field["name"] = self.from_name

            payload: dict = {
                "personalizations": [personalization],
                "from": from_field,
                "subject": request.subject,
                "content": content,
            }

            smtp_headers: dict[str, str] = {}
            if request.in_reply_to:
                smtp_headers["In-Reply-To"] = request.in_reply_to
                smtp_headers["References"] = request.in_reply_to
            if request.conversation_id:
                smtp_headers["X-Entity-Ref-ID"] = request.conversation_id
            if smtp_headers:
                payload["headers"] = smtp_headers

            response = await client.post(
                self.SENDGRID_API_URL,
                headers=self._headers(),
                json=payload,
            )

            if response.status_code == 202:
                sg_message_id = response.headers.get("X-Message-Id", "")
                msg_id = (
                    f"sendgrid-{sg_message_id}"
                    if sg_message_id
                    else f"sendgrid-{uuid.uuid4().hex[:12]}"
                )
                logger.info(
                    "Sent email to %s subject=%r (message_id=%s)",
                    request.to_addresses,
                    request.subject,
                    msg_id,
                )
                return SendEmailResult(
                    message_id=msg_id,
                    conversation_id=request.conversation_id,
                    sent_at=datetime.now(UTC),
                    success=True,
                )

            error_text = response.text[:500]
            logger.error(
                "SendGrid API returned %d: %s",
                response.status_code,
                error_text,
            )
            return SendEmailResult(
                message_id=f"failed-{uuid.uuid4().hex[:12]}",
                sent_at=datetime.now(UTC),
                success=False,
                error_message=f"SendGrid API error {response.status_code}: {error_text}",
            )

        except Exception as e:
            logger.error("Failed to send email via SendGrid: %s", e, exc_info=True)
            return SendEmailResult(
                message_id=f"failed-{uuid.uuid4().hex[:12]}",
                sent_at=datetime.now(UTC),
                success=False,
                error_message=str(e),
            )

    async def fetch_emails(self, since: datetime | None = None, limit: int = 50) -> list[dict]:
        """Return an empty inbox — SendGrid is send-only.

        Wire an inbox-capable adapter (e.g. ``GraphEmailAdapter``) on a
        separate port for inbound mail.
        """
        logger.warning(
            "SendGridEmailAdapter.fetch_emails called — SendGrid is send-only, returning empty list"
        )
        return []
