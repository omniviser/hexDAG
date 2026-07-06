"""No-op email adapter that logs but does not send anything.

Used for tests, dry runs, and environments where email sending should
be completely disabled.
"""

import uuid
from datetime import UTC, datetime

from hexdag.kernel.logging import get_logger
from hexdag.kernel.ports.messaging import SendEmailRequest, SendEmailResult, SupportsEmail

logger = get_logger(__name__)


class NoopEmailAdapter(SupportsEmail):
    """No-op email adapter that logs but does not send anything.

    All ``send_email`` calls return a successful result with a generated
    message ID. ``fetch_emails`` always returns an empty list.
    """

    auto_quotes_thread: bool = True

    async def send_email(self, request: SendEmailRequest) -> SendEmailResult:
        """Log the email and return a fake success result."""
        msg_id = f"noop-{uuid.uuid4().hex[:12]}"
        logger.info(
            "[NOOP] Would send email to %s subject=%r (message_id=%s)",
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

    async def fetch_emails(self, since: datetime | None = None, limit: int = 50) -> list[dict]:
        """Return an empty inbox."""
        logger.info("[NOOP] fetch_emails called — returning empty list")
        return []
