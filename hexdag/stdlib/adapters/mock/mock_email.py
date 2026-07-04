"""Mock email adapter for testing threaded conversation flows."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from hexdag.kernel.ports.messaging import (
    Messaging,
    SendEmailRequest,
    SendEmailResult,
    SupportsEmail,
)
from hexdag.stdlib.adapters.base import HexDAGAdapter


class MockEmailAdapter(
    HexDAGAdapter,
    Messaging,
    SupportsEmail,
    yaml_alias="mock_email",
    port="email",
):
    """Mock implementation of the ``SupportsEmail`` capability for testing.

    Records every sent request in ``sent_requests``, returns deterministic
    message ids, and serves a scripted inbox via :meth:`queue_inbound`.

    Parameters
    ----------
    should_raise : bool
        If True, ``send_email`` raises to simulate delivery failure.

    Examples
    --------
    >>> import asyncio
    >>> mock = MockEmailAdapter()
    >>> result = asyncio.run(mock.send_email(SendEmailRequest(
    ...     to_addresses=["a@example.com"], subject="Hi", body_text="Hello",
    ... )))
    >>> result.message_id
    'mock-message-1'
    >>> mock.sent_requests[0].subject
    'Hi'
    """

    auto_quotes_thread = True

    def __init__(self, should_raise: bool = False, **kwargs: Any) -> None:
        """Create the mock; set *should_raise* to simulate send failures."""
        self.should_raise = should_raise
        self.sent_requests: list[SendEmailRequest] = []
        self._inbox: list[dict[str, Any]] = []

    async def send_email(self, request: SendEmailRequest) -> SendEmailResult:
        """Record the request and return a deterministic result."""
        if self.should_raise:
            raise RuntimeError("MockEmailAdapter delivery failure (configured)")
        self.sent_requests.append(request)
        return SendEmailResult(
            message_id=f"mock-message-{len(self.sent_requests)}",
            conversation_id=request.conversation_id or f"mock-conv-{len(self.sent_requests)}",
            sent_at=datetime.now(UTC),
            success=True,
        )

    async def send_email_from_dict(self, request: dict) -> dict:
        """Send email from dict (delegates to :meth:`send_email`)."""
        from hexdag.kernel.ports.dict_bridge import (  # lazy: avoid import cost at module load
            dict_bridge_call,
        )

        return await dict_bridge_call(self.send_email, request, SendEmailRequest)

    async def fetch_emails(self, since: datetime | None = None, limit: int = 50) -> list[dict]:
        """Return queued inbound emails (optionally filtered by ``since``)."""
        emails = self._inbox
        if since is not None:
            emails = [e for e in emails if e.get("received_at") is None or e["received_at"] > since]
        return emails[:limit]

    def queue_inbound(self, email: dict[str, Any]) -> None:
        """Script an inbound email for the next :meth:`fetch_emails` call."""
        self._inbox.append(email)

    def reset(self) -> None:
        """Clear recorded sends and the scripted inbox."""
        self.sent_requests.clear()
        self._inbox.clear()
