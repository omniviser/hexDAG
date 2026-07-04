"""Messaging port family — human/system communication capabilities.

``Messaging`` is the marker port for adapters that deliver messages to
humans or external systems. Capabilities follow the ``Supports*``
convention (like ``DataStore`` + ``SupportsKeyValue``):

- :class:`SupportsNotification` — one-way, fire-and-forget alerts
  (console, Slack, PagerDuty). Pairs naturally with ``WaitNode``:
  notify the approver, suspend the pipeline, resume on their decision.
- :class:`SupportsEmail` — two-way, threaded email conversation:
  send, reply-in-thread (``in_reply_to``/``conversation_id``), and
  inbox fetch.

Rule of thumb: sending an alert → ``SupportsNotification``; holding a
conversation → ``SupportsEmail``. An email adapter MAY implement both
(``asend`` as a one-way email) so approval flows can use it.

``Notification`` (``hexdag.kernel.ports.notification``) is a deprecated
alias of :class:`SupportsNotification`.
"""

from __future__ import annotations

import html as html_mod
import re
from abc import abstractmethod
from datetime import datetime  # noqa: TC003  # pydantic field annotation needs runtime import
from html.parser import HTMLParser
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from hexdag.kernel.ports.dict_bridge import dict_bridge_call


@runtime_checkable
class Messaging(Protocol):
    """Marker port for messaging adapters.

    Carries no methods of its own — adapters implement one or more
    ``Supports*`` capabilities (:class:`SupportsNotification`,
    :class:`SupportsEmail`).
    """


@runtime_checkable
class SupportsNotification(Protocol):
    """Capability: one-way outbound notification delivery.

    Implementations deliver messages to humans or external systems:
    - Console/log output (development)
    - Slack / Teams webhooks
    - Email (see hexdag_plugins.mail adapters)
    - PagerDuty / Opsgenie

    The capability is intentionally minimal: one message, optional title,
    optional channel routing, and a metadata dict for adapter-specific
    extras (approval links, correlation keys, severity, ...). For two-way
    threaded conversations with inbox fetch, use :class:`SupportsEmail`.

    Examples
    --------
    Sending an approval request before suspending on a WaitNode::

        notification = get_port("notification")
        await notification.asend(
            f"Order {order_id} needs approval",
            title="Approval required",
            metadata={"event_key": f"approval:{order_id}"},
        )
    """

    @abstractmethod
    async def asend(
        self,
        message: str,
        *,
        title: str | None = None,
        channel: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Send a notification (async).

        Args
        ----
            message: Notification body text.
            title: Optional short title/subject.
            channel: Optional routing hint (e.g. Slack channel, email
                address, topic name). Adapter-specific.
            metadata: Optional adapter-specific extras (links,
                correlation keys, severity, ...).

        Raises
        ------
        Exception
            Adapter-specific delivery errors. Callers that must not
            fail on notification errors should catch and log.
        """
        ...


# ---------------------------------------------------------------------------
# Email capability
# ---------------------------------------------------------------------------


class SendEmailRequest(BaseModel):
    """Email send request."""

    to_addresses: list[str]
    subject: str
    body_text: str
    body_html: str | None = None
    cc_addresses: list[str] | None = None
    in_reply_to: str | None = None
    conversation_id: str | None = None
    metadata: dict[str, Any] | None = None


class SendEmailResult(BaseModel):
    """Email send result."""

    message_id: str
    conversation_id: str | None = None
    sent_at: datetime
    success: bool = True
    error_message: str | None = None


@runtime_checkable
class SupportsEmail(Protocol):
    """Capability: two-way threaded email conversation.

    The adapter handles sending pre-composed emails and fetching new
    inbound emails. Template rendering is handled separately (e.g. by an
    application-level templated email node).

    Threading: replies carry ``in_reply_to`` (message id) and/or
    ``conversation_id`` in :class:`SendEmailRequest`; adapters use them
    to keep the provider-side thread intact.

    Convention (documentation-level contract — ``runtime_checkable``
    protocols cannot check non-method attributes): adapters expose a class
    attribute ``auto_quotes_thread: bool``. ``True`` means the provider
    automatically appends the quoted conversation history on reply (e.g.
    MS Graph ``/reply``); ``False`` means callers must build the threaded
    body themselves (e.g. SendGrid).
    """

    async def send_email(self, request: SendEmailRequest) -> SendEmailResult:
        """Send an email."""
        ...

    async def send_email_from_dict(self, request: dict) -> dict:
        """Send email from dict (for node compatibility).

        Default implementation delegates to :meth:`send_email` via Pydantic
        ``model_validate`` / ``model_dump(mode="json")``.  Adapters do NOT
        need to override this.
        """
        return await dict_bridge_call(self.send_email, request, SendEmailRequest)

    async def fetch_emails(self, since: datetime | None = None, limit: int = 50) -> list[dict]:
        """Fetch recent inbound emails from inbox.

        Args:
            since: Only fetch emails received after this datetime.
            limit: Maximum number of emails to return.

        Returns:
            List of email dicts with fields: message_id, conversation_id,
            in_reply_to, from_address, to_addresses, cc_addresses, subject,
            body_text, body_html, received_at, has_attachments.
        """
        ...


# ---------------------------------------------------------------------------
# Email utilities
# ---------------------------------------------------------------------------


class _HTMLTextExtractor(HTMLParser):
    """Strips HTML tags and returns plain text."""

    def __init__(self) -> None:
        """Initialize with an empty text buffer."""
        super().__init__()
        self._pieces: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Skip script/style content; emit newlines for block tags."""
        if tag in ("script", "style"):
            self._skip = True
        elif tag in ("br", "p", "div", "tr", "li"):
            self._pieces.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """Stop skipping when script/style closes."""
        if tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data: str) -> None:
        """Collect text outside skipped tags."""
        if not self._skip:
            self._pieces.append(data)

    def get_text(self) -> str:
        """Return the accumulated plain text."""
        return "".join(self._pieces)


def html_to_plain_text(html_content: str) -> str:
    """Convert HTML to plain text using stdlib parser.

    Args:
        html_content: Raw HTML string.

    Returns:
        Plain text with HTML tags stripped, entities decoded, and
        excessive whitespace collapsed.
    """
    if not html_content:
        return ""
    extractor = _HTMLTextExtractor()
    extractor.feed(html_content)
    text = extractor.get_text()
    text = html_mod.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
