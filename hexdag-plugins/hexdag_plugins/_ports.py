"""Port protocols used by hexdag_plugins.

These protocols were previously in hexdag.kernel.ports but were moved here
because they are only used by the plugins package, not by core hexDAG.
"""

import html as html_mod
import re
from datetime import datetime
from html.parser import HTMLParser
from typing import Any, Protocol, runtime_checkable

from hexdag.kernel.ports.dict_bridge import dict_bridge_call
from hexdag.kernel.ports.healthcheck import HealthStatus
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Email port
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
    """Protocol for email adapters.

    The adapter handles sending pre-composed emails and fetching
    new inbound emails. Template rendering is handled separately
    (e.g. by TemplatedEmailNode).
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
        super().__init__()
        self._pieces: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self._skip = True
        elif tag in ("br", "p", "div", "tr", "li"):
            self._pieces.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._pieces.append(data)

    def get_text(self) -> str:
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


@runtime_checkable
class FileStorage(Protocol):
    """Port for file storage operations.

    Provides a unified interface for local and cloud file storage.
    """

    async def aupload(self, local_path: str, remote_path: str) -> dict:
        """Upload a file."""
        ...

    async def adownload(self, remote_path: str, local_path: str) -> dict:
        """Download a file."""
        ...

    async def adelete(self, remote_path: str) -> dict:
        """Delete a file."""
        ...

    async def alist(self, prefix: str = "") -> list[str]:
        """List files with optional prefix."""
        ...

    async def aexists(self, remote_path: str) -> bool:
        """Check if file exists."""
        ...

    async def aget_metadata(self, remote_path: str) -> dict:
        """Get file metadata."""
        ...

    async def ahealth_check(self) -> HealthStatus:
        """Check storage health."""
        ...
