"""Microsoft Graph email adapter for hexDAG.

Uses OAuth2 client_credentials flow to authenticate with Microsoft Graph.
Sends email via the sendMail endpoint and fetches inbox messages with
OData filters. Access tokens are cached in memory and refreshed
automatically before expiry.

Examples
--------
YAML configuration::

    spec:
      ports:
        email:
          adapter: hexdag_plugins.email.GraphEmailAdapter
          config:
            tenant_id: ${GRAPH_TENANT_ID}
            client_id: ${GRAPH_CLIENT_ID}
            client_secret: ${GRAPH_CLIENT_SECRET}
            user_id: ${EMAIL_USER_ID}
"""

import html as html_mod
import os
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from hexdag.kernel.logging import get_logger

from hexdag_plugins._ports import (
    SendEmailRequest,
    SendEmailResult,
    SupportsEmail,
    html_to_plain_text,
)

logger = get_logger(__name__)


class GraphEmailAdapter(SupportsEmail):
    """Microsoft Graph email adapter for production use.

    Uses OAuth2 client_credentials flow to authenticate with Microsoft
    Graph. Sends email via the sendMail endpoint and fetches inbox
    messages with OData filters.

    Parameters
    ----------
    tenant_id : str | None
        Azure AD tenant ID. Falls back to ``GRAPH_TENANT_ID`` env var.
    client_id : str | None
        Azure AD application (client) ID. Falls back to ``GRAPH_CLIENT_ID``.
    client_secret : str | None
        Azure AD client secret. Falls back to ``GRAPH_CLIENT_SECRET``.
    user_id : str | None
        Mailbox user ID (email). Falls back to ``EMAIL_USER_ID``.
    timeout : float
        Request timeout in seconds.
    """

    auto_quotes_thread: bool = True

    def __init__(
        self,
        tenant_id: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        user_id: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.tenant_id = tenant_id or os.environ.get("GRAPH_TENANT_ID", "")
        self.client_id = client_id or os.environ.get("GRAPH_CLIENT_ID", "")
        self.client_secret = client_secret or os.environ.get("GRAPH_CLIENT_SECRET", "")
        self.user_id = user_id or os.environ.get("EMAIL_USER_ID", "")
        self.timeout = timeout

        if not all([self.tenant_id, self.client_id, self.client_secret, self.user_id]):
            raise ValueError(
                "Graph adapter requires GRAPH_TENANT_ID, GRAPH_CLIENT_ID, "
                "GRAPH_CLIENT_SECRET, and EMAIL_USER_ID"
            )

        self._access_token: str | None = None
        self._token_expires_at: float = 0.0
        self._client: httpx.AsyncClient | None = None
        self._graph_base = f"https://graph.microsoft.com/v1.0/users/{self.user_id}"
        self._token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"

    # ------------------------------------------------------------------
    # HTTP client
    # ------------------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        """Return a reusable HTTP client, creating one if needed."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    async def _ensure_token(self) -> str:
        """Acquire or return a cached OAuth2 access token."""
        if self._access_token and time.monotonic() < self._token_expires_at:
            return self._access_token

        client = self._get_client()
        response = await client.post(
            self._token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": "https://graph.microsoft.com/.default",
            },
        )
        response.raise_for_status()
        token_data = response.json()

        self._access_token = token_data["access_token"]
        expires_in = int(token_data.get("expires_in", 3600))
        # Refresh 5 minutes before expiry
        self._token_expires_at = time.monotonic() + expires_in - 300

        logger.info("Acquired new Graph API access token")
        return self._access_token

    async def _headers(self) -> dict[str, str]:
        """Return authorization headers with a valid access token."""
        token = await self._ensure_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Message lookup (for threading)
    # ------------------------------------------------------------------

    async def _find_graph_message_id_by_internet_id(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        internet_message_id: str,
    ) -> str | None:
        """Find a Graph message ID by its internetMessageId."""
        escaped_id = internet_message_id.replace("'", "''")
        response = await client.get(
            f"{self._graph_base}/messages",
            headers=headers,
            params={
                "$filter": f"internetMessageId eq '{escaped_id}'",
                "$top": "1",
                "$select": "id",
            },
        )
        if response.status_code == 200:
            messages = response.json().get("value", [])
            if messages:
                logger.info(
                    "Found Graph message %s for internetMessageId=%s",
                    messages[0]["id"],
                    internet_message_id[:60],
                )
                return messages[0]["id"]
            logger.info(
                "No Graph message found for internetMessageId=%s",
                internet_message_id[:60],
            )
        else:
            logger.warning(
                "Graph message lookup returned %d for internetMessageId=%s",
                response.status_code,
                internet_message_id[:60],
            )
        return None

    async def _find_graph_message_id_by_conversation(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        conversation_id: str,
    ) -> str | None:
        """Find the latest Graph message ID in a conversation (fallback)."""
        escaped_id = conversation_id.replace("'", "''")
        response = await client.get(
            f"{self._graph_base}/messages",
            headers=headers,
            params={
                "$filter": f"conversationId eq '{escaped_id}'",
                "$orderby": "receivedDateTime desc",
                "$top": "1",
                "$select": "id",
            },
        )
        if response.status_code == 200:
            messages = response.json().get("value", [])
            if messages:
                logger.info(
                    "Found Graph message %s for conversationId=%s...",
                    messages[0]["id"],
                    conversation_id[:40],
                )
                return messages[0]["id"]
            logger.info(
                "No Graph messages found for conversationId=%s...",
                conversation_id[:40],
            )
        else:
            logger.warning(
                "Graph message lookup returned %d for conversationId=%s...",
                response.status_code,
                conversation_id[:40],
            )
        return None

    # ------------------------------------------------------------------
    # Send email
    # ------------------------------------------------------------------

    async def send_email(self, request: SendEmailRequest) -> SendEmailResult:
        """Send email via Microsoft Graph API.

        Uses the ``/reply`` endpoint when ``conversation_id`` is available
        to preserve email threading. Falls back to ``/sendMail`` for new
        conversations.
        """
        try:
            headers = await self._headers()
            client = self._get_client()

            recipients = [{"emailAddress": {"address": addr}} for addr in request.to_addresses]

            graph_msg_id: str | None = None

            if request.in_reply_to:
                graph_msg_id = await self._find_graph_message_id_by_internet_id(
                    client, headers, request.in_reply_to
                )

            if not graph_msg_id and request.conversation_id:
                graph_msg_id = await self._find_graph_message_id_by_conversation(
                    client, headers, request.conversation_id
                )

            if graph_msg_id:
                logger.info(
                    "Threading via /reply endpoint (in_reply_to=%s, conversation=%s)",
                    request.in_reply_to,
                    request.conversation_id,
                )
                return await self._send_as_reply(client, headers, graph_msg_id, request, recipients)

            if request.in_reply_to or request.conversation_id:
                logger.info("No Graph message found for threading, falling back to /sendMail")

            return await self._send_new(client, headers, request, recipients)

        except Exception as e:
            logger.error("Failed to send email via Graph: %s", e, exc_info=True)
            return SendEmailResult(
                message_id=f"failed-{uuid.uuid4().hex[:12]}",
                sent_at=datetime.now(UTC),
                success=False,
                error_message=str(e),
            )

    @staticmethod
    def _to_html_comment(body_text: str, body_html: str | None) -> str:
        """Convert email body to HTML for the Graph /reply comment field."""
        if body_html:
            return body_html
        escaped = html_mod.escape(body_text)
        return escaped.replace("\n", "<br>")

    async def _send_as_reply(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        graph_msg_id: str,
        request: SendEmailRequest,
        recipients: list[dict[str, Any]],
    ) -> SendEmailResult:
        """Send as a reply to an existing message, preserving the thread."""
        comment_html = self._to_html_comment(request.body_text, request.body_html)
        reply_payload: dict[str, Any] = {
            "message": {"toRecipients": recipients},
            "comment": comment_html,
        }
        if request.cc_addresses:
            reply_payload["message"]["ccRecipients"] = [
                {"emailAddress": {"address": addr}} for addr in request.cc_addresses
            ]

        response = await client.post(
            f"{self._graph_base}/messages/{graph_msg_id}/reply",
            headers=headers,
            json=reply_payload,
        )

        if response.status_code == 202:
            msg_id = f"graph-{uuid.uuid4().hex[:12]}"
            logger.info(
                "Sent reply to %s subject=%r",
                request.to_addresses,
                request.subject,
            )
            return SendEmailResult(
                message_id=msg_id,
                conversation_id=request.conversation_id,
                sent_at=datetime.now(UTC),
                success=True,
            )

        error_text = response.text[:500]
        logger.error("Graph reply returned %d: %s", response.status_code, error_text)
        logger.info("Falling back to sendMail")
        return await self._send_new(client, headers, request, recipients)

    async def _send_new(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        request: SendEmailRequest,
        recipients: list[dict[str, Any]],
    ) -> SendEmailResult:
        """Send a new email (not a reply)."""
        body_content_type = "HTML" if request.body_html else "Text"
        body_content = request.body_html or request.body_text

        message: dict[str, Any] = {
            "subject": request.subject,
            "body": {
                "contentType": body_content_type,
                "content": body_content,
            },
            "toRecipients": recipients,
        }

        if request.cc_addresses:
            message["ccRecipients"] = [
                {"emailAddress": {"address": addr}} for addr in request.cc_addresses
            ]

        if request.conversation_id:
            message["conversationId"] = request.conversation_id

        payload = {"message": message, "saveToSentItems": True}

        response = await client.post(
            f"{self._graph_base}/sendMail",
            headers=headers,
            json=payload,
        )

        if response.status_code == 202:
            msg_id = f"graph-{uuid.uuid4().hex[:12]}"
            logger.info(
                "Sent email to %s subject=%r",
                request.to_addresses,
                request.subject,
            )
            return SendEmailResult(
                message_id=msg_id,
                conversation_id=request.conversation_id,
                sent_at=datetime.now(UTC),
                success=True,
            )

        error_text = response.text[:500]
        logger.error(
            "Graph sendMail returned %d: %s",
            response.status_code,
            error_text,
        )
        return SendEmailResult(
            message_id=f"failed-{uuid.uuid4().hex[:12]}",
            sent_at=datetime.now(UTC),
            success=False,
            error_message=f"Graph API error {response.status_code}: {error_text}",
        )

    # ------------------------------------------------------------------
    # Fetch emails
    # ------------------------------------------------------------------

    async def fetch_emails(
        self,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Fetch recent emails from inbox via Microsoft Graph API.

        Uses a time-window filter. Fetched messages are marked as read
        after retrieval.
        """
        try:
            headers = await self._headers()
            client = self._get_client()

            if not since:
                since = datetime.now(UTC) - timedelta(hours=24)
            since_str = since.strftime("%Y-%m-%dT%H:%M:%SZ")
            filters = [f"receivedDateTime ge {since_str}"]

            params: dict[str, str | int] = {
                "$top": limit,
                "$orderby": "receivedDateTime desc",
                "$filter": " and ".join(filters),
                "$select": (
                    "id,internetMessageId,conversationId,"
                    "subject,bodyPreview,body,from,"
                    "toRecipients,ccRecipients,"
                    "receivedDateTime,hasAttachments,isRead"
                ),
            }

            response = await client.get(
                f"{self._graph_base}/mailFolders/inbox/messages",
                headers=headers,
                params=params,
            )
            response.raise_for_status()

            raw_messages = response.json().get("value", [])

            emails: list[dict] = []
            for msg in raw_messages:
                from_field = msg.get("from", {})
                from_addr = from_field.get("emailAddress", {}).get("address", "")
                body = msg.get("body", {})
                body_content = body.get("content", "")
                body_content_type = body.get("contentType", "")

                if body_content_type == "html":
                    body_html = body_content
                    body_text = html_to_plain_text(body_content)
                elif body_content_type == "text":
                    body_html = None
                    body_text = body_content
                else:
                    body_html = None
                    body_text = ""

                emails.append(
                    {
                        "message_id": msg.get("internetMessageId", msg["id"]),
                        "conversation_id": msg.get("conversationId"),
                        "in_reply_to": None,
                        "from_address": from_addr,
                        "to_addresses": [
                            r["emailAddress"]["address"] for r in msg.get("toRecipients", [])
                        ],
                        "cc_addresses": [
                            r["emailAddress"]["address"] for r in msg.get("ccRecipients", [])
                        ],
                        "subject": msg.get("subject", ""),
                        "body_text": body_text or msg.get("bodyPreview", ""),
                        "body_html": body_html,
                        "received_at": msg.get("receivedDateTime"),
                        "has_attachments": msg.get("hasAttachments", False),
                    }
                )

            # Mark as read (best effort)
            for msg in raw_messages:
                try:
                    await client.patch(
                        f"{self._graph_base}/messages/{msg['id']}",
                        headers=headers,
                        json={"isRead": True},
                    )
                except Exception:
                    logger.warning("Failed to mark message %s as read", msg["id"])

            logger.info("Fetched %d emails from Graph inbox", len(emails))
            return emails

        except Exception as e:
            logger.error("Failed to fetch emails via Graph: %s", e, exc_info=True)
            return []
