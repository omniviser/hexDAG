"""Outlook nodes for reading and sending emails via Microsoft Graph API.

These nodes provide email integration for ETL pipelines using Microsoft 365/Outlook.
"""

from typing import Any

from hexdag.builtin.nodes.base_node_factory import BaseNodeFactory
from hexdag.core.domain.dag import NodeSpec
from hexdag.core.registry import node
from hexdag.core.registry.models import NodeSubtype
from pydantic import BaseModel


class EmailMessage(BaseModel):
    """Model representing an email message."""

    id: str
    subject: str
    sender: str
    recipients: list[str]
    body: str
    body_preview: str
    received_at: str | None = None
    has_attachments: bool = False
    is_read: bool = False


class OutlookReaderOutput(BaseModel):
    """Output model for OutlookReaderNode."""

    messages: list[dict[str, Any]]
    count: int
    folder: str


class OutlookSenderOutput(BaseModel):
    """Output model for OutlookSenderNode."""

    message_id: str
    subject: str
    recipients: list[str]
    success: bool


@node(name="outlook_reader_node", subtype=NodeSubtype.FUNCTION, namespace="etl")
class OutlookReaderNode(BaseNodeFactory):
    """Node for reading emails from Microsoft Outlook via Graph API.

    Requires Microsoft Graph API credentials configured via environment variables
    or passed directly:
    - MICROSOFT_CLIENT_ID
    - MICROSOFT_CLIENT_SECRET
    - MICROSOFT_TENANT_ID

    Examples
    --------
    YAML pipeline::

        - kind: etl:outlook_reader_node
          metadata:
            name: read_inbox
          spec:
            folder: inbox
            max_messages: 50
            filter: "isRead eq false"
          dependencies: []

        - kind: etl:outlook_reader_node
          metadata:
            name: read_sent
          spec:
            folder: sentitems
            max_messages: 20
          dependencies: []
    """

    def __call__(
        self,
        name: str,
        folder: str = "inbox",
        max_messages: int = 50,
        filter: str | None = None,
        select: list[str] | None = None,
        include_body: bool = True,
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create an Outlook reader node specification.

        Parameters
        ----------
        name : str
            Node name
        folder : str
            Mail folder to read from: 'inbox', 'sentitems', 'drafts', 'deleteditems'
            Or a custom folder path like 'inbox/subfolder'
        max_messages : int
            Maximum number of messages to retrieve (default: 50)
        filter : str, optional
            OData filter expression (e.g., "isRead eq false", "from/emailAddress/address eq 'someone@example.com'")
        select : list[str], optional
            Fields to retrieve. Default: subject, from, toRecipients, body, receivedDateTime, hasAttachments, isRead
        include_body : bool
            Whether to include full email body (default: True)
        deps : list[str], optional
            Dependency node names
        **kwargs : Any
            Additional node parameters

        Returns
        -------
        NodeSpec
            Node specification ready for execution
        """
        wrapped_fn = self._create_reader_function(name, folder, max_messages, filter, select, include_body)

        input_schema = {"input_data": dict | None}
        output_model = OutlookReaderOutput

        input_model = self.create_pydantic_model(f"{name}Input", input_schema)

        node_params = {
            "folder": folder,
            "max_messages": max_messages,
            "filter": filter,
            "select": select,
            "include_body": include_body,
            **kwargs,
        }

        return NodeSpec(
            name=name,
            fn=wrapped_fn,
            in_model=input_model,
            out_model=output_model,
            deps=frozenset(deps or []),
            params=node_params,
        )

    def _create_reader_function(
        self,
        name: str,
        folder: str,
        max_messages: int,
        filter: str | None,
        select: list[str] | None,
        include_body: bool,
    ) -> Any:
        """Create the email reading function."""

        async def read_emails(input_data: Any = None) -> dict[str, Any]:
            """Read emails from Outlook via Microsoft Graph API."""
            import os

            # Get credentials from environment or input
            client_id = os.environ.get("MICROSOFT_CLIENT_ID")
            client_secret = os.environ.get("MICROSOFT_CLIENT_SECRET")
            tenant_id = os.environ.get("MICROSOFT_TENANT_ID")

            if not all([client_id, client_secret, tenant_id]):
                raise ValueError(
                    "Microsoft Graph API credentials not configured. "
                    "Set MICROSOFT_CLIENT_ID, MICROSOFT_CLIENT_SECRET, and MICROSOFT_TENANT_ID environment variables."
                )

            # Import Azure Identity and Graph SDK
            try:
                from azure.identity import ClientSecretCredential
                from msgraph import GraphServiceClient
            except ImportError as e:
                raise ImportError(
                    "Microsoft Graph SDK not installed. Install with: pip install azure-identity msgraph-sdk"
                ) from e

            # Create credential and client
            credential = ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret,
            )
            client = GraphServiceClient(credential)

            # Build query parameters
            default_select = [
                "id",
                "subject",
                "from",
                "toRecipients",
                "receivedDateTime",
                "hasAttachments",
                "isRead",
                "bodyPreview",
            ]
            if include_body:
                default_select.append("body")

            query_select = select or default_select

            # Map folder names to Graph API paths
            folder_map = {
                "inbox": "inbox",
                "sentitems": "sentItems",
                "sent": "sentItems",
                "drafts": "drafts",
                "deleteditems": "deletedItems",
                "deleted": "deletedItems",
                "junk": "junkemail",
                "archive": "archive",
            }
            graph_folder = folder_map.get(folder.lower(), folder)

            # Build request
            messages_request = client.me.mail_folders.by_mail_folder_id(graph_folder).messages

            # Get messages
            result = await messages_request.get(
                request_configuration=lambda config: setattr(config.query_parameters, "top", max_messages)
                or (setattr(config.query_parameters, "select", query_select) if query_select else None)
                or (setattr(config.query_parameters, "filter", filter) if filter else None)
            )

            messages = []
            if result and result.value:
                for msg in result.value:
                    message_dict = {
                        "id": msg.id,
                        "subject": msg.subject or "",
                        "sender": msg.from_.email_address.address if msg.from_ and msg.from_.email_address else "",
                        "recipients": [r.email_address.address for r in (msg.to_recipients or []) if r.email_address],
                        "body": msg.body.content if msg.body else "",
                        "body_preview": msg.body_preview or "",
                        "received_at": msg.received_date_time.isoformat() if msg.received_date_time else None,
                        "has_attachments": msg.has_attachments or False,
                        "is_read": msg.is_read or False,
                    }
                    messages.append(message_dict)

            return {
                "messages": messages,
                "count": len(messages),
                "folder": folder,
            }

        read_emails.__name__ = f"outlook_reader_{name}"
        read_emails.__doc__ = f"Read emails from Outlook folder: {folder}"

        return read_emails


@node(name="outlook_sender_node", subtype=NodeSubtype.FUNCTION, namespace="etl")
class OutlookSenderNode(BaseNodeFactory):
    """Node for sending emails via Microsoft Outlook Graph API.

    Requires Microsoft Graph API credentials configured via environment variables:
    - MICROSOFT_CLIENT_ID
    - MICROSOFT_CLIENT_SECRET
    - MICROSOFT_TENANT_ID

    Examples
    --------
    YAML pipeline::

        - kind: etl:outlook_sender_node
          metadata:
            name: send_report
          spec:
            to:
              - recipient@example.com
            subject: "Daily Report - {{date}}"
            body_template: |
              Hello,

              Please find the daily report attached.

              Summary:
              - Total records: {{total}}
              - Processed: {{processed}}

              Best regards
          dependencies:
            - generate_report

        - kind: etl:outlook_sender_node
          metadata:
            name: send_alert
          spec:
            to:
              - alerts@example.com
            cc:
              - manager@example.com
            subject: "Alert: {{alert_type}}"
            body_template: "{{alert_message}}"
            importance: high
          dependencies:
            - check_alerts
    """

    def __call__(
        self,
        name: str,
        to: list[str] | None = None,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
        subject: str = "",
        body_template: str = "",
        body_type: str = "text",
        importance: str = "normal",
        save_to_sent: bool = True,
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create an Outlook sender node specification.

        Parameters
        ----------
        name : str
            Node name
        to : list[str], optional
            List of recipient email addresses (can be templated from input)
        cc : list[str], optional
            List of CC recipients
        bcc : list[str], optional
            List of BCC recipients
        subject : str
            Email subject (supports Jinja2 templating)
        body_template : str
            Email body template (supports Jinja2 templating)
        body_type : str
            Body content type: 'text' or 'html' (default: 'text')
        importance : str
            Email importance: 'low', 'normal', 'high' (default: 'normal')
        save_to_sent : bool
            Save copy to Sent Items (default: True)
        deps : list[str], optional
            Dependency node names
        **kwargs : Any
            Additional node parameters

        Returns
        -------
        NodeSpec
            Node specification ready for execution
        """
        wrapped_fn = self._create_sender_function(
            name, to, cc, bcc, subject, body_template, body_type, importance, save_to_sent
        )

        input_schema = {"input_data": dict | None}
        output_model = OutlookSenderOutput

        input_model = self.create_pydantic_model(f"{name}Input", input_schema)

        node_params = {
            "to": to,
            "cc": cc,
            "bcc": bcc,
            "subject": subject,
            "body_template": body_template,
            "body_type": body_type,
            "importance": importance,
            "save_to_sent": save_to_sent,
            **kwargs,
        }

        return NodeSpec(
            name=name,
            fn=wrapped_fn,
            in_model=input_model,
            out_model=output_model,
            deps=frozenset(deps or []),
            params=node_params,
        )

    def _create_sender_function(
        self,
        name: str,
        to: list[str] | None,
        cc: list[str] | None,
        bcc: list[str] | None,
        subject: str,
        body_template: str,
        body_type: str,
        importance: str,
        save_to_sent: bool,
    ) -> Any:
        """Create the email sending function."""

        async def send_email(input_data: Any = None) -> dict[str, Any]:
            """Send email via Microsoft Graph API."""
            import os

            from jinja2 import Template

            # Get credentials from environment
            client_id = os.environ.get("MICROSOFT_CLIENT_ID")
            client_secret = os.environ.get("MICROSOFT_CLIENT_SECRET")
            tenant_id = os.environ.get("MICROSOFT_TENANT_ID")

            if not all([client_id, client_secret, tenant_id]):
                raise ValueError(
                    "Microsoft Graph API credentials not configured. "
                    "Set MICROSOFT_CLIENT_ID, MICROSOFT_CLIENT_SECRET, and MICROSOFT_TENANT_ID environment variables."
                )

            try:
                from azure.identity import ClientSecretCredential
                from msgraph import GraphServiceClient
                from msgraph.generated.models.body_type import BodyType
                from msgraph.generated.models.email_address import EmailAddress
                from msgraph.generated.models.importance import Importance
                from msgraph.generated.models.item_body import ItemBody
                from msgraph.generated.models.message import Message
                from msgraph.generated.models.recipient import Recipient
                from msgraph.generated.users.item.send_mail.send_mail_post_request_body import (
                    SendMailPostRequestBody,
                )
            except ImportError as e:
                raise ImportError(
                    "Microsoft Graph SDK not installed. Install with: pip install azure-identity msgraph-sdk"
                ) from e

            # Prepare template context from input_data
            context = {}
            if isinstance(input_data, dict):
                context = input_data

            # Render templates
            rendered_subject = Template(subject).render(**context)
            rendered_body = Template(body_template).render(**context)

            # Get recipients - from spec or from input_data
            recipients_to = to or context.get("to", [])
            recipients_cc = cc or context.get("cc", [])
            recipients_bcc = bcc or context.get("bcc", [])

            if not recipients_to:
                raise ValueError("No recipients specified. Set 'to' in spec or provide in input_data.")

            # Create credential and client
            credential = ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret,
            )
            client = GraphServiceClient(credential)

            # Build message
            def make_recipient(email: str) -> Recipient:
                recipient = Recipient()
                recipient.email_address = EmailAddress()
                recipient.email_address.address = email
                return recipient

            message = Message()
            message.subject = rendered_subject
            message.body = ItemBody()
            message.body.content_type = BodyType.Html if body_type.lower() == "html" else BodyType.Text
            message.body.content = rendered_body
            message.to_recipients = [make_recipient(r) for r in recipients_to]

            if recipients_cc:
                message.cc_recipients = [make_recipient(r) for r in recipients_cc]
            if recipients_bcc:
                message.bcc_recipients = [make_recipient(r) for r in recipients_bcc]

            # Set importance
            importance_map = {
                "low": Importance.Low,
                "normal": Importance.Normal,
                "high": Importance.High,
            }
            message.importance = importance_map.get(importance.lower(), Importance.Normal)

            # Send the message
            request_body = SendMailPostRequestBody()
            request_body.message = message
            request_body.save_to_sent_items = save_to_sent

            await client.me.send_mail.post(request_body)

            return {
                "message_id": message.id or "sent",
                "subject": rendered_subject,
                "recipients": recipients_to,
                "success": True,
            }

        send_email.__name__ = f"outlook_sender_{name}"
        send_email.__doc__ = f"Send email: {subject}"

        return send_email
