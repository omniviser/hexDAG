"""Port interface for outbound notifications (Slack, email, console, ...).

The ``Notification`` port is the framework's channel for reaching humans:
approval requests, pipeline alerts, human-in-the-loop prompts.  It pairs
naturally with ``WaitNode`` — notify the approver, suspend the pipeline,
resume when their decision arrives::

    notify (Notification port)  →  wait_node (suspend)  →  act on decision
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Notification(Protocol):
    """Port interface for outbound notification delivery.

    Implementations deliver messages to humans or external systems:
    - Console/log output (development)
    - Slack / Teams webhooks
    - Email (see hexdag_plugins email adapters)
    - PagerDuty / Opsgenie

    The port is intentionally minimal: one message, optional title,
    optional channel routing, and a metadata dict for adapter-specific
    extras (approval links, correlation keys, severity, ...).

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
