"""Console notification adapter — logs notifications to the terminal.

Development/default implementation of the ``Messaging`` port's
``SupportsNotification`` capability.  Use it to see approval requests
and alerts during local runs; swap in a Slack, email, or webhook adapter
for production without changing the pipeline.
"""

from __future__ import annotations

from typing import Any

from hexdag.kernel.logging import get_logger
from hexdag.kernel.ports.messaging import Messaging, SupportsNotification
from hexdag.stdlib.adapters.base import HexDAGAdapter

logger = get_logger(__name__)


class ConsoleNotificationAdapter(
    HexDAGAdapter,
    Messaging,
    SupportsNotification,
    yaml_alias="console_notification",
    port="notification",
):
    """Notification adapter that writes to the hexDAG logger.

    Also keeps an in-memory history of sent notifications (``sent``)
    so callers can inspect what was delivered.

    Parameters
    ----------
    log_level : str
        Log level used for emitted notifications ("info", "warning", ...).

    Examples
    --------
    YAML port declaration::

        ports:
          notification:
            adapter: console_notification

    Python usage::

        adapter = ConsoleNotificationAdapter()
        await adapter.asend("Order 42 needs approval", title="Approval required")
        adapter.sent[0]["message"]
        # 'Order 42 needs approval'
    """

    def __init__(self, log_level: str = "info", **kwargs: Any) -> None:
        """Create the adapter; *log_level* labels each console line."""
        self.log_level = log_level
        self.sent: list[dict[str, Any]] = []

    async def asend(
        self,
        message: str,
        *,
        title: str | None = None,
        channel: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log the notification and record it in ``sent``."""
        record = {
            "message": message,
            "title": title,
            "channel": channel,
            "metadata": metadata or {},
        }
        self.sent.append(record)

        header = f"[{title}] " if title else ""
        target = f" → {channel}" if channel else ""
        log_fn = getattr(logger, self.log_level, logger.info)
        log_fn("🔔 {}{}{}", header, message, target)
