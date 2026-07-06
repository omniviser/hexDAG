"""Mock notification adapter for testing approval / alert flows."""

from __future__ import annotations

from typing import Any

from hexdag.kernel.ports.messaging import Messaging, SupportsNotification
from hexdag.stdlib.adapters.base import HexDAGAdapter


class MockNotification(
    HexDAGAdapter,
    Messaging,
    SupportsNotification,
    yaml_alias="mock_notification",
    port="notification",
):
    """Mock implementation of the notification capability for testing.

    Records every sent notification for inspection and can simulate
    delivery failures.

    Parameters
    ----------
    should_raise : bool
        If True, ``asend`` raises to simulate delivery failure.

    Examples
    --------
    >>> import asyncio
    >>> mock = MockNotification()
    >>> asyncio.run(mock.asend("hello", title="Hi"))
    >>> mock.sent[0]["title"]
    'Hi'
    """

    def __init__(self, should_raise: bool = False, **kwargs: Any) -> None:
        """Create the mock; set *should_raise* to simulate send failures."""
        self.should_raise = should_raise
        self.sent: list[dict[str, Any]] = []

    async def asend(
        self,
        message: str,
        *,
        title: str | None = None,
        channel: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record the notification (or raise if configured to fail)."""
        if self.should_raise:
            raise RuntimeError("MockNotification delivery failure (configured)")
        self.sent.append({
            "message": message,
            "title": title,
            "channel": channel,
            "metadata": metadata or {},
        })

    def reset(self) -> None:
        """Clear recorded notifications."""
        self.sent.clear()
