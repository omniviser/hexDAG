"""Suspension signal for wait-capable pipelines.

A ``Suspended`` return value signals the orchestrator to save a checkpoint
and stop execution.  This is a data-flow signal — not an exception —
because suspension is a success path, not an error.

Example
-------
A node that sends an email and waits for a reply::

    async def send_and_wait(inputs, context):
        await send_email(inputs["to"], inputs["body"])
        return Suspended(
            event_key=f"email_reply:{inputs['conversation_id']}",
            timeout_seconds=7 * 86400,
            setup_result={"email_sent": True},
        )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Suspended:
    """Return value that tells the orchestrator to suspend execution.

    Attributes
    ----------
    event_key : str
        Correlation key for the external event that will resume
        this pipeline (e.g. ``"email_reply:conv-123"``).
    timeout_seconds : float | None
        How long to wait before timing out.  ``None`` = wait forever.
    setup_result : Any
        Data the node produced before suspending (e.g. confirmation
        that an email was sent).  Stored in the checkpoint but is
        **not** the node's final output — that comes from the
        external event on resume.
    metadata : dict[str, Any]
        Additional metadata for the wait registration.
    """

    event_key: str
    timeout_seconds: float | None = None
    setup_result: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
