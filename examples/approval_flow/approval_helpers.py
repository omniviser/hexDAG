"""Business logic for the approval-flow example.

Referenced from approval_pipeline.yaml by module path
(``approval_helpers.prepare_request`` etc.) — run_approval_flow.py adds
this directory to ``sys.path``.
"""

from __future__ import annotations

from typing import Any

from hexdag.kernel.context import get_port


def _to_plain_dict(value: Any) -> dict[str, Any]:
    """Unwrap pydantic-validated node inputs into a plain dict."""
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    return value if isinstance(value, dict) else {"value": value}


def prepare_request(order: Any) -> dict[str, Any]:
    """Shape the incoming order into an approval request."""
    data = _to_plain_dict(order)
    return {
        "order_id": data.get("order_id", "unknown"),
        "amount": data.get("amount", 0),
        "summary": f"Order {data.get('order_id')} for ${data.get('amount')}",
    }


async def notify_approver(request: Any) -> dict[str, Any]:
    """Send the approval request through the Notification port."""
    data = _to_plain_dict(request)
    if "request" in data:  # unwrap input_mapping field
        data = _to_plain_dict(data["request"])

    notification = get_port("notification")
    if notification is None:
        raise RuntimeError("Notification port not configured")

    await notification.asend(
        f"{data['summary']} — reply approve/reject",
        title="Approval required",
        metadata={"event_key": f"approval:{data['order_id']}"},
    )
    # wait_node renders its event_key template from this output
    return {"order_id": data["order_id"], "notified": True}


def finalize(decision: Any) -> dict[str, Any]:
    """Act on the human decision (runs after resume)."""
    data = _to_plain_dict(decision)
    if "decision" in data:  # unwrap input_mapping field
        data = _to_plain_dict(data["decision"])

    approved = bool(data.get("approved"))
    return {
        "status": "approved" if approved else "rejected",
        "approver": data.get("approver", "unknown"),
        "comment": data.get("comment", ""),
    }
