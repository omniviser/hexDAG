"""
Envelope builder for the Simple Event Taxonomy.

Builds a minimal, structured payload and validates it.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .ids import generate_event_id
from .mapping import map_classname_to_event_type
from .types import Severity
from .validators import validate_envelope


@dataclass
class SimpleContext:
    """Execution context required to build the envelope."""

    pipeline: str
    pipeline_run_id: str
    tenant: str | None = None
    project: str | None = None
    environment: str | None = None
    correlation_id: str | None = None
    node: str | None = None
    wave: int | None = None


def _now_rfc3339_ms() -> str:
    """Return current UTC time in RFC3339 with milliseconds and Z."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _coerce(v: Any) -> Any:
    """Coerce values to JSON-friendly forms."""
    from enum import Enum

    if isinstance(v, Exception):
        return str(v)
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="replace")
    if hasattr(v, "isoformat"):
        try:
            return v.isoformat()  # datetimes, dates
        except Exception:
            return str(v)
    return v


def _obj_to_attrs(event: Any) -> dict[str, Any]:
    """Extract whitelisted event-specific fields as attrs."""
    attrs: dict[str, Any] = {}
    for key in ("total_waves", "total_nodes", "error_type", "retryable"):
        if hasattr(event, key):
            attrs[key] = _coerce(getattr(event, key))
    return attrs


def _infer_severity(event_type: str) -> str:
    """Map action to severity: failed|error -> error, else info."""
    _, action = event_type.split(":", 1)
    if action in ("failed", "error"):
        return Severity.error.value
    return Severity.info.value


def to_simple_event(event: Any, context: SimpleContext) -> dict[str, Any]:
    """Translate an internal event object into a canonical envelope dict."""
    class_name = type(event).__name__
    event_type = map_classname_to_event_type(class_name)

    env: dict[str, Any] = {
        "event_type": event_type,
        "event_id": generate_event_id(getattr(event, "event_id", None)),
        "timestamp": _now_rfc3339_ms(),
        "pipeline": getattr(event, "pipeline", None) or context.pipeline,
        "pipeline_run_id": getattr(event, "pipeline_run_id", None) or context.pipeline_run_id,
        "severity": _infer_severity(event_type),
        "attrs": _obj_to_attrs(event),
    }

    # Optional context fields if available
    node = getattr(event, "node", None) or getattr(event, "node_name", None) or context.node
    wave = getattr(event, "wave", None) or getattr(event, "wave_index", None) or context.wave
    if node is not None:
        env["node"] = str(node)
    if wave is not None:
        env["wave"] = int(wave)
    if context.tenant:
        env["tenant"] = context.tenant
    if context.project:
        env["project"] = context.project
    if context.environment:
        env["environment"] = context.environment
    if context.correlation_id:
        env["correlation_id"] = context.correlation_id

    validate_envelope(env)
    return env
