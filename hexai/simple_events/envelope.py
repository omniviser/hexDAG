"""
Envelope builder for the Simple Event Taxonomy.

Builds a minimal, structured payload and validates it.
"""

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .ids import generate_event_id
from .mapping import map_classname_to_event_type
from .types import Severity
from .validators import validate_envelope


@dataclass
class SimpleContext:
    """Execution context required to build the envelope."""

    pipeline: str
    pipeline_run_id: str
    tenant: Optional[str] = None
    project: Optional[str] = None
    environment: Optional[str] = None
    correlation_id: Optional[str] = None
    node: Optional[str] = None
    wave: Optional[int] = None


def _now_rfc3339_ms() -> str:
    """Return current UTC time in RFC3339 with milliseconds and Z."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


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


def _obj_to_attrs(event: Any) -> Dict[str, Any]:
    if is_dataclass(event) and not isinstance(event, type):
        data = asdict(event)  # instance OK
    elif hasattr(event, "__dict__"):
        data = dict(event.__dict__)
    else:
        data = {}

    # Drop envelope-reserved keys if present on event object
    for k in (
        "event_id",
        "timestamp",
        "pipeline",
        "pipeline_run_id",
        "node",
        "node_name",
        "wave",
        "wave_index",
        "severity",
        "event_type",
    ):
        data.pop(k, None)

    return {k: _coerce(v) for k, v in data.items()}


def _infer_severity(event_type: str) -> str:
    """Map action to severity: failed|error -> error, else info."""
    _, action = event_type.split(":", 1)
    if action in ("failed", "error"):
        return Severity.error.value
    return Severity.info.value


def to_simple_event(event: Any, context: SimpleContext) -> Dict[str, Any]:
    """Translate an internal event object into a canonical envelope dict."""
    class_name = type(event).__name__
    event_type = map_classname_to_event_type(class_name)

    env: Dict[str, Any] = {
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
