"""
Envelope builder for the Simple Event Taxonomy.

Builds a minimal, structured payload and validates it.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .ids import generate_event_id
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


def _obj_to_attrs(event: Any, exclude: set[str]) -> dict[str, Any]:
    """Extract event-specific fields excluding top-level ones."""
    attrs: dict[str, Any] = {}
    for key, value in vars(event).items():
        if key in exclude:
            continue
        attrs[key] = _coerce(value)
    return attrs


def _infer_severity(event_type: str) -> str:
    """Map action to severity: failed|error -> error, else info."""
    _, action = event_type.split(":", 1)
    if action in ("failed", "error"):
        return Severity.error.value
    return Severity.info.value


def to_simple_event(event: Any, context: SimpleContext) -> dict[str, Any]:
    """Translate an internal event object into a canonical envelope dict."""
    event_type_attr = getattr(event, "event_type", None)
    if event_type_attr is None:
        raise TypeError(f"event object {event!r} missing 'event_type'")
    event_type = event_type_attr() if callable(event_type_attr) else event_type_attr
    if not isinstance(event_type, str):
        raise TypeError(f"event_type must be str, got {type(event_type).__name__}")

    namespace, _action = event_type.split(":", 1)

    pipeline_attr_used: str | None = None
    pipeline_value = None
    if namespace == "pipeline":
        for candidate in ("pipeline", "name"):
            value = getattr(event, candidate, None)
            if value:
                pipeline_attr_used = candidate
                pipeline_value = value
                break
    else:
        value = getattr(event, "pipeline", None)
        if value:
            pipeline_attr_used = "pipeline"
            pipeline_value = value

    pipeline = pipeline_value or context.pipeline

    run_id_attr_used: str | None = None
    run_id_value = getattr(event, "pipeline_run_id", None)
    if run_id_value:
        run_id_attr_used = "pipeline_run_id"

    node_attr_used: str | None = None
    node_value = getattr(event, "node", None) or getattr(event, "node_name", None)
    if node_value is None and namespace == "node":
        name_value = getattr(event, "name", None)
        if name_value is not None:
            node_value = name_value
            node_attr_used = "name"
    elif node_value is not None:
        node_attr_used = "node" if hasattr(event, "node") else "node_name"

    wave_attr_used: str | None = None
    wave_value = getattr(event, "wave", None) or getattr(event, "wave_index", None)
    if wave_value is not None:
        wave_attr_used = "wave" if hasattr(event, "wave") else "wave_index"

    exclude_keys: set[str] = {"timestamp", "event_id"}
    if pipeline_attr_used:
        exclude_keys.add(pipeline_attr_used)
    if run_id_attr_used:
        exclude_keys.add(run_id_attr_used)
    if node_attr_used:
        exclude_keys.add(node_attr_used)
    if wave_attr_used:
        exclude_keys.add(wave_attr_used)

    env: dict[str, Any] = {
        "event_type": event_type,
        "event_id": generate_event_id(getattr(event, "event_id", None)),
        "timestamp": _now_rfc3339_ms(),
        "pipeline": pipeline,
        "pipeline_run_id": run_id_value or context.pipeline_run_id,
        "severity": _infer_severity(event_type),
        "attrs": _obj_to_attrs(event, exclude_keys),
    }

    # Optional context fields if available
    node = node_value or context.node
    wave = wave_value or context.wave
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
