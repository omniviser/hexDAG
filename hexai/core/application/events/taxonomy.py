"""Event taxonomy: registry metadata, envelope building, validation."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from .events import EVENT_REGISTRY, Event, EventSpec

# Canonical namespace/action definitions
EVENT_TYPE_RE = re.compile(r"^[a-z]+:[a-z]+$")

APPROVED_ACTIONS: dict[str, set[str]] = {
    "pipeline": {"started", "completed", "failed"},
    "dag": {"started", "completed", "failed"},
    "wave": {"started", "completed"},
    "node": {"started", "completed", "failed", "skipped"},
    "policy": {"decision"},
    "observer": {"timeout", "error"},
    "registry": {"resolved", "missing"},
    "tool": {"called", "completed"},
    "llm": {"prompt", "response"},
    "port": set(),
    "memory": set(),
}


class Severity(StrEnum):
    """Severity levels used in the event envelope."""

    info = "info"
    warn = "warn"
    error = "error"


@dataclass(slots=True)
class EventContext:
    """Execution context supplied when building an envelope."""

    pipeline: str
    pipeline_run_id: str
    tenant: str | None = None
    project: str | None = None
    environment: str | None = None
    correlation_id: str | None = None


def generate_event_id(preferred: str | None = None) -> str:
    """Generate a stable string identifier for an event."""
    if preferred:
        return str(preferred)

    func = getattr(uuid, "uuid7", None)
    if func is not None:
        return str(func())
    return str(uuid.uuid4())


def _now_rfc3339_ms() -> str:
    """Return current UTC timestamp with millisecond precision."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _coerce(value: Any) -> Any:
    """Convert common non-JSON types into serializable representations."""
    if isinstance(value, Exception):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover - fallback path
            return str(value)
    return value


def _ensure_json_serializable(payload: Any) -> None:
    """Verify payload can be encoded as JSON using stdlib encoder.

    Raises
    ------
    TypeError
        If ``payload`` cannot be JSON-encoded using the standard library ``json`` module.
    """
    try:
        json.dumps(payload)
    except TypeError as exc:  # pragma: no cover - surfaced as ValueError via caller
        raise TypeError(f"attrs not JSON-serializable: {exc}") from exc


class EventEnvelope(BaseModel):
    """Pydantic model describing the canonical event envelope."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    event_type: str
    event_id: str
    timestamp: str
    pipeline: str
    pipeline_run_id: str
    severity: Severity
    attrs: dict[str, Any]
    node: str | None = None
    wave: int | None = None
    tenant: str | None = None
    project: str | None = None
    environment: str | None = None
    correlation_id: str | None = None

    @field_validator("event_type")
    @classmethod
    def _validate_event_type_field(cls, value: str) -> str:
        validate_event_type(value)
        return value

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp_field(cls, value: str) -> str:
        _validate_timestamp(value)
        return value

    @field_validator("attrs")
    @classmethod
    def _validate_attrs_field(cls, value: dict[str, Any]) -> dict[str, Any]:
        _ensure_json_serializable(value)
        return value


def _infer_severity(event_type: str) -> Severity:
    """Derive severity from event action."""
    _, action = event_type.split(":", 1)
    if action in {"failed", "error"}:
        return Severity.error
    return Severity.info


def _resolve_attr_fields(event: Event, spec: EventSpec) -> tuple[str, ...]:
    if spec.attr_fields is not None:
        return spec.attr_fields
    mapped = set(spec.envelope_fields.values())
    attr_fields: list[str] = []
    for field in fields(event):
        if not field.init:
            continue
        if field.name in mapped:
            continue
        attr_fields.append(field.name)
    return tuple(attr_fields)


def build_envelope(event: Event, context: EventContext) -> dict[str, Any]:
    """Convert an internal event object into a canonical envelope dict.

    Raises
    ------
    KeyError
        If the event class is not registered in ``EVENT_REGISTRY``.
    AttributeError
        If the event instance is missing an attribute required by the
        registry mapping.
    ValueError
        If the generated payload fails envelope validation.
    """
    class_name = type(event).__name__
    try:
        spec = EVENT_REGISTRY[class_name]
    except KeyError as exc:
        raise KeyError(f"unmapped event class: {class_name}") from exc

    payload: dict[str, Any] = {
        "event_type": spec.event_type,
        "event_id": generate_event_id(getattr(event, "event_id", None)),
        "timestamp": _now_rfc3339_ms(),
        "pipeline": context.pipeline,
        "pipeline_run_id": context.pipeline_run_id,
        "severity": _infer_severity(spec.event_type).value,
        "attrs": {},
    }

    # Optional context fields when provided
    for field in ("tenant", "project", "environment", "correlation_id"):
        value = getattr(context, field)
        if value:
            payload[field] = value

    # Populate mapped top-level fields from the event data
    for target, attr_name in spec.envelope_fields.items():
        if not hasattr(event, attr_name):
            raise AttributeError(
                f"Event '{class_name}' missing attribute '{attr_name}' required for '{target}'"
            )
        value = getattr(event, attr_name)
        payload[target] = _coerce(value)

    # Build attrs from declared event fields
    for attr_name in _resolve_attr_fields(event, spec):
        if not hasattr(event, attr_name):
            continue
        payload["attrs"][attr_name] = _coerce(getattr(event, attr_name))

    try:
        envelope_model = EventEnvelope.model_validate(payload)
    except ValidationError as exc:  # pragma: no cover - handled via ValueError for callers
        raise ValueError(str(exc)) from exc

    return envelope_model.model_dump(exclude_none=True)


def validate_event_type(event_type: str) -> None:
    """Ensure event_type follows namespace:action pattern and is approved.

    Raises
    ------
    ValueError
        If the provided event type does not match the canonical pattern or
        is not part of the approved namespace/action sets.
    """
    if not EVENT_TYPE_RE.match(event_type):
        raise ValueError(f"event_type not matching ^[a-z]+:[a-z]+$: {event_type}")
    namespace, action = event_type.split(":", 1)
    actions = APPROVED_ACTIONS.get(namespace)
    if actions is None or (actions and action not in actions):
        raise ValueError(f"event_type not in approved sets: {event_type}")


def _validate_timestamp(ts: str) -> None:
    try:
        datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be ISO 8601 compatible") from exc


REQUIRED_FIELDS = (
    "event_type",
    "event_id",
    "timestamp",
    "pipeline",
    "pipeline_run_id",
    "severity",
    "attrs",
)


def validate_envelope(envelope: dict[str, Any]) -> None:
    """Validate a complete event envelope against required fields and rules.

    Raises
    ------
    ValueError
        If mandatory fields are missing or the payload fails schema validation.
    """
    for field_name in REQUIRED_FIELDS:
        if field_name not in envelope:
            raise ValueError(f"missing field: {field_name}")

    try:
        EventEnvelope.model_validate(envelope)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
