"""
Validation utilities for the Simple Event Taxonomy.

This module enforces lintable rules:
- event_type matches ^[a-z]+:[a-z]+$ and is approved
- event_id present
- timestamp in RFC3339 UTC with Z
- pipeline_run_id required
- attrs must be JSON-serializable
"""

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from .types import APPROVED_ACTIONS, EVENT_TYPE_RE

REQUIRED_FIELDS = (
    "event_type",
    "event_id",
    "timestamp",
    "pipeline",
    "pipeline_run_id",
    "severity",
    "attrs",
)


def _is_rfc3339_z(ts: str) -> bool:
    """Return True if timestamp is RFC3339 UTC with Z suffix and ms precision."""
    try:
        if not ts.endswith("Z"):
            return False
        datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return True
    except Exception:
        return False


def validate_event_type(et: str) -> None:
    """Check that event_type matches regex and is in approved namespace/action sets."""
    if not EVENT_TYPE_RE.match(et):
        raise ValueError(f"event_type not matching ^[a-z]+:[a-z]+$: {et}")
    ns, act = et.split(":", 1)
    if ns not in APPROVED_ACTIONS or act not in APPROVED_ACTIONS[ns]:
        raise ValueError(f"event_type not in approved sets: {et}")


def validate_json_serializable(obj: Any) -> None:
    """Raise TypeError if obj cannot be JSON-serialized."""
    try:
        json.dumps(obj)
    except TypeError as e:
        raise TypeError(f"attrs not JSON-serializable: {e}") from e


def validate_envelope(env: Mapping[str, Any]) -> None:
    """Validate a complete event envelope against required fields and rules."""
    for f in REQUIRED_FIELDS:
        if f not in env:
            raise ValueError(f"missing field: {f}")
    validate_event_type(env["event_type"])
    if not _is_rfc3339_z(env["timestamp"]):
        raise ValueError("timestamp must be RFC3339 UTC with 'Z'")
    validate_json_serializable(env["attrs"])
