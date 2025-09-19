"""
Canonical event namespaces, actions and severity levels.

This module defines:
- EVENT_TYPE_RE: regex that validates <namespace>:<action>
- APPROVED_ACTIONS: dict of allowed actions per namespace
- Severity: Enum of severity levels
"""

import re
from enum import Enum

# Regex pattern validating <namespace>:<action>
EVENT_TYPE_RE = re.compile(r"^[a-z]+:[a-z]+$")

# Allowed actions for each namespace (single source of truth)
APPROVED_ACTIONS: dict[str, set[str]] = {
    "pipeline": {"started", "completed", "failed"},
    "dag": {"started", "completed", "failed"},
    "wave": {"started", "completed"},
    "node": {"started", "completed", "failed", "skipped"},
    "policy": {"decision"},
    "observer": {"timeout", "error"},
    "registry": {"resolved", "missing"},
}


class Severity(str, Enum):
    """Severity levels used in the event envelope."""

    info = "info"
    warn = "warn"
    error = "error"
