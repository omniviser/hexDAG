"""
Canonical event namespaces, actions and severity levels.

This module defines:
- EVENT_TYPE_RE: regex that validates <namespace>:<action>
- Namespace: Enum of canonical namespaces
- APPROVED_ACTIONS: dict of allowed actions per namespace
- Severity: Enum of severity levels
"""

import re
from enum import Enum

#: Regex pattern validating <namespace>:<action>
EVENT_TYPE_RE = re.compile(r"^[a-z]+:[a-z]+$")


class Namespace(str, Enum):
    """Canonical event namespaces."""

    pipeline = "pipeline"
    dag = "dag"
    wave = "wave"
    node = "node"
    policy = "policy"
    observer = "observer"
    registry = "registry"
    # reserve for: port, tool, memory


#: Allowed actions for each namespace.
APPROVED_ACTIONS = {
    Namespace.pipeline.value: {"started", "completed", "failed"},
    Namespace.dag.value: {"started", "completed", "failed"},
    Namespace.wave.value: {"started", "completed"},
    Namespace.node.value: {"started", "completed", "failed", "skipped"},
    Namespace.policy.value: {"decision"},
    Namespace.observer.value: {"timeout", "error"},
    Namespace.registry.value: {"resolved", "missing"},
}


class Severity(str, Enum):
    """Severity levels used in the event envelope."""

    info = "info"
    warn = "warn"
    error = "error"
