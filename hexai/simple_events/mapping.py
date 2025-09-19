"""
Mapping from existing event class names to the canonical format.

This module provides:
- _MAP_CLASS_TO_TYPE: dict[str, tuple[namespace, action]]
- build_event_type: validate and join namespace + action
- map_classname_to_event_type: convert class name to event_type.
"""

from .types import APPROVED_ACTIONS, EVENT_TYPE_RE

#: Single source of truth for mapping class names to taxonomy.
_MAP_CLASS_TO_TYPE: dict[str, tuple[str, str]] = {
    # Pipeline
    "PipelineStartedEvent": ("pipeline", "started"),
    "PipelineCompletedEvent": ("pipeline", "completed"),
    "PipelineFailedEvent": ("pipeline", "failed"),
    # DAG
    "DagStartedEvent": ("dag", "started"),
    "DagCompletedEvent": ("dag", "completed"),
    "DagFailedEvent": ("dag", "failed"),
    # Wave
    "WaveStartedEvent": ("wave", "started"),
    "WaveCompletedEvent": ("wave", "completed"),
    # Node
    "NodeStartedEvent": ("node", "started"),
    "NodeCompletedEvent": ("node", "completed"),
    "NodeFailedEvent": ("node", "failed"),
    "NodeSkippedEvent": ("node", "skipped"),
    # Policy / Control decisions
    "PolicyDecisionEvent": ("policy", "decision"),
    "ControlDecisionEvent": ("policy", "decision"),
    # Observer outcomes
    "ObserverTimeoutEvent": ("observer", "timeout"),
    "ObserverErrorEvent": ("observer", "error"),
    # Registry
    "RegistryResolvedEvent": ("registry", "resolved"),
    "RegistryMissingEvent": ("registry", "missing"),
}


def build_event_type(namespace: str, action: str) -> str:
    """
    Build and validate canonical event_type string.
    """
    if namespace not in APPROVED_ACTIONS:
        raise ValueError(f"namespace not approved: {namespace}")
    if action not in APPROVED_ACTIONS[namespace]:
        raise ValueError(f"action not approved for {namespace}: {action}")
    et = f"{namespace}:{action}"
    if not EVENT_TYPE_RE.match(et):
        raise ValueError(f"event_type invalid: {et}")
    return et


def map_classname_to_event_type(class_name: str) -> str:
    """
    Convert event class name into canonical <namespace>:<action>.
    """
    try:
        ns, act = _MAP_CLASS_TO_TYPE[class_name]
    except KeyError as err:
        raise KeyError(f"unmapped event class: {class_name}") from err
    return build_event_type(ns, act)
