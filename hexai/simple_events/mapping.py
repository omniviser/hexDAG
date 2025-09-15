"""
Mapping from existing event class names to the canonical format.

This module provides:
- _MAP_CLASS_TO_TYPE: dict[str, tuple[namespace, action]]
- build_event_type: validate and join namespace + action
- map_classname_to_event_type: convert class name to event_type
"""

from .types import APPROVED_ACTIONS, EVENT_TYPE_RE, Namespace

#: Single source of truth for mapping class names to taxonomy.
_MAP_CLASS_TO_TYPE: dict[str, tuple[str, str]] = {
    # Pipeline
    "PipelineStartedEvent": (Namespace.pipeline.value, "started"),
    "PipelineCompletedEvent": (Namespace.pipeline.value, "completed"),
    "PipelineFailedEvent": (Namespace.pipeline.value, "failed"),
    # DAG
    "DagStartedEvent": (Namespace.dag.value, "started"),
    "DagCompletedEvent": (Namespace.dag.value, "completed"),
    "DagFailedEvent": (Namespace.dag.value, "failed"),
    # Wave
    "WaveStartedEvent": (Namespace.wave.value, "started"),
    "WaveCompletedEvent": (Namespace.wave.value, "completed"),
    # Node
    "NodeStartedEvent": (Namespace.node.value, "started"),
    "NodeCompletedEvent": (Namespace.node.value, "completed"),
    "NodeFailedEvent": (Namespace.node.value, "failed"),
    "NodeSkippedEvent": (Namespace.node.value, "skipped"),
    # Policy / Control decisions
    "PolicyDecisionEvent": (Namespace.policy.value, "decision"),
    "ControlDecisionEvent": (Namespace.policy.value, "decision"),
    # Observer outcomes
    "ObserverTimeoutEvent": (Namespace.observer.value, "timeout"),
    "ObserverErrorEvent": (Namespace.observer.value, "error"),
    # Registry
    "RegistryResolvedEvent": (Namespace.registry.value, "resolved"),
    "RegistryMissingEvent": (Namespace.registry.value, "missing"),
}


def build_event_type(namespace: str, action: str) -> str:
    """
    Build and validate canonical event_type string.

    Parameters
    ----------
    namespace : str
        One of the approved namespaces.
    action : str
        One of the approved actions for that namespace.

    Returns
    -------
    str
        The event type in <namespace>:<action> format.

    Raises
    ------
    ValueError
        If namespace or action is not approved, or regex does not match.
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

    Parameters
    ----------
    class_name : str
        The name of the event class, e.g. 'PipelineStartedEvent'.

    Returns
    -------
    str
        The canonical event type, e.g. 'pipeline:started'.

    Raises
    ------
    KeyError
        If the class name is not found in the mapping.
    """
    try:
        ns, act = _MAP_CLASS_TO_TYPE[class_name]
    except KeyError as err:
        raise KeyError(f"unmapped event class: {class_name}") from err
    return build_event_type(ns, act)
