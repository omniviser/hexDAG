"""Event system for the Hex-DAG framework.

Clean, simplified event system with clear separation of concerns:
- events.py: Kernel lifecycle event data classes (just data, no behavior)
- Port events live co-located with their port definitions in kernel/ports/
- observers/: Observer implementations for monitoring and observability
"""

# Kernel lifecycle events
from .events import (
    BodyCompleted,
    BodyFailed,
    BodyStarted,
    EntityCompensationEvent,
    EntityGarbageCollected,
    EntityObligationFailed,
    Event,
    NodeCancelled,
    NodeCompleted,
    NodeFailed,
    NodeSkipped,
    NodeStarted,
    PipelineCompleted,
    PipelineStarted,
    PortCallEvent,
    ProcessCompleted,
    ProcessStarted,
    StateTransitionEvent,
    SystemCompleted,
    SystemStarted,
    TransitionContext,
    WaveCompleted,
)

# Observer implementations
from .observers import (
    AlertingObserver,
    CostProfilerObserver,
    DataQualityObserver,
    ExecutionTracerObserver,
    NodeCostMetrics,
    PerformanceMetricsObserver,
    ResourceMonitorObserver,
    SimpleLoggingObserver,
)

# Event taxonomy - grouped event types for observer filtering
NODE_LIFECYCLE_EVENTS = (NodeStarted, NodeCompleted, NodeFailed, NodeCancelled, NodeSkipped)
WAVE_EVENTS = (WaveCompleted,)
PIPELINE_EVENTS = (PipelineStarted, PipelineCompleted)
BODY_EVENTS = (BodyStarted, BodyCompleted, BodyFailed)
SYSTEM_EVENTS = (SystemStarted, ProcessStarted, ProcessCompleted, SystemCompleted)
HEALTH_EVENTS = ()  # HealthCheckEvent now lives in kernel.ports.healthcheck
PORT_CALL_EVENTS = (PortCallEvent,)  # Base for all port/adapter method calls
ENTITY_EVENTS = (
    StateTransitionEvent,
    EntityGarbageCollected,
    EntityObligationFailed,
    EntityCompensationEvent,
)

# Commonly used combinations
ALL_NODE_EVENTS = NODE_LIFECYCLE_EVENTS + WAVE_EVENTS
ALL_EXECUTION_EVENTS = NODE_LIFECYCLE_EVENTS + WAVE_EVENTS + PIPELINE_EVENTS
ALL_SYSTEM_EVENTS = SYSTEM_EVENTS + ALL_EXECUTION_EVENTS
ALL_BODY_EVENTS = BODY_EVENTS + NODE_LIFECYCLE_EVENTS

__all__ = [
    # Events
    "Event",
    "PortCallEvent",
    "NodeStarted",
    "NodeCompleted",
    "NodeFailed",
    "NodeCancelled",
    "NodeSkipped",
    "WaveCompleted",
    "PipelineStarted",
    "PipelineCompleted",
    "BodyStarted",
    "BodyCompleted",
    "BodyFailed",
    "SystemStarted",
    "ProcessStarted",
    "ProcessCompleted",
    "SystemCompleted",
    # Entity lifecycle events
    "TransitionContext",
    "StateTransitionEvent",
    "EntityGarbageCollected",
    "EntityObligationFailed",
    "EntityCompensationEvent",
    # Core Observers
    "PerformanceMetricsObserver",
    "AlertingObserver",
    "ExecutionTracerObserver",
    "SimpleLoggingObserver",
    "ResourceMonitorObserver",
    "DataQualityObserver",
    "CostProfilerObserver",
    # Observer Models
    "NodeCostMetrics",
    # Event Taxonomy
    "NODE_LIFECYCLE_EVENTS",
    "WAVE_EVENTS",
    "PIPELINE_EVENTS",
    "BODY_EVENTS",
    "SYSTEM_EVENTS",
    "HEALTH_EVENTS",
    "PORT_CALL_EVENTS",
    "ENTITY_EVENTS",
    "ALL_NODE_EVENTS",
    "ALL_EXECUTION_EVENTS",
    "ALL_BODY_EVENTS",
    "ALL_SYSTEM_EVENTS",
]
