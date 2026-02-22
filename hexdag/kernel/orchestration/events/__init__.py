"""Event system for the Hex-DAG framework.

Clean, simplified event system with clear separation of concerns:
- events.py: Event data classes (just data, no behavior)
- observers/: Observer implementations for monitoring and observability
"""

# Event classes
from .events import (
    CheckpointRestored,
    CheckpointSaved,
    Event,
    HealthCheckCompleted,
    LLMPromptSent,
    LLMResponseReceived,
    NodeCancelled,
    NodeCompleted,
    NodeFailed,
    NodeSkipped,
    NodeStarted,
    PipelineCancelled,
    PipelineCompleted,
    PipelineStarted,
    PolicyEvaluated,
    PolicyFallback,
    PolicyRetry,
    PolicySkipped,
    PolicyTriggered,
    ToolCalled,
    ToolCompleted,
    WaveCompleted,
    WaveStarted,
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
WAVE_EVENTS = (WaveStarted, WaveCompleted)
PIPELINE_EVENTS = (PipelineStarted, PipelineCompleted, PipelineCancelled)
LLM_EVENTS = (LLMPromptSent, LLMResponseReceived)
TOOL_EVENTS = (ToolCalled, ToolCompleted)
POLICY_EVENTS = (PolicyEvaluated, PolicyTriggered, PolicyRetry, PolicySkipped, PolicyFallback)
CHECKPOINT_EVENTS = (CheckpointSaved, CheckpointRestored)
HEALTH_EVENTS = (HealthCheckCompleted,)

# Commonly used combinations
ALL_NODE_EVENTS = NODE_LIFECYCLE_EVENTS + WAVE_EVENTS
ALL_EXECUTION_EVENTS = NODE_LIFECYCLE_EVENTS + WAVE_EVENTS + PIPELINE_EVENTS
ALL_MONITORING_EVENTS = NODE_LIFECYCLE_EVENTS + LLM_EVENTS + TOOL_EVENTS + POLICY_EVENTS
COST_PROFILING_EVENTS = PIPELINE_EVENTS + NODE_LIFECYCLE_EVENTS + WAVE_EVENTS + LLM_EVENTS

__all__ = [
    # Events
    "Event",
    "NodeStarted",
    "NodeCompleted",
    "NodeFailed",
    "NodeCancelled",
    "NodeSkipped",
    "WaveStarted",
    "WaveCompleted",
    "PipelineStarted",
    "PipelineCompleted",
    "PipelineCancelled",
    "LLMPromptSent",
    "LLMResponseReceived",
    "ToolCalled",
    "ToolCompleted",
    "PolicyEvaluated",
    "PolicyTriggered",
    "PolicySkipped",
    "PolicyFallback",
    "PolicyRetry",
    "CheckpointSaved",
    "CheckpointRestored",
    "HealthCheckCompleted",
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
    "LLM_EVENTS",
    "TOOL_EVENTS",
    "POLICY_EVENTS",
    "CHECKPOINT_EVENTS",
    "HEALTH_EVENTS",
    "ALL_NODE_EVENTS",
    "ALL_EXECUTION_EVENTS",
    "ALL_MONITORING_EVENTS",
    "COST_PROFILING_EVENTS",
]
