"""Event system for the Hex-DAG framework.

Clean, simplified event system with clear separation of concerns:
- events.py: Event data classes (just data, no behavior)
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

__all__ = [
    # Events
    "Event",
    "NodeStarted",
    "NodeCompleted",
    "NodeFailed",
    "NodeCancelled",
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
]
