"""Event system for the Hex-DAG framework.

Clean, simplified event system with clear separation of concerns:
- events.py: Event data classes (just data, no behavior)
"""

# Event classes
from .events import (
    Event,
    LLMPromptSent,
    LLMResponseReceived,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
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
    "WaveStarted",
    "WaveCompleted",
    "PipelineStarted",
    "PipelineCompleted",
    "LLMPromptSent",
    "LLMResponseReceived",
    "ToolCalled",
    "ToolCompleted",
    "PolicyEvaluated",
    "PolicyTriggered",
    "PolicySkipped",
    "PolicyFallback",
    "PolicyRetry",
]
