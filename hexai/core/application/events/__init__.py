"""Event system for the Hex-DAG framework.

Provides event data classes for pipeline observability and
basic observer implementations.
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

# Observer implementations
from .observers import (
    CollectingObserver,
    CompositeObserver,
    FilteringObserver,
    LoggingObserver,
    NullObserver,
    Observer,
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
    "PolicyFallback",
    "PolicyRetry",
    "PolicySkipped",
    "PolicyTriggered",
    # Observers
    "Observer",
    "LoggingObserver",
    "CollectingObserver",
    "FilteringObserver",
    "CompositeObserver",
    "NullObserver",
]
