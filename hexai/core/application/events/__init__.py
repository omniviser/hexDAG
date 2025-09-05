"""Clean event system with clear separation of concerns.

This module provides a simple event system for pipeline execution:
- Events: Simple data classes representing what happened
- ObserverManager (ObserverManager): Fire-and-forget observability (logging, metrics)
- EventBus: Execution control that can veto operations
"""

from .bus import ControlHandler, EventBus
from .events import (
    Event,
    LLMPromptSent,
    LLMResponseReceived,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    PipelineCompleted,
    PipelineStarted,
    ToolCalled,
    ToolCompleted,
    WaveCompleted,
    WaveStarted,
)
from .manager import Observer, ObserverManager
from .null_manager import get_event_manager
from .observers import (
    FileObserver,
    LoggingObserver,
    MetricsObserver,
    NodeStateObserver,
    WebSocketObserver,
)

__all__ = [
    # Core system
    "Event",
    "ObserverManager",  # Alias for backward compatibility
    "Observer",
    "EventBus",
    "ControlBus",  # Alias for backward compatibility
    "ControlHandler",
    "get_event_manager",
    # Event types
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
    # Observers
    "LoggingObserver",
    "MetricsObserver",
    "FileObserver",
    "NodeStateObserver",
    "WebSocketObserver",
]
