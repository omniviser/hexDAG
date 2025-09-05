"""Clean event system with clear separation of concerns.

This module provides a simple event system for pipeline execution:
- Events: Simple data classes representing what happened
- ObserverManager: Fire-and-forget observability (logging, metrics)
- EventBus: Execution control that can veto operations
"""

from .bus import ControlHandler, EventBus
from .context import ExecutionContext
from .control import ControlResponse, ControlSignal
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
    "ObserverManager",
    "Observer",
    "EventBus",
    "ControlHandler",
    "ControlResponse",
    "ControlSignal",
    "ExecutionContext",
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
