"""Clean event system with clear separation of concerns.

This module provides a simple event system for pipeline execution:
- Events: Simple data classes representing what happened
- ObserverManager: Fire-and-forget observability (logging, metrics)
- EventBus: Execution control that can veto operations
"""

from .bus import ControlHandler, ControlHandlerBase, EventBus
from .context import ExecutionContext
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
from .manager import ObserverManager
from .models import ControlResponse, ControlSignal, HandlerMetadata, Observer
from .null_manager import NULL_EVENT_BUS, NULL_OBSERVER_MANAGER
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
    "ControlHandlerBase",
    "HandlerMetadata",
    "ControlResponse",
    "ControlSignal",
    "ExecutionContext",
    # Null implementations for testing
    "NULL_EVENT_BUS",
    "NULL_OBSERVER_MANAGER",
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
