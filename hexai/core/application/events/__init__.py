"""Clean event system with clear separation of concerns.

This module provides a simple event system for pipeline execution:
- Events: Simple data classes representing what happened
- ObserverManager: Fire-and-forget observability (logging, metrics)
- ControlManager: Execution control that can veto operations
- BaseEventManager: Common base class for managers
"""

from .base_manager import BaseEventManager
from .context import ExecutionContext
from .control_manager import ControlHandler, ControlHandlerBase, ControlManager
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
from .models import ControlResponse, ControlSignal, HandlerMetadata, Observer
from .null_manager import NULL_CONTROL_MANAGER, NULL_OBSERVER_MANAGER
from .observer_manager import ObserverManager
from .observers import (
    FileObserver,
    LoggingObserver,
    MetricsObserver,
    NodeStateObserver,
    WebSocketObserver,
)

__all__ = [
    # Core system
    "BaseEventManager",
    "Event",
    "ObserverManager",
    "Observer",
    "ControlManager",
    "ControlHandler",
    "ControlHandlerBase",
    "HandlerMetadata",
    "ControlResponse",
    "ControlSignal",
    "ExecutionContext",
    # Null implementations for testing
    "NULL_CONTROL_MANAGER",
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
