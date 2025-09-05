"""Pipeline event system with consolidated events."""

from .base import EventType, Observer, PipelineEvent, SyncObserver
from .bus import EventBus, HandlerPriority
from .events import ExecutionEvent, ExecutionLevel, ExecutionPhase, HookEvent, LLMEvent, MetaEvent
from .manager import PipelineEventManager
from .observers import (
    FileObserver,
    LoggingObserver,
    MetricsObserver,
    NodeObserver,
    WebSocketObserver,
)

__all__ = [
    # Base classes
    "EventType",
    "Observer",
    "PipelineEvent",
    "SyncObserver",
    "HandlerPriority",
    "EventBus",
    "PipelineEventManager",
    # Consolidated events
    "ExecutionEvent",
    "ExecutionLevel",
    "ExecutionPhase",
    "LLMEvent",
    "HookEvent",
    "MetaEvent",
    # Observers
    "LoggingObserver",
    "MetricsObserver",
    "NodeObserver",
    "WebSocketObserver",
    "FileObserver",
]
