"""Event system for the Hex-DAG framework.

Clean, simplified event system with clear separation of concerns:
- events.py: Event data classes (just data, no behavior)
- models.py: Core types, protocols, and base classes
- observer_manager.py: Observer management (read-only monitoring)
- control_manager.py: Control flow management (can affect execution)
- config.py: Configuration and null implementations
"""

# Configuration and helpers
from .config import (
    NULL_CONTROL_MANAGER,
    NULL_OBSERVER_MANAGER,
    NullControlManager,
    NullObserverManager,
    get_control_manager,
    get_observer_manager,
)

# Managers
from .control_manager import ControlManager

# Decorators
from .decorators import EventDecoratorMetadata, control_handler, observer

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
    ToolCalled,
    ToolCompleted,
    WaveCompleted,
    WaveStarted,
)

# Models and protocols
from .models import (
    BaseEventManager,
    ControlHandler,
    ControlResponse,
    ControlSignal,
    ExecutionContext,
    HandlerMetadata,
    Observer,
)
from .observer_manager import ObserverManager

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
    # Models
    "ExecutionContext",
    "ControlSignal",
    "ControlResponse",
    "HandlerMetadata",
    "Observer",
    "ControlHandler",
    "BaseEventManager",
    # Managers
    "ObserverManager",
    "ControlManager",
    # Decorators
    "EventDecoratorMetadata",
    "control_handler",
    "observer",
    # Config
    "NullObserverManager",
    "NullControlManager",
    "NULL_OBSERVER_MANAGER",
    "NULL_CONTROL_MANAGER",
    "get_observer_manager",
    "get_control_manager",
]
