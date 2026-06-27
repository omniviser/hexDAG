"""Observer implementations for the hexDAG event system.

Observers are passive listeners that accumulate metrics during pipeline runs.
They live in stdlib because they are implementations, not kernel contracts.
"""

from hexdag.stdlib.lib.observers.core_observers import (
    ExecutionTracerObserver,
    SimpleLoggingObserver,
)
from hexdag.stdlib.lib.observers.opentelemetry_observer import OpenTelemetryObserver
from hexdag.stdlib.lib.observers.port_call_observers import (
    PortCallLogObserver,
    PortCallStoreObserver,
    StoredPortCall,
)
from hexdag.stdlib.lib.observers.state_transition_observer import (
    StateTransitionObserver,
    StoredTransition,
)

__all__ = [
    # Core Observers
    "ExecutionTracerObserver",
    "SimpleLoggingObserver",
    # Exporter Observers (optional dep: hexdag[otel]).
    # Langfuse exporter lives in hexdag_plugins.langfuse (vendor-specific).
    "OpenTelemetryObserver",
    # Port Call Observers
    "PortCallStoreObserver",
    "PortCallLogObserver",
    "StoredPortCall",
    # State Transition Observer
    "StateTransitionObserver",
    "StoredTransition",
]
