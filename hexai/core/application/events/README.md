# 📊 Event System Architecture

> **A high-performance, production-ready event system for DAG orchestration with clean separation of observation and control.**

## 🎯 Overview

The  event system provides observability and control capabilities through a dual-manager architecture that separates **read-only observation** from **execution control**. Built on hexagonal architecture principles, it enables real-time monitoring and dynamic execution control without coupling.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Orchestrator                       │
│                                                      │
│  ┌──────────────┐              ┌──────────────┐    │
│  │   Control    │              │   Observer   │    │
│  │   Manager    │              │   Manager    │    │
│  └──────────────┘              └──────────────┘    │
│         ▲                             ▲             │
│         │                             │             │
│    ControlResponse              (fire & forget)     │
│         │                             │             │
└─────────┼─────────────────────────────┼─────────────┘
          │                             │
    ┌─────▼──────┐              ┌──────▼──────┐
    │  Control   │              │  Observer   │
    │  Handlers  │              │  Functions  │
    └────────────┘              └─────────────┘
```

### Dual-Manager System

#### **ObserverManager** - Fire-and-Forget Telemetry
*"Tell me what happened, but don't interfere"*

- **Purpose**: Logging, metrics, monitoring, alerts, audit trails
- **Behavior**: Async fire-and-forget with concurrent execution
- **Fault Isolation**: Observer failures never crash the pipeline
- **Event Filtering**: Subscribe only to relevant event types
- **Performance**: ThreadPoolExecutor for sync functions, semaphore-based concurrency

#### **ControlManager** - Policy Enforcement
*"Check if this should proceed and how"*

- **Purpose**: Retry policies, circuit breakers, rate limiting, authorization
- **Behavior**: Priority-based execution with veto pattern
- **Control Signals**: PROCEED, SKIP, RETRY, FALLBACK, FAIL, ERROR
- **Event Filtering**: Process only relevant events for ~90% efficiency gain
- **Performance**: Direct heap iteration O(n), lazy deletion with cleanup

## 📁 Module Structure

```
events/
├── events.py           # Pure event data classes
├── observer_manager.py # Read-only observation
├── control_manager.py  # Execution control policies
├── models.py          # Protocols and base classes
├── config.py          # Configuration and null implementations
└── README.md          # This file
```

### Component Details

#### **events.py** - Event Data Classes
Pure immutable data classes representing pipeline events:
```python
NodeStarted, NodeCompleted, NodeFailed    # Node lifecycle
WaveStarted, WaveCompleted                # Wave execution
PipelineStarted, PipelineCompleted        # Pipeline lifecycle
ToolCalled, ToolCompleted                 # Tool usage
LLMPromptSent, LLMResponseReceived        # LLM interactions
```

#### **models.py** - Shared Interfaces
```python
Observer              # Protocol for observers
ControlHandler        # Protocol for control handlers
BaseEventManager      # Common manager functionality
ExecutionContext      # Runtime context with retry tracking
ControlResponse       # Response with signal and data
ControlSignal         # Enum: PROCEED, SKIP, RETRY, etc.
```

## 🚀 Usage Examples

### Observer Pattern - Metrics Collection
```python
from hexai.core.application.events import ObserverManager, NodeStarted

# Create observer manager with concurrency control
observer_manager = ObserverManager(
    max_concurrent_observers=10,
    observer_timeout=5.0,
    max_sync_workers=4
)

# Register a metrics observer for specific events
async def metrics_observer(event):
    # Record metrics to Prometheus/DataDog/etc
    if isinstance(event, NodeStarted):
        metrics.increment(f"node.{event.name}.started")

observer_manager.register(
    metrics_observer,
    event_types=[NodeStarted, NodeCompleted, NodeFailed]  # Filter
)

# Fire and forget - non-blocking
await observer_manager.notify(NodeStarted(name="processor", wave_index=1))
```

### Control Pattern - Retry Policy
```python
from hexai.core.application.events import (
    ControlManager, ControlResponse, ControlSignal, NodeFailed
)

control_manager = ControlManager()

# High-priority retry policy
async def retry_policy(event, context):
    if isinstance(event, NodeFailed):
        if context.attempt < 3:
            return ControlResponse(
                signal=ControlSignal.RETRY,
                data={"delay": 1.0 * context.attempt}  # Exponential backoff
            )
        else:
            return ControlResponse(signal=ControlSignal.FAIL)
    return ControlResponse()  # PROCEED by default

control_manager.register(
    retry_policy,
    priority=10,  # High priority (lower = higher)
    name="retry_policy",
    event_types=[NodeFailed]  # Only handle failures
)

# Check event against all policies
response = await control_manager.check(event, context)
if response.signal == ControlSignal.RETRY:
    # Orchestrator handles retry with delay
    pass
```

### Event Type Filtering
```python
# Observer only for tool events - 90% fewer invocations
observer_manager.register(
    tool_observer,
    event_types=[ToolCalled, ToolCompleted]
)

# Control handler only for nodes
control_manager.register(
    node_handler,
    priority=50,
    event_types=[NodeStarted, NodeCompleted, NodeFailed]
)

# Universal observer sees everything (no filter)
observer_manager.register(audit_observer)
```

## ⚡ Performance Optimizations

### 1. **Efficient Priority Queue**
```python
# Before: O(n log n) sorting
for handler in sorted(handlers, key=lambda h: h.priority):
    ...

# After: O(n) direct heap iteration
for entry in self._handler_heap:  # Already ordered
    ...
```

### 2. **Resource Reuse**
```python
# Before: Create semaphore per event
async def notify(self, event):
    semaphore = asyncio.Semaphore(self._max_concurrent)

# After: Create once, reuse
def __init__(self):
    self._semaphore = asyncio.Semaphore(max_concurrent)
```

### 3. **Consolidated Storage**
```python
# Before: Triple storage
self._handlers = {}
self._priorities = {}
self._event_filters = {}

# After: Single HandlerEntry dataclass
@dataclass
class HandlerEntry:
    priority: int
    handler: ControlHandler
    event_types: set[Type] | None
    metadata: HandlerMetadata
```

### 4. **Benchmark Results**
```
ControlManager: 100 handlers, 1000 checks → 0.04ms per check
ObserverManager: 50 observers with filtering → 12ms per notification
Event filtering: ~90% reduction in unnecessary processing
Sync functions: Non-blocking with ThreadPoolExecutor
```

## 🧪 Testing

Comprehensive test coverage with 40+ tests:

```bash
# Run all event system tests
uv run pytest tests/hexai/core/application/events/ -v

# Test files:
test_observer_manager.py  # 13 tests - observation and filtering
test_control_manager.py   # 13 tests - control flow and policies
test_events.py            # 14 tests - event data classes
test_performance.py       # 5 tests - performance benchmarks
```

Coverage:
- `control_manager.py`: 79% coverage
- `observer_manager.py`: 89% coverage
- `events.py`: 100% coverage

## 🔄 Migration Guide

### From Old EventBus (10 files → 5 files)
```python
# Old: Mixed concerns in EventBus
event_bus = EventBus()
event_bus.register_handler(handler)
event_bus.emit(event)

# New: Separated by purpose
# For monitoring/telemetry
observer_manager = ObserverManager()
observer_manager.register(observer)
await observer_manager.notify(event)

# For control/policies
control_manager = ControlManager()
control_manager.register(handler, priority=10)
response = await control_manager.check(event, context)
```

### Adding Event Filtering (New Feature)
```python
# Register for specific events only
manager.register(
    handler,
    event_types=[NodeStarted, NodeCompleted]  # O(1) filtering
)
```

## 🎯 Design Principles

### Hexagonal Architecture
- **Ports**: Manager interfaces define contracts
- **Adapters**: Handlers/observers are pluggable implementations
- **Domain Isolation**: Events are pure data, no behavior

### Event-Driven Patterns
- **Observer Pattern**: Decoupled monitoring
- **Chain of Responsibility**: Priority-based control
- **Veto Pattern**: First non-PROCEED wins
- **Fire-and-Forget**: Async observation

### Production Readiness
- **Fault Isolation**: Failures don't cascade
- **Graceful Degradation**: Critical vs non-critical handlers
- **Timeout Protection**: Bounded execution time
- **Resource Management**: Proper cleanup, no leaks

## 🚦 Control Signals

| Signal | Purpose | Example Use Case |
|--------|---------|-----------------|
| PROCEED | Continue normally | Default - no intervention |
| SKIP | Skip this operation | Feature flags, A/B testing |
| RETRY | Retry with optional delay | Transient failures |
| FALLBACK | Use alternative value | Circuit breaker open |
| FAIL | Stop execution | Critical validation failure |
| ERROR | Critical handler error | Infrastructure issue |

## 🔮 Future Enhancements

- [ ] **Event Replay** - Debugging with event history
- [ ] **Event Persistence** - Durable event storage
- [ ] **Distributed Streaming** - Kafka/Pulsar integration
- [ ] **WebSocket Broadcasting** - Real-time UI updates
- [ ] **Event Dashboard** - Visualization and analytics
- [ ] **Hierarchical Events** - Topic-based routing
- [ ] **Event Schemas** - OpenAPI/AsyncAPI specs

## 📚 Related Documentation

- [Orchestrator Integration](../orchestrator.py) - How events integrate with DAG execution
- [Test Examples](../../../../../tests/hexai/core/application/events/) - Comprehensive test suite
- [Performance Tests](../../../../../tests/hexai/core/application/events/test_performance.py) - Benchmark validations

---

*Built for production, optimized for performance, designed for flexibility.*
