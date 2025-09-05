"""Example demonstrating the simplified event system."""

import asyncio

from hexai.core.application.events import (
    ControlResponse,
    ControlSignal,
    Event,
    EventBus,
    ExecutionContext,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    ObserverManager,
)


# Example control handler that skips test nodes
async def skip_test_nodes(event: Event, context: ExecutionContext) -> ControlResponse:
    """Skip nodes with 'test' in the name."""
    if isinstance(event, NodeStarted):
        if "test" in event.name.lower():
            print(f"  Control: Skipping test node '{event.name}'")
            return ControlResponse(signal=ControlSignal.SKIP, data={"skipped": True})
    return ControlResponse()  # Default: proceed


# Example control handler for fallback on errors
async def fallback_on_error(event: Event, context: ExecutionContext) -> ControlResponse:
    """Provide fallback value for failed API nodes."""
    if isinstance(event, NodeFailed):
        if "api" in event.name.lower():
            print(f"  Control: Providing fallback for '{event.name}'")
            return ControlResponse(
                signal=ControlSignal.FALLBACK, data={"status": "offline", "data": []}
            )
    return ControlResponse()


# Example observer for logging
async def log_observer(event: Event) -> None:
    """Simple logging observer."""
    if isinstance(event, NodeStarted):
        print(f"  Observer: Node '{event.name}' started")
    elif isinstance(event, NodeCompleted):
        print(f"  Observer: Node '{event.name}' completed in {event.duration_ms}ms")
    elif isinstance(event, NodeFailed):
        print(f"  Observer: Node '{event.name}' failed with {event.error}")


# Example observer for metrics
class MetricsObserver:
    """Observer that collects metrics."""

    def __init__(self):
        self.node_count = 0
        self.total_duration = 0.0

    async def handle(self, event: Event) -> None:
        """Handle events for metrics collection."""
        if isinstance(event, NodeStarted):
            self.node_count += 1
        elif isinstance(event, NodeCompleted):
            self.total_duration += event.duration_ms


async def main():
    """Demonstrate the simplified event system."""
    print("Simplified Event System Demo\n" + "=" * 50)

    # Create EventBus for control and ObserverManager for observability
    control_bus = EventBus()
    observers = ObserverManager()

    # Register control handlers
    control_bus.register(skip_test_nodes)
    control_bus.register(fallback_on_error)

    # Register observers
    observers.register(log_observer)
    metrics = MetricsObserver()
    observers.register(metrics)

    # Create execution context
    context = ExecutionContext(dag_id="demo_pipeline")

    # Simulate normal node execution
    print("\n1. Normal Node Execution:")
    event1 = NodeStarted(name="process_data", wave_index=1, dependencies=[])
    await observers.notify(event1)
    response1 = await control_bus.check(event1, context)
    print(f"   Control decision: {response1.signal.value}")

    # Simulate node completion
    event1_complete = NodeCompleted(
        name="process_data", wave_index=1, result={"data": 123}, duration_ms=150.5
    )
    await observers.notify(event1_complete)

    # Simulate test node (will be skipped by control)
    print("\n2. Test Node (should be skipped):")
    event2 = NodeStarted(name="test_validation", wave_index=1, dependencies=[])
    await observers.notify(event2)
    response2 = await control_bus.check(event2, context)
    print(f"   Control decision: {response2.signal.value}")
    if response2.data:
        print(f"   Control data: {response2.data}")

    # Simulate API failure (will get fallback)
    print("\n3. API Node Failure (should get fallback):")
    event3 = NodeFailed(name="api_call", wave_index=1, error=Exception("Connection timeout"))
    await observers.notify(event3)
    response3 = await control_bus.check(event3, context)
    print(f"   Control decision: {response3.signal.value}")
    if response3.data:
        print(f"   Fallback data: {response3.data}")

    # Simulate regular failure (no special handling)
    print("\n4. Regular Node Failure:")
    event4 = NodeFailed(name="compute", wave_index=1, error=Exception("Out of memory"))
    await observers.notify(event4)
    response4 = await control_bus.check(event4, context)
    print(f"   Control decision: {response4.signal.value}")

    # Show metrics
    print("\n" + "=" * 50)
    print(f"Metrics: {metrics.node_count} nodes started, {metrics.total_duration}ms total duration")
    print("Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
