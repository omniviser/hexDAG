"""Example demonstrating the simplified event system."""

import asyncio

from hexai.adapters.local import LocalObserverManager, LocalPolicyManager
from hexai.core.application.events import (
    Event,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
)
from hexai.core.application.policies.models import (
    PolicyContext,
    PolicyResponse,
    PolicySignal,
)


# Example policy that skips test nodes
async def skip_test_nodes(context: PolicyContext) -> PolicyResponse:
    """Skip nodes with 'test' in the name."""
    if isinstance(context.event, NodeStarted) and "test" in context.event.name.lower():
        print(f"  Policy: Skipping test node '{context.event.name}'")
        return PolicyResponse(signal=PolicySignal.SKIP, data={"skipped": True})
    return PolicyResponse(signal=PolicySignal.PROCEED)  # Default: proceed


# Example policy for fallback on errors
async def fallback_on_error(context: PolicyContext) -> PolicyResponse:
    """Provide fallback value for failed API nodes."""
    if isinstance(context.event, NodeFailed) and "api" in context.event.name.lower():
        print(f"  Policy: Providing fallback for '{context.event.name}'")
        return PolicyResponse(signal=PolicySignal.FALLBACK, data={"status": "offline", "data": []})
    return PolicyResponse(signal=PolicySignal.PROCEED)


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

    # Create PolicyManager for execution control and ObserverManager for observability
    policy_manager = LocalPolicyManager()
    observer_manager = LocalObserverManager()

    # Register policies
    # Create simple policy objects
    from hexai.core.application.policies.models import SubscriberType

    class SkipTestPolicy:
        priority = 0

        async def evaluate(self, context):
            return await skip_test_nodes(context)

    class FallbackPolicy:
        priority = 1

        async def evaluate(self, context):
            return await fallback_on_error(context)

    # Keep strong references to policies
    skip_policy = SkipTestPolicy()
    fallback_policy = FallbackPolicy()

    # Subscribe as CORE to ensure they stay in memory
    policy_manager.subscribe(skip_policy, SubscriberType.CORE)
    policy_manager.subscribe(fallback_policy, SubscriberType.CORE)

    # Register observers
    observer_manager.register(log_observer)
    metrics = MetricsObserver()
    observer_manager.register(metrics.handle)

    # Simulate normal node execution
    print("\n1. Normal Node Execution:")
    event1 = NodeStarted(name="process_data", wave_index=1, dependencies=[])
    await observer_manager.notify(event1)
    policy_context1 = PolicyContext(
        event=event1, dag_id="demo_pipeline", node_id="process_data", wave_index=1, attempt=1
    )
    response1 = await policy_manager.evaluate(policy_context1)
    print(f"   Policy decision: {response1.signal.value}")

    # Simulate node completion
    event1_complete = NodeCompleted(
        name="process_data", wave_index=1, result={"data": 123}, duration_ms=150.5
    )
    await observer_manager.notify(event1_complete)

    # Simulate test node (will be skipped by control)
    print("\n2. Test Node (should be skipped):")
    event2 = NodeStarted(name="test_validation", wave_index=1, dependencies=[])
    await observer_manager.notify(event2)
    policy_context2 = PolicyContext(
        event=event2, dag_id="demo_pipeline", node_id="test_validation", wave_index=1, attempt=1
    )
    response2 = await policy_manager.evaluate(policy_context2)
    print(f"   Policy decision: {response2.signal.value}")
    if response2.data:
        print(f"   Control data: {response2.data}")

    # Simulate API failure (will get fallback)
    print("\n3. API Node Failure (should get fallback):")
    event3 = NodeFailed(name="api_call", wave_index=1, error=Exception("Connection timeout"))
    await observer_manager.notify(event3)
    policy_context3 = PolicyContext(
        event=event3, dag_id="demo_pipeline", node_id="api_call", wave_index=1, attempt=1
    )
    response3 = await policy_manager.evaluate(policy_context3)
    print(f"   Policy decision: {response3.signal.value}")
    if response3.data:
        print(f"   Fallback data: {response3.data}")

    # Simulate regular failure (no special handling)
    print("\n4. Regular Node Failure:")
    event4 = NodeFailed(name="compute", wave_index=1, error=Exception("Out of memory"))
    await observer_manager.notify(event4)
    policy_context4 = PolicyContext(
        event=event4, dag_id="demo_pipeline", node_id="compute", wave_index=1, attempt=1
    )
    response4 = await policy_manager.evaluate(policy_context4)
    print(f"   Policy decision: {response4.signal.value}")

    # Show metrics
    print("\n" + "=" * 50)
    print(f"Metrics: {metrics.node_count} nodes started, {metrics.total_duration}ms total duration")
    print("Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
