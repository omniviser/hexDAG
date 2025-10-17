"""Integration tests for the event system (observers and policies)."""

import pytest

from hexdag.builtin.adapters.local import LocalObserverManager, LocalPolicyManager
from hexdag.core.orchestration.events import (
    Event,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
)
from hexdag.core.orchestration.policies.models import (
    PolicyContext,
    PolicyResponse,
    PolicySignal,
    SubscriberType,
)


class TestEventSystemIntegration:
    """Test event system integration with observers and policies."""

    @pytest.fixture
    def policy_manager(self):
        """Create a policy manager."""
        return LocalPolicyManager()

    @pytest.fixture
    def observer_manager(self):
        """Create an observer manager."""
        return LocalObserverManager()

    @pytest.mark.asyncio
    async def test_observer_notification(self, observer_manager):
        """Test that observers receive and handle events."""
        events_received = []

        async def test_observer(event: Event) -> None:
            events_received.append(event)

        observer_manager.register(test_observer)

        # Emit events
        event1 = NodeStarted(name="node1", wave_index=1, dependencies=[])
        event2 = NodeCompleted(name="node1", wave_index=1, result={"data": 123}, duration_ms=150.5)

        await observer_manager.notify(event1)
        await observer_manager.notify(event2)

        assert len(events_received) == 2
        assert isinstance(events_received[0], NodeStarted)
        assert isinstance(events_received[1], NodeCompleted)
        assert events_received[0].name == "node1"
        assert events_received[1].result == {"data": 123}

    @pytest.mark.asyncio
    async def test_observer_class_based(self, observer_manager):
        """Test class-based observer with state."""

        class MetricsObserver:
            def __init__(self):
                self.node_count = 0
                self.total_duration = 0.0

            async def handle(self, event: Event) -> None:
                if isinstance(event, NodeStarted):
                    self.node_count += 1
                elif isinstance(event, NodeCompleted):
                    self.total_duration += event.duration_ms

        metrics = MetricsObserver()
        observer_manager.register(metrics.handle)

        # Emit events
        await observer_manager.notify(NodeStarted(name="node1", wave_index=1, dependencies=[]))
        await observer_manager.notify(
            NodeCompleted(name="node1", wave_index=1, result={}, duration_ms=100.0)
        )
        await observer_manager.notify(NodeStarted(name="node2", wave_index=1, dependencies=[]))
        await observer_manager.notify(
            NodeCompleted(name="node2", wave_index=1, result={}, duration_ms=200.0)
        )

        assert metrics.node_count == 2
        assert metrics.total_duration == 300.0

    @pytest.mark.asyncio
    async def test_policy_skip_signal(self, policy_manager):
        """Test policy that returns SKIP signal."""

        class SkipTestPolicy:
            priority = 0

            async def evaluate(self, context: PolicyContext) -> PolicyResponse:
                if isinstance(context.event, NodeStarted) and "test" in context.event.name.lower():
                    return PolicyResponse(signal=PolicySignal.SKIP, data={"skipped": True})
                return PolicyResponse(signal=PolicySignal.PROCEED)

        policy = SkipTestPolicy()
        policy_manager.subscribe(policy, SubscriberType.CORE)

        # Test node should be skipped
        event = NodeStarted(name="test_validation", wave_index=1, dependencies=[])
        context = PolicyContext(
            event=event, dag_id="test_dag", node_id="test_validation", wave_index=1, attempt=1
        )
        response = await policy_manager.evaluate(context)

        assert response.signal == PolicySignal.SKIP
        assert response.data == {"skipped": True}

        # Regular node should proceed
        event2 = NodeStarted(name="process_data", wave_index=1, dependencies=[])
        context2 = PolicyContext(
            event=event2, dag_id="test_dag", node_id="process_data", wave_index=1, attempt=1
        )
        response2 = await policy_manager.evaluate(context2)

        assert response2.signal == PolicySignal.PROCEED

    @pytest.mark.asyncio
    async def test_policy_fallback_signal(self, policy_manager):
        """Test policy that returns FALLBACK signal on failures."""

        class FallbackPolicy:
            priority = 1

            async def evaluate(self, context: PolicyContext) -> PolicyResponse:
                if isinstance(context.event, NodeFailed) and "api" in context.event.name.lower():
                    return PolicyResponse(
                        signal=PolicySignal.FALLBACK, data={"status": "offline", "data": []}
                    )
                return PolicyResponse(signal=PolicySignal.PROCEED)

        policy = FallbackPolicy()
        policy_manager.subscribe(policy, SubscriberType.CORE)

        # API failure should get fallback
        event = NodeFailed(name="api_call", wave_index=1, error=Exception("Connection timeout"))
        context = PolicyContext(
            event=event, dag_id="test_dag", node_id="api_call", wave_index=1, attempt=1
        )
        response = await policy_manager.evaluate(context)

        assert response.signal == PolicySignal.FALLBACK
        assert response.data == {"status": "offline", "data": []}

        # Regular failure should proceed (propagate)
        event2 = NodeFailed(name="compute", wave_index=1, error=Exception("Out of memory"))
        context2 = PolicyContext(
            event=event2, dag_id="test_dag", node_id="compute", wave_index=1, attempt=1
        )
        response2 = await policy_manager.evaluate(context2)

        assert response2.signal == PolicySignal.PROCEED

    @pytest.mark.asyncio
    async def test_multiple_policies_priority(self, policy_manager):
        """Test that policies are evaluated in priority order."""
        evaluation_order = []

        class HighPriorityPolicy:
            priority = 0

            async def evaluate(self, context: PolicyContext) -> PolicyResponse:
                evaluation_order.append("high")
                return PolicyResponse(signal=PolicySignal.PROCEED)

        class LowPriorityPolicy:
            priority = 10

            async def evaluate(self, context: PolicyContext) -> PolicyResponse:
                evaluation_order.append("low")
                return PolicyResponse(signal=PolicySignal.PROCEED)

        high_policy = HighPriorityPolicy()
        low_policy = LowPriorityPolicy()

        policy_manager.subscribe(low_policy, SubscriberType.CORE)
        policy_manager.subscribe(high_policy, SubscriberType.CORE)

        event = NodeStarted(name="test", wave_index=1, dependencies=[])
        context = PolicyContext(
            event=event, dag_id="test_dag", node_id="test", wave_index=1, attempt=1
        )

        await policy_manager.evaluate(context)

        # High priority (lower number) should execute first
        assert evaluation_order == ["high", "low"]

    @pytest.mark.asyncio
    async def test_policy_short_circuit(self, policy_manager):
        """Test that non-PROCEED response stops policy evaluation chain."""
        evaluation_order = []

        class FirstPolicy:
            priority = 0

            async def evaluate(self, context: PolicyContext) -> PolicyResponse:
                evaluation_order.append("first")
                return PolicyResponse(signal=PolicySignal.SKIP)

        class SecondPolicy:
            priority = 1

            async def evaluate(self, context: PolicyContext) -> PolicyResponse:
                evaluation_order.append("second")
                return PolicyResponse(signal=PolicySignal.PROCEED)

        first_policy = FirstPolicy()
        second_policy = SecondPolicy()

        policy_manager.subscribe(first_policy, SubscriberType.CORE)
        policy_manager.subscribe(second_policy, SubscriberType.CORE)

        event = NodeStarted(name="test", wave_index=1, dependencies=[])
        context = PolicyContext(
            event=event, dag_id="test_dag", node_id="test", wave_index=1, attempt=1
        )

        response = await policy_manager.evaluate(context)

        # First policy returns SKIP, so second policy should not execute
        assert evaluation_order == ["first"]
        assert response.signal == PolicySignal.SKIP

    @pytest.mark.asyncio
    async def test_integrated_observers_and_policies(self, observer_manager, policy_manager):
        """Test observers and policies working together."""
        events_observed = []

        async def tracking_observer(event: Event) -> None:
            events_observed.append(event)

        class SkipPolicy:
            priority = 0

            async def evaluate(self, context: PolicyContext) -> PolicyResponse:
                if isinstance(context.event, NodeStarted) and context.event.name == "skip_me":
                    return PolicyResponse(signal=PolicySignal.SKIP)
                return PolicyResponse(signal=PolicySignal.PROCEED)

        observer_manager.register(tracking_observer)
        policy = SkipPolicy()
        policy_manager.subscribe(policy, SubscriberType.CORE)

        # Normal node
        event1 = NodeStarted(name="normal", wave_index=1, dependencies=[])
        await observer_manager.notify(event1)
        context1 = PolicyContext(
            event=event1, dag_id="test_dag", node_id="normal", wave_index=1, attempt=1
        )
        response1 = await policy_manager.evaluate(context1)

        assert response1.signal == PolicySignal.PROCEED
        assert len(events_observed) == 1

        # Skipped node
        event2 = NodeStarted(name="skip_me", wave_index=1, dependencies=[])
        await observer_manager.notify(event2)
        context2 = PolicyContext(
            event=event2, dag_id="test_dag", node_id="skip_me", wave_index=1, attempt=1
        )
        response2 = await policy_manager.evaluate(context2)

        assert response2.signal == PolicySignal.SKIP
        assert len(events_observed) == 2  # Observer still notified
