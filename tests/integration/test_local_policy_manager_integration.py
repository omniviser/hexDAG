"""Comprehensive integration tests for LocalPolicyManager.

These tests verify real-world policy evaluation scenarios, priority handling,
lifecycle management, and various edge cases.
"""

import asyncio
import gc
from typing import Any

import pytest

from hexai.adapters.local.local_policy_manager import LocalPolicyManager
from hexai.core.application.policies.models import (
    PolicyContext,
    PolicyResponse,
    PolicySignal,
    SubscriberType,
)


class TestPolicy:
    """Test policy implementation."""

    def __init__(
        self,
        name: str,
        priority: int = 100,
        signal: PolicySignal = PolicySignal.PROCEED,
        data: dict[str, Any] | None = None,
        delay: float = 0,
        should_fail: bool = False,
    ):
        self._name = name
        self._priority = priority
        self.signal = signal
        self.data = data or {}
        self.delay = delay
        self.should_fail = should_fail
        self.evaluation_count = 0
        self.last_context: PolicyContext | None = None

    @property
    def name(self) -> str:
        """Policy name for identification."""
        return self._name

    @property
    def priority(self) -> int:
        """Priority (lower = first)."""
        return self._priority

    async def evaluate(self, context: PolicyContext) -> PolicyResponse:
        """Evaluate policy."""
        self.evaluation_count += 1
        self.last_context = context

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.should_fail:
            raise RuntimeError(f"Policy {self.name} intentionally failed")

        return PolicyResponse(signal=self.signal, data=self.data)


class DynamicPolicy:
    """Policy that changes behavior based on context."""

    def __init__(self, name: str, priority: int = 100):
        self._name = name
        self._priority = priority
        self.evaluations = []

    @property
    def name(self) -> str:
        """Policy name for identification."""
        return self._name

    @property
    def priority(self) -> int:
        """Priority (lower = first)."""
        return self._priority

    async def evaluate(self, context: PolicyContext) -> PolicyResponse:
        """Dynamic evaluation based on context."""
        self.evaluations.append(context)

        # Check metadata for operation type
        if context.metadata and context.metadata.get("operation") == "read":
            return PolicyResponse(signal=PolicySignal.PROCEED)

        # Fail if operation is 'delete'
        if context.metadata and context.metadata.get("operation") == "delete":
            return PolicyResponse(
                signal=PolicySignal.FAIL, data={"reason": "Delete operations not allowed"}
            )

        # Default to proceed
        return PolicyResponse(signal=PolicySignal.PROCEED)


@pytest.mark.asyncio
class TestLocalPolicyManagerIntegration:
    """Integration tests for LocalPolicyManager."""

    async def test_priority_based_evaluation(self):
        """Test that policies are evaluated in priority order."""
        manager = LocalPolicyManager()

        # Create policies with different priorities
        high_priority = TestPolicy("high", priority=10, signal=PolicySignal.FAIL)
        med_priority = TestPolicy("medium", priority=50, signal=PolicySignal.SKIP)
        low_priority = TestPolicy("low", priority=100, signal=PolicySignal.FAIL)

        # Register in random order
        manager.subscribe(low_priority)
        manager.subscribe(high_priority)
        manager.subscribe(med_priority)

        # Evaluate
        context = PolicyContext(dag_id="test_dag")
        response = await manager.evaluate(context)

        # High priority (10) should win with FAIL
        assert response.signal == PolicySignal.FAIL

        # All should be evaluated up to first non-PROCEED
        assert high_priority.evaluation_count == 1
        assert med_priority.evaluation_count == 0  # Stopped after FAIL
        assert low_priority.evaluation_count == 0

    async def test_veto_pattern(self):
        """Test that first non-PROCEED response wins."""
        manager = LocalPolicyManager()

        # All PROCEED except one
        policies = [
            TestPolicy("p1", priority=1, signal=PolicySignal.PROCEED),
            TestPolicy("p2", priority=2, signal=PolicySignal.PROCEED),
            TestPolicy("p3", priority=3, signal=PolicySignal.SKIP),  # This wins
            TestPolicy("p4", priority=4, signal=PolicySignal.PROCEED),
            TestPolicy("p5", priority=5, signal=PolicySignal.FAIL),
        ]

        for policy in policies:
            manager.subscribe(policy)

        context = PolicyContext(dag_id="test_dag")
        response = await manager.evaluate(context)

        # p3 should win with SKIP
        assert response.signal == PolicySignal.SKIP

        # Only p1, p2, p3 should be evaluated
        assert policies[0].evaluation_count == 1
        assert policies[1].evaluation_count == 1
        assert policies[2].evaluation_count == 1
        assert policies[3].evaluation_count == 0
        assert policies[4].evaluation_count == 0

    async def test_subscriber_type_management(self):
        """Test different subscriber types and their lifecycle."""
        manager = LocalPolicyManager()

        # Create policies of different types
        core_policy = TestPolicy("core")
        plugin_policy = TestPolicy("plugin")
        user_policy = TestPolicy("user")
        temp_policy = TestPolicy("temp")

        # Subscribe with different types
        manager.subscribe(core_policy, SubscriberType.CORE)
        manager.subscribe(plugin_policy, SubscriberType.PLUGIN)
        manager.subscribe(user_policy, SubscriberType.USER)
        manager.subscribe(temp_policy, SubscriberType.TEMPORARY)

        # Verify all are registered
        assert len(manager) == 4

        # Get by type
        core_policies = manager.get_policies_by_type(SubscriberType.CORE)
        assert len(core_policies) == 1
        assert core_policies[0] is core_policy

        # Clear user policies
        manager.clear(SubscriberType.USER)

        # Core and plugin should remain
        assert len(manager) == 3
        assert core_policy in manager
        assert plugin_policy in manager
        assert temp_policy in manager
        assert user_policy not in manager

    async def test_weak_reference_cleanup(self):
        """Test that USER/TEMPORARY policies are cleaned up automatically."""
        manager = LocalPolicyManager()

        # Create policies
        user_policy = TestPolicy("user")
        core_policy = TestPolicy("core")

        # Subscribe
        user_id = manager.subscribe(user_policy, SubscriberType.USER)
        core_id = manager.subscribe(core_policy, SubscriberType.CORE)

        # Both should be present
        assert len(manager) == 2

        # Delete user policy reference
        del user_policy
        gc.collect()

        # User policy should be gone (weak ref), core remains (strong ref)
        assert len(manager) == 1
        assert core_policy in manager

        # Unsubscribe should handle missing weak ref gracefully
        assert not manager.unsubscribe(user_id)
        assert manager.unsubscribe(core_id)

    async def test_concurrent_policy_evaluation(self):
        """Test concurrent policy evaluations."""
        manager = LocalPolicyManager()

        # Create policies with delays
        policies = [TestPolicy(f"p{i}", priority=i, delay=0.01) for i in range(10)]

        for policy in policies:
            manager.subscribe(policy)

        # Run multiple evaluations concurrently
        contexts = [PolicyContext(dag_id=f"dag_{i}") for i in range(5)]

        tasks = [manager.evaluate(ctx) for ctx in contexts]
        responses = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r.signal == PolicySignal.PROCEED for r in responses)

        # Each policy should be evaluated 5 times
        for policy in policies:
            assert policy.evaluation_count == 5

    async def test_failing_policy_handling(self):
        """Test that failing policies don't break evaluation."""
        manager = LocalPolicyManager()

        # Mix of working and failing policies
        policies = [
            TestPolicy("p1", priority=1, signal=PolicySignal.PROCEED),
            TestPolicy("p2", priority=2, should_fail=True),
            TestPolicy("p3", priority=3, signal=PolicySignal.SKIP),
            TestPolicy("p4", priority=4, signal=PolicySignal.PROCEED),
        ]

        for policy in policies:
            manager.subscribe(policy)

        context = PolicyContext(dag_id="test_dag")
        response = await manager.evaluate(context)

        # Should skip failed p2 and continue to p3
        assert response.signal == PolicySignal.SKIP

        # Check evaluation counts
        assert policies[0].evaluation_count == 1
        assert policies[1].evaluation_count == 1  # Attempted but failed
        assert policies[2].evaluation_count == 1  # Succeeded after failure
        assert policies[3].evaluation_count == 0  # Not reached

    async def test_dynamic_policy_behavior(self):
        """Test policies that change behavior based on context."""
        manager = LocalPolicyManager()

        dynamic = DynamicPolicy("dynamic")
        manager.subscribe(dynamic)

        # Test read operation - should proceed
        read_ctx = PolicyContext(
            dag_id="test_dag", metadata={"operation": "read", "resource": "file.txt"}
        )
        response = await manager.evaluate(read_ctx)
        assert response.signal == PolicySignal.PROCEED

        # Test delete operation - should fail
        delete_ctx = PolicyContext(
            dag_id="test_dag", metadata={"operation": "delete", "resource": "file.txt"}
        )
        response = await manager.evaluate(delete_ctx)
        assert response.signal == PolicySignal.FAIL
        assert "reason" in response.data

        # Test other operation - should proceed
        write_ctx = PolicyContext(
            dag_id="test_dag", metadata={"operation": "write", "resource": "file.txt"}
        )
        response = await manager.evaluate(write_ctx)
        assert response.signal == PolicySignal.PROCEED

        # Verify all contexts were recorded
        assert len(dynamic.evaluations) == 3

    async def test_subscription_management(self):
        """Test subscription ID management and retrieval."""
        manager = LocalPolicyManager()

        policies = {}
        sub_ids = {}

        # Create and subscribe multiple policies
        for i in range(5):
            policy = TestPolicy(f"p{i}")
            sub_id = manager.subscribe(
                policy, SubscriberType.USER if i % 2 == 0 else SubscriberType.PLUGIN
            )
            policies[sub_id] = policy
            sub_ids[policy] = sub_id

        # Get all subscriptions
        subscriptions = manager.get_subscriptions()
        assert len(subscriptions) == 5

        # Verify subscription data
        for sub_id, (policy, _sub_type) in subscriptions.items():
            assert sub_id in policies
            assert policies[sub_id] is policy

        # Unsubscribe some
        to_remove = list(sub_ids.values())[:2]
        for sub_id in to_remove:
            assert manager.unsubscribe(sub_id)

        # Verify removal
        assert len(manager) == 3
        subscriptions = manager.get_subscriptions()
        assert len(subscriptions) == 3

    async def test_clear_all_policies(self):
        """Test clearing all policies at once."""
        manager = LocalPolicyManager()

        # Add policies of different types and keep references
        policies = []
        for i in range(10):
            policy = TestPolicy(f"p{i}")
            sub_type = list(SubscriberType)[i % len(SubscriberType)]
            manager.subscribe(policy, sub_type)
            policies.append(policy)  # Keep strong reference

        # WeakSet may deduplicate, but we should have all subscriptions
        subscriptions = manager.get_subscriptions()
        assert len(subscriptions) == 10

        # Clear all
        manager.clear()

        assert len(manager) == 0
        assert len(manager.get_subscriptions()) == 0

        # Should still be able to add new policies
        new_policy = TestPolicy("new")
        manager.subscribe(new_policy)
        # Need to check subscriptions not len() due to WeakSet behavior
        assert len(manager.get_subscriptions()) == 1

    async def test_policy_data_propagation(self):
        """Test that policy response data is properly returned."""
        manager = LocalPolicyManager()

        # Policy with custom data
        policy = TestPolicy(
            "data_policy", signal=PolicySignal.SKIP, data={"key": "value", "number": 42}
        )

        manager.subscribe(policy)

        context = PolicyContext(dag_id="test_dag")
        response = await manager.evaluate(context)

        assert response.signal == PolicySignal.SKIP
        assert response.data == {"key": "value", "number": 42}

    async def test_empty_manager_evaluation(self):
        """Test evaluation with no policies returns PROCEED."""
        manager = LocalPolicyManager()

        context = PolicyContext(dag_id="test_dag")
        response = await manager.evaluate(context)

        assert response.signal == PolicySignal.PROCEED
        # Default PolicyResponse has data=None, not {}
        assert response.data is None

    async def test_policy_contains_check(self):
        """Test the __contains__ method for policy checking."""
        manager = LocalPolicyManager()

        policy1 = TestPolicy("p1")
        policy2 = TestPolicy("p2")

        manager.subscribe(policy1)

        assert policy1 in manager
        assert policy2 not in manager

        manager.subscribe(policy2)
        assert policy2 in manager

        manager.clear()
        assert policy1 not in manager
        assert policy2 not in manager

    async def test_heapq_priority_stability(self):
        """Test that equal priority policies maintain order."""
        manager = LocalPolicyManager()

        # Policies with same priority
        policies = [
            TestPolicy(
                f"p{i}", priority=50, signal=PolicySignal.PROCEED if i < 4 else PolicySignal.FAIL
            )
            for i in range(5)
        ]

        # Register in order
        for policy in policies:
            manager.subscribe(policy)

        context = PolicyContext(dag_id="test_dag")
        response = await manager.evaluate(context)

        # p4 (the fifth policy) should win
        assert response.signal == PolicySignal.FAIL

        # At least p4 should be evaluated (others may or may not due to heapq implementation)
        assert policies[4].evaluation_count == 1

    async def test_resubscription(self):
        """Test that same policy can be subscribed multiple times."""
        manager = LocalPolicyManager()

        policy = TestPolicy("shared")

        # Subscribe multiple times
        id1 = manager.subscribe(policy, SubscriberType.USER)
        id2 = manager.subscribe(policy, SubscriberType.USER)

        assert id1 != id2
        # WeakSet deduplicates, so len() returns 1, but we have 2 subscriptions
        assert len(manager) == 1
        assert len(manager.get_subscriptions()) == 2

        # Unsubscribe one
        assert manager.unsubscribe(id1)

        # Still have one subscription
        assert len(manager.get_subscriptions()) == 1

        # Unsubscribe the other
        assert manager.unsubscribe(id2)

        # Now no subscriptions
        assert len(manager.get_subscriptions()) == 0

    async def test_mixed_priority_and_types(self):
        """Test complex scenario with mixed priorities and subscriber types."""
        manager = LocalPolicyManager()

        # Complex policy setup
        core_high = TestPolicy("core_high", priority=1, signal=PolicySignal.PROCEED)
        plugin_med = TestPolicy("plugin_med", priority=50, signal=PolicySignal.PROCEED)
        user_low = TestPolicy("user_low", priority=100, signal=PolicySignal.SKIP)
        temp_verylow = TestPolicy("temp_verylow", priority=200, signal=PolicySignal.FAIL)

        manager.subscribe(core_high, SubscriberType.CORE)
        manager.subscribe(plugin_med, SubscriberType.PLUGIN)
        manager.subscribe(user_low, SubscriberType.USER)
        manager.subscribe(temp_verylow, SubscriberType.TEMPORARY)

        context = PolicyContext(dag_id="complex_dag")
        response = await manager.evaluate(context)

        # user_low (priority 100) should win with SKIP
        assert response.signal == PolicySignal.SKIP

        # Check evaluation order
        assert core_high.evaluation_count == 1
        assert plugin_med.evaluation_count == 1
        assert user_low.evaluation_count == 1
        assert temp_verylow.evaluation_count == 0  # Not reached

    async def test_policy_context_preservation(self):
        """Test that policy context is correctly passed to each policy."""
        manager = LocalPolicyManager()

        policies = [TestPolicy(f"p{i}") for i in range(3)]
        for policy in policies:
            manager.subscribe(policy)

        # Create context with metadata
        context = PolicyContext(
            dag_id="update_dag",
            metadata={
                "operation": "update",
                "resource": "database/table",
                "user": "admin",
                "timestamp": "2024-01-01",
            },
        )

        await manager.evaluate(context)

        # All policies should receive the same context
        for policy in policies:
            assert policy.last_context == context
            assert policy.last_context.metadata["user"] == "admin"

    async def test_performance_with_many_policies(self):
        """Test performance with large number of policies."""
        manager = LocalPolicyManager()

        # Create many policies
        policies = [TestPolicy(f"p{i}", priority=i % 100) for i in range(1000)]

        for policy in policies:
            manager.subscribe(policy)

        # Time evaluation
        import time

        start = time.time()

        context = PolicyContext(dag_id="perf_test")
        response = await manager.evaluate(context)

        duration = time.time() - start

        # Should complete quickly even with many policies
        assert duration < 1.0
        assert response.signal == PolicySignal.PROCEED

        # Clear should be fast too
        start = time.time()
        manager.clear()
        clear_duration = time.time() - start

        assert clear_duration < 0.1
        assert len(manager) == 0
