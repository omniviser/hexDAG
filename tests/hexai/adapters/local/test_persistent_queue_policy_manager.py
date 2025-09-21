"""Tests for PersistentQueuePolicyManager."""

import pytest

from hexai.adapters.local.persistent_queue_policy_manager import PersistentQueuePolicyManager
from hexai.core.application.policies.execution_policies import (
    CircuitBreakerPolicy,
    FallbackPolicy,
    RetryPolicy,
)
from hexai.core.application.policies.models import (
    PolicyContext,
    PolicySignal,
    SubscriberType,
)


class TestPersistentQueuePolicyManager:
    """Tests for PersistentQueuePolicyManager."""

    @pytest.mark.asyncio
    async def test_empty_manager(self):
        """Empty manager returns PROCEED."""
        adapter = PersistentQueuePolicyManager()
        context = PolicyContext(dag_id="test")
        response = await adapter.evaluate(context)
        assert response.signal == PolicySignal.PROCEED
        assert len(adapter) == 0

    @pytest.mark.asyncio
    async def test_basic_subscription(self):
        """Test basic subscription and evaluation."""
        adapter = PersistentQueuePolicyManager()

        retry = RetryPolicy(max_retries=3)
        sub_id = adapter.subscribe(retry, SubscriberType.USER)

        assert sub_id is not None
        assert len(adapter) == 1
        assert retry in adapter

        # Test evaluation with error
        context = PolicyContext(dag_id="test", attempt=1, error=Exception())
        response = await adapter.evaluate(context)
        assert response.signal == PolicySignal.RETRY

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Test priority-based execution order."""
        adapter = PersistentQueuePolicyManager()

        # Add policies with different priorities
        circuit = CircuitBreakerPolicy(failure_threshold=1)  # priority 5
        retry = RetryPolicy(max_retries=3)  # priority 10
        fallback = FallbackPolicy("default")  # priority 30

        adapter.subscribe(fallback, SubscriberType.USER)
        adapter.subscribe(retry, SubscriberType.USER)
        adapter.subscribe(circuit, SubscriberType.CORE)

        # Circuit breaker should execute first and fail
        context = PolicyContext(dag_id="test", error=Exception())
        response = await adapter.evaluate(context)
        assert response.signal == PolicySignal.FAIL

    @pytest.mark.asyncio
    async def test_persistent_queue_efficiency(self):
        """Test that queue persists between evaluations."""
        adapter = PersistentQueuePolicyManager()

        # Add policies
        retry = RetryPolicy(max_retries=2)
        fallback = FallbackPolicy("backup")

        adapter.subscribe(retry, SubscriberType.USER)
        adapter.subscribe(fallback, SubscriberType.USER)

        # First evaluation
        context1 = PolicyContext(dag_id="test", attempt=1, error=Exception())
        response1 = await adapter.evaluate(context1)
        assert response1.signal == PolicySignal.RETRY

        # Second evaluation - queue should be reused
        context2 = PolicyContext(dag_id="test", attempt=3, error=Exception())
        response2 = await adapter.evaluate(context2)
        assert response2.signal == PolicySignal.FALLBACK

        # Verify queue wasn't marked dirty
        assert not adapter._dirty

    @pytest.mark.asyncio
    async def test_queue_rebuilds_after_unsubscribe(self):
        """Test that queue rebuilds after unsubscribe."""
        adapter = PersistentQueuePolicyManager()

        retry = RetryPolicy()
        circuit = CircuitBreakerPolicy()

        sub1 = adapter.subscribe(retry, SubscriberType.USER)
        adapter.subscribe(circuit, SubscriberType.CORE)

        # Unsubscribe one
        adapter.unsubscribe(sub1)

        # Queue should be marked dirty
        assert adapter._dirty

        # Evaluation should trigger rebuild
        context = PolicyContext(dag_id="test", error=Exception())
        await adapter.evaluate(context)

        # Queue should be clean after rebuild
        assert not adapter._dirty
        assert len(adapter) == 1

    @pytest.mark.asyncio
    async def test_clear_by_type(self):
        """Test clearing policies by subscriber type."""
        adapter = PersistentQueuePolicyManager()

        # Add different types
        retry = RetryPolicy()
        circuit = CircuitBreakerPolicy()
        fallback = FallbackPolicy("default")

        adapter.subscribe(retry, SubscriberType.CORE)
        adapter.subscribe(circuit, SubscriberType.PLUGIN)
        adapter.subscribe(fallback, SubscriberType.USER)

        assert len(adapter) == 3

        # Clear only USER policies
        adapter.clear(SubscriberType.USER)
        assert len(adapter) == 2
        assert adapter._dirty

        # USER policies should be gone
        user_policies = adapter.get_policies_by_type(SubscriberType.USER)
        assert len(user_policies) == 0

        # CORE and PLUGIN should remain
        core_policies = adapter.get_policies_by_type(SubscriberType.CORE)
        plugin_policies = adapter.get_policies_by_type(SubscriberType.PLUGIN)
        assert len(core_policies) == 1
        assert len(plugin_policies) == 1

    @pytest.mark.asyncio
    async def test_clear_all(self):
        """Test clearing all policies."""
        adapter = PersistentQueuePolicyManager()

        # Add policies
        retry = RetryPolicy()
        circuit = CircuitBreakerPolicy()

        adapter.subscribe(retry, SubscriberType.CORE)
        adapter.subscribe(circuit, SubscriberType.PLUGIN)

        assert len(adapter) == 2

        # Clear all
        adapter.clear()
        assert len(adapter) == 0
        assert not adapter._dirty  # Clear all resets dirty flag
        assert len(adapter._queue) == 0

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Test unsubscription."""
        adapter = PersistentQueuePolicyManager()

        policy = RetryPolicy()
        sub_id = adapter.subscribe(policy, SubscriberType.CORE)

        assert len(adapter) == 1

        # Unsubscribe
        result = adapter.unsubscribe(sub_id)
        assert result is True
        assert len(adapter) == 0
        assert adapter._dirty

        # Second unsubscribe should fail
        result = adapter.unsubscribe(sub_id)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_subscriptions(self):
        """Test getting all subscriptions."""
        adapter = PersistentQueuePolicyManager()

        retry = RetryPolicy()
        circuit = CircuitBreakerPolicy()

        sub1 = adapter.subscribe(retry, SubscriberType.CORE)
        sub2 = adapter.subscribe(circuit, SubscriberType.PLUGIN)

        subscriptions = adapter.get_subscriptions()
        assert len(subscriptions) == 2

        policy1, type1 = subscriptions[sub1]
        assert policy1.name == retry.name
        assert type1 == SubscriberType.CORE

        policy2, type2 = subscriptions[sub2]
        assert policy2.name == circuit.name
        assert type2 == SubscriberType.PLUGIN

    @pytest.mark.asyncio
    async def test_failed_policy_skipped(self):
        """Test that failed policies are skipped."""
        adapter = PersistentQueuePolicyManager()

        class FailingPolicy:
            name = "failing"
            priority = 1

            async def evaluate(self, context):
                raise ValueError("Policy error")

        class WorkingPolicy:
            name = "working"
            priority = 2

            async def evaluate(self, context):
                from hexai.core.application.policies.models import PolicyResponse, PolicySignal

                if context.error:
                    return PolicyResponse(signal=PolicySignal.RETRY)
                return PolicyResponse(signal=PolicySignal.PROCEED)

        failing = FailingPolicy()
        working = WorkingPolicy()

        adapter.subscribe(failing, SubscriberType.USER)
        adapter.subscribe(working, SubscriberType.USER)

        context = PolicyContext(dag_id="test", error=Exception())
        response = await adapter.evaluate(context)

        # Should skip failing and use working
        assert response.signal == PolicySignal.RETRY

    @pytest.mark.asyncio
    async def test_stable_sorting(self):
        """Test that policies with same priority are stable sorted."""
        adapter = PersistentQueuePolicyManager()

        class TestPolicy:
            def __init__(self, name: str, priority: int = 10):
                self.name = name
                self.priority = priority
                self.evaluated = False

            async def evaluate(self, context):
                self.evaluated = True
                from hexai.core.application.policies.models import PolicyResponse, PolicySignal

                return PolicyResponse(signal=PolicySignal.PROCEED)

        # Add policies with same priority
        policy1 = TestPolicy("first", 10)
        policy2 = TestPolicy("second", 10)
        policy3 = TestPolicy("third", 10)

        adapter.subscribe(policy1)
        adapter.subscribe(policy2)
        adapter.subscribe(policy3)

        # Should maintain insertion order for same priority
        context = PolicyContext(dag_id="test")
        await adapter.evaluate(context)

        # All should be evaluated
        assert policy1.evaluated
        assert policy2.evaluated
        assert policy3.evaluated

    @pytest.mark.asyncio
    async def test_no_weak_references(self):
        """Test that policies are not garbage collected (strong references)."""
        import gc

        adapter = PersistentQueuePolicyManager()

        # Create and subscribe policy
        sub_id = adapter.subscribe(RetryPolicy(), SubscriberType.USER)

        assert len(adapter) == 1

        # Force garbage collection
        gc.collect()

        # Policy should still be there (strong reference)
        assert len(adapter) == 1
        assert sub_id in adapter.get_subscriptions()

    @pytest.mark.asyncio
    async def test_contains_operator(self):
        """Test the __contains__ operator."""
        adapter = PersistentQueuePolicyManager()

        retry = RetryPolicy()
        circuit = CircuitBreakerPolicy()

        adapter.subscribe(retry, SubscriberType.USER)

        assert retry in adapter
        assert circuit not in adapter

        adapter.subscribe(circuit, SubscriberType.USER)
        assert circuit in adapter
