"""Tests for LocalPolicyManager."""

import gc

import pytest

from hexai.adapters.local.local_policy_manager import LocalPolicyManager
from hexai.core.application.policies.execution_policies import (
    CircuitBreakerPolicy,
    FallbackPolicy,
    RetryPolicy,
)
from hexai.core.ports.policy_manager import (
    PolicyContext,
    PolicySignal,
    SubscriberType,
)


class TestLocalPolicyManager:
    """Tests for LocalPolicyManager."""

    @pytest.mark.asyncio
    async def test_empty_manager(self):
        """Empty manager returns PROCEED."""
        adapter = LocalPolicyManager()
        context = PolicyContext(dag_id="test")
        response = await adapter.evaluate(context)
        assert response.signal == PolicySignal.PROCEED

    @pytest.mark.asyncio
    async def test_basic_subscription(self):
        """Test basic subscription and evaluation."""
        adapter = LocalPolicyManager()

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
        adapter = LocalPolicyManager()

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
    async def test_strong_references_for_core_plugins(self):
        """Test that CORE and PLUGIN policies are kept alive."""
        adapter = LocalPolicyManager()

        # Subscribe CORE policy
        core_policy = RetryPolicy()
        adapter.subscribe(core_policy, SubscriberType.CORE)

        # Subscribe PLUGIN policy
        plugin_policy = CircuitBreakerPolicy()
        adapter.subscribe(plugin_policy, SubscriberType.PLUGIN)

        # Subscribe USER policy
        user_policy = FallbackPolicy("default")
        adapter.subscribe(user_policy, SubscriberType.USER)

        # Delete local references
        del core_policy
        del plugin_policy
        del user_policy

        # Force garbage collection
        gc.collect()

        # CORE and PLUGIN should still be alive
        assert len(adapter) >= 2  # At least CORE and PLUGIN

        # Check by type
        core_policies = adapter.get_policies_by_type(SubscriberType.CORE)
        plugin_policies = adapter.get_policies_by_type(SubscriberType.PLUGIN)

        assert len(core_policies) == 1
        assert len(plugin_policies) == 1

    @pytest.mark.asyncio
    async def test_user_policies_are_weak_referenced(self):
        """Test that USER and TEMPORARY policies can be garbage collected."""
        adapter = LocalPolicyManager()

        # Subscribe USER policy
        user_policy = RetryPolicy()
        adapter.subscribe(user_policy, SubscriberType.USER)

        # Subscribe TEMPORARY policy
        temp_policy = FallbackPolicy("temp")
        adapter.subscribe(temp_policy, SubscriberType.TEMPORARY)

        assert len(adapter) == 2

        # Delete local references
        del user_policy
        del temp_policy
        gc.collect()

        # Policies should be gone
        assert len(adapter) == 0

    @pytest.mark.asyncio
    async def test_clear_by_type(self):
        """Test clearing policies by subscriber type."""
        adapter = LocalPolicyManager()

        # Add different types - keep references to prevent GC
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

        # USER policies should be gone
        user_policies = adapter.get_policies_by_type(SubscriberType.USER)
        assert len(user_policies) == 0

        # CORE and PLUGIN should remain
        core_policies = adapter.get_policies_by_type(SubscriberType.CORE)
        plugin_policies = adapter.get_policies_by_type(SubscriberType.PLUGIN)
        assert len(core_policies) == 1
        assert len(plugin_policies) == 1

        # Clear all
        adapter.clear()
        assert len(adapter) == 0

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Test unsubscription."""
        adapter = LocalPolicyManager()

        policy = RetryPolicy()
        sub_id = adapter.subscribe(policy, SubscriberType.CORE)

        assert len(adapter) == 1

        # Unsubscribe
        result = adapter.unsubscribe(sub_id)
        assert result is True
        assert len(adapter) == 0

        # Second unsubscribe should fail
        result = adapter.unsubscribe(sub_id)
        assert result is False

    @pytest.mark.asyncio
    async def test_weak_reference_cleanup(self):
        """Policies are cleaned up when garbage collected."""
        adapter = LocalPolicyManager()

        # Subscribe a policy
        retry = RetryPolicy()
        adapter.subscribe(retry, SubscriberType.USER)
        assert len(adapter) == 1

        # Delete the policy and force garbage collection
        del retry
        gc.collect()

        # Policy should be gone due to weak reference
        assert len(adapter) == 0

    @pytest.mark.asyncio
    async def test_get_subscriptions(self):
        """Test getting all subscriptions."""
        adapter = LocalPolicyManager()

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
        adapter = LocalPolicyManager()

        class FailingPolicy:
            name = "failing"
            priority = 1

            async def evaluate(self, context):
                raise ValueError("Policy error")

        class WorkingPolicy:
            name = "working"
            priority = 2

            async def evaluate(self, context):
                from hexai.core.ports.policy_manager import PolicyResponse, PolicySignal

                if context.error:
                    return PolicyResponse(signal=PolicySignal.RETRY)
                return PolicyResponse(signal=PolicySignal.PROCEED)

        # Keep references to prevent GC
        failing = FailingPolicy()
        working = WorkingPolicy()

        adapter.subscribe(failing, SubscriberType.USER)
        adapter.subscribe(working, SubscriberType.USER)

        context = PolicyContext(dag_id="test", error=Exception())
        response = await adapter.evaluate(context)

        # Should skip failing and use working
        assert response.signal == PolicySignal.RETRY
