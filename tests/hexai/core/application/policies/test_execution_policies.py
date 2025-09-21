"""Test that execution policies are properly registered."""

import pytest

from hexai.core.application.policies.execution_policies import (
    CircuitBreakerPolicy,
    FallbackPolicy,
    RetryPolicy,
)


class TestPolicyRegistration:
    """Test policy registration with decorators."""

    def test_retry_policy_has_metadata(self):
        """RetryPolicy should have registry metadata."""
        assert hasattr(RetryPolicy, "_hexdag_type")
        assert RetryPolicy._hexdag_type == "policy"
        assert hasattr(RetryPolicy, "_hexdag_name")
        assert RetryPolicy._hexdag_name == "retry"
        assert hasattr(RetryPolicy, "_hexdag_namespace")
        assert RetryPolicy._hexdag_namespace == "user"
        assert hasattr(RetryPolicy, "_hexdag_description")
        assert "Retry failed operations" in RetryPolicy._hexdag_description

    def test_circuit_breaker_policy_has_metadata(self):
        """CircuitBreakerPolicy should have registry metadata."""
        assert hasattr(CircuitBreakerPolicy, "_hexdag_type")
        assert CircuitBreakerPolicy._hexdag_type == "policy"
        assert hasattr(CircuitBreakerPolicy, "_hexdag_name")
        assert CircuitBreakerPolicy._hexdag_name == "circuit_breaker"
        assert hasattr(CircuitBreakerPolicy, "_hexdag_namespace")
        assert CircuitBreakerPolicy._hexdag_namespace == "user"
        assert hasattr(CircuitBreakerPolicy, "_hexdag_description")
        assert "failure threshold" in CircuitBreakerPolicy._hexdag_description

    def test_fallback_policy_has_metadata(self):
        """FallbackPolicy should have registry metadata."""
        assert hasattr(FallbackPolicy, "_hexdag_type")
        assert FallbackPolicy._hexdag_type == "policy"
        assert hasattr(FallbackPolicy, "_hexdag_name")
        assert FallbackPolicy._hexdag_name == "fallback"
        assert hasattr(FallbackPolicy, "_hexdag_namespace")
        assert FallbackPolicy._hexdag_namespace == "user"
        assert hasattr(FallbackPolicy, "_hexdag_description")
        assert "fallback value" in FallbackPolicy._hexdag_description

    def test_policies_can_be_instantiated(self):
        """Policies should still be instantiable as normal classes."""
        retry = RetryPolicy(max_retries=5)
        assert retry.max_retries == 5
        assert retry.name == "retry_5"
        assert retry.priority == 10

        circuit = CircuitBreakerPolicy(failure_threshold=3)
        assert circuit.failure_threshold == 3
        assert circuit.name == "circuit_breaker_3"
        assert circuit.priority == 5

        fallback = FallbackPolicy(fallback_value="default")
        assert fallback.fallback_value == "default"
        assert fallback.name == "fallback"
        assert fallback.priority == 30

    @pytest.mark.asyncio
    async def test_policies_work_with_registry_metadata(self):
        """Policies should work normally despite registry metadata."""
        from hexai.core.application.policies.models import PolicyContext

        # Test retry policy
        retry = RetryPolicy(max_retries=2)
        context = PolicyContext(dag_id="test", attempt=1, error=Exception())
        response = await retry.evaluate(context)
        assert response.signal.value == "retry"

        # Test circuit breaker
        circuit = CircuitBreakerPolicy(failure_threshold=1)
        context = PolicyContext(dag_id="test", error=Exception())
        response = await circuit.evaluate(context)
        assert response.signal.value == "fail"  # Opens after 1 failure

        # Test fallback
        fallback = FallbackPolicy(fallback_value="backup")
        context = PolicyContext(dag_id="test", error=Exception())
        response = await fallback.evaluate(context)
        assert response.signal.value == "fallback"
        assert response.data == "backup"
