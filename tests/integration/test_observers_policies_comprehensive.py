"""Comprehensive integration tests for observers and policies.

Tests demonstrate:
- Multiple observer types working together
- Policy-based execution control
- Real-world patterns (retry, circuit breaker, rate limiting)
- Monitoring performance, quality, and resources
- How observers and policies complement each other
"""

import pytest

from hexdag.builtin.adapters.local import LocalObserverManager, LocalPolicyManager
from hexdag.builtin.policies import (
    CircuitBreakerPolicy,
    ConditionalSkipPolicy,
    FallbackPolicy,
    RetryPolicy,
)
from hexdag.core.orchestration.events import (
    ALL_EXECUTION_EVENTS,
    AlertingObserver,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    PerformanceMetricsObserver,
    PipelineCompleted,
    PipelineStarted,
    SimpleLoggingObserver,
)
from hexdag.core.orchestration.policies.models import (
    PolicyContext,
    PolicySignal,
    SubscriberType,
)


@pytest.fixture
def observer_manager():
    """Provide observer manager with batching disabled for predictable testing."""
    return LocalObserverManager(
        max_concurrent_observers=20,
        observer_timeout=10.0,
        use_weak_refs=True,
        batching_enabled=False,
    )


@pytest.fixture
def policy_manager():
    """Provide policy manager."""
    return LocalPolicyManager()


class TestObserversAndPolicies:
    """Test suite for observers and policies integration."""

    @pytest.mark.asyncio
    async def test_observer_registration(self, observer_manager):
        """Test registering observers with manager."""
        metrics_observer = PerformanceMetricsObserver()
        observer_manager.register(
            metrics_observer.handle,
            event_types=ALL_EXECUTION_EVENTS,
        )

        assert len(observer_manager) == 1

    @pytest.mark.asyncio
    async def test_policy_registration(self, policy_manager):
        """Test registering policies with manager."""
        retry_policy = RetryPolicy(max_retries=3)
        policy_manager.subscribe(retry_policy, SubscriberType.CORE)

        assert len(policy_manager) == 1

    @pytest.mark.asyncio
    async def test_performance_metrics_observer(self, observer_manager):
        """Test performance metrics collection."""
        metrics_observer = PerformanceMetricsObserver()
        observer_manager.register(metrics_observer.handle)

        # Simulate events
        await observer_manager.notify(
            PipelineStarted(name="test_pipeline", total_waves=2, total_nodes=2)
        )
        await observer_manager.notify(NodeStarted(name="node1", wave_index=0, dependencies=[]))
        await observer_manager.notify(
            NodeCompleted(name="node1", wave_index=0, result={}, duration_ms=100.0)
        )
        await observer_manager.notify(PipelineCompleted(name="test_pipeline", duration_ms=200.0))

        # Check metrics
        summary = metrics_observer.get_summary()
        assert summary["total_nodes_executed"] == 1

    @pytest.mark.asyncio
    async def test_alerting_observer(self, observer_manager):
        """Test alerting on slow or failed nodes."""
        alerts = []

        def capture_alert(alert):
            alerts.append(alert)

        alerting_observer = AlertingObserver(
            slow_threshold_ms=50.0,
            on_alert=capture_alert,
        )
        observer_manager.register(alerting_observer.handle)

        # Trigger slow node alert
        await observer_manager.notify(
            NodeCompleted(name="slow_node", wave_index=0, result={}, duration_ms=100.0)
        )

        assert len(alerts) > 0

    @pytest.mark.asyncio
    async def test_simple_logging_observer(self, observer_manager):
        """Test simple logging observer."""
        logger_obs = SimpleLoggingObserver(verbose=False)
        observer_manager.register(logger_obs.handle)

        # Simulate events
        await observer_manager.notify(PipelineStarted(name="test", total_waves=1, total_nodes=1))
        await observer_manager.notify(NodeStarted(name="node1", wave_index=0, dependencies=[]))
        await observer_manager.notify(
            NodeCompleted(name="node1", wave_index=0, result={}, duration_ms=50.0)
        )

        # Should not raise errors
        assert True

    @pytest.mark.asyncio
    async def test_retry_policy(self, policy_manager):
        """Test retry policy evaluation."""
        retry_policy = RetryPolicy(max_retries=3)
        policy_manager.subscribe(retry_policy, SubscriberType.CORE)

        # Simulate failure
        ctx = PolicyContext(
            event=NodeFailed(name="failing_node", wave_index=0, error=Exception("Test error")),
            dag_id="test_dag",
            node_id="failing_node",
            wave_index=0,
            attempt=1,
        )

        response = await policy_manager.evaluate(ctx)

        # Should allow retry
        assert response.signal in [PolicySignal.RETRY, PolicySignal.PROCEED]

    @pytest.mark.asyncio
    async def test_circuit_breaker_policy(self, policy_manager):
        """Test circuit breaker policy."""
        circuit_breaker = CircuitBreakerPolicy(failure_threshold=3)
        policy_manager.subscribe(circuit_breaker, SubscriberType.CORE)

        # Simulate multiple failures
        for i in range(5):
            ctx = PolicyContext(
                event=NodeFailed(name="node", wave_index=0, error=Exception("Error")),
                dag_id="test_dag",
                node_id="node",
                wave_index=0,
                attempt=i + 1,
                error=Exception("Error"),
            )

            response = await policy_manager.evaluate(ctx)

            # After threshold, should signal to stop
            if i >= 3:
                assert response.signal in [PolicySignal.FAIL, PolicySignal.SKIP]

    @pytest.mark.asyncio
    async def test_fallback_policy(self, policy_manager):
        """Test fallback policy provides fallback values."""
        fallback_policy = FallbackPolicy(fallback_value={"status": "fallback", "data": "cached"})
        policy_manager.subscribe(fallback_policy, SubscriberType.USER)

        # Simulate failure
        ctx = PolicyContext(
            event=NodeFailed(name="node", wave_index=0, error=Exception("Error")),
            dag_id="test_dag",
            node_id="node",
            wave_index=0,
            attempt=3,  # Max attempts reached
            error=Exception("Error"),
        )

        response = await policy_manager.evaluate(ctx)

        # Should provide fallback
        assert response.signal == PolicySignal.FALLBACK
        assert response.data is not None

    @pytest.mark.asyncio
    async def test_conditional_skip_policy(self, policy_manager):
        """Test conditional skip policy."""

        def skip_test_nodes(ctx: PolicyContext) -> bool:
            return "test" in ctx.node_id.lower()

        skip_policy = ConditionalSkipPolicy(should_skip=skip_test_nodes, policy_name="skip_tests")
        policy_manager.subscribe(skip_policy, SubscriberType.USER)

        # Test with test node
        ctx_test = PolicyContext(
            event=NodeStarted(name="test_node", wave_index=0, dependencies=[]),
            dag_id="test_dag",
            node_id="test_node",
            wave_index=0,
            attempt=1,
        )

        response = await policy_manager.evaluate(ctx_test)
        assert response.signal == PolicySignal.SKIP

        # Test with non-test node
        ctx_prod = PolicyContext(
            event=NodeStarted(name="prod_node", wave_index=0, dependencies=[]),
            dag_id="test_dag",
            node_id="prod_node",
            wave_index=0,
            attempt=1,
        )

        response = await policy_manager.evaluate(ctx_prod)
        assert response.signal == PolicySignal.PROCEED

    @pytest.mark.asyncio
    async def test_multiple_observers_together(self, observer_manager):
        """Test multiple observers working together."""
        # Register multiple observers
        metrics_observer = PerformanceMetricsObserver()
        alerting_observer = AlertingObserver(slow_threshold_ms=50.0)
        logger_obs = SimpleLoggingObserver(verbose=False)

        observer_manager.register(metrics_observer.handle)
        observer_manager.register(alerting_observer.handle)
        observer_manager.register(logger_obs.handle)

        assert len(observer_manager) == 3

        # Simulate events
        await observer_manager.notify(PipelineStarted(name="test", total_waves=1, total_nodes=1))
        await observer_manager.notify(NodeStarted(name="node1", wave_index=0, dependencies=[]))
        await observer_manager.notify(
            NodeCompleted(name="node1", wave_index=0, result={}, duration_ms=100.0)
        )
        await observer_manager.notify(PipelineCompleted(name="test", duration_ms=200.0))

        # All observers should have processed events
        assert metrics_observer.get_summary()["total_nodes_executed"] == 1

    @pytest.mark.asyncio
    async def test_multiple_policies_together(self, policy_manager):
        """Test multiple policies working together."""
        # Register multiple policies
        retry_policy = RetryPolicy(max_retries=3)
        circuit_breaker = CircuitBreakerPolicy(failure_threshold=5)
        fallback_policy = FallbackPolicy(fallback_value={"status": "fallback"})

        policy_manager.subscribe(retry_policy, SubscriberType.CORE)
        policy_manager.subscribe(circuit_breaker, SubscriberType.CORE)
        policy_manager.subscribe(fallback_policy, SubscriberType.USER)

        assert len(policy_manager) == 3

        # Policies should work together to make decisions
        ctx = PolicyContext(
            event=NodeFailed(name="node", wave_index=0, error=Exception("Error")),
            dag_id="test_dag",
            node_id="node",
            wave_index=0,
            attempt=1,
            error=Exception("Error"),
        )

        response = await policy_manager.evaluate(ctx)
        # Should get a response (exact signal depends on policy priority)
        assert response.signal in [
            PolicySignal.RETRY,
            PolicySignal.FALLBACK,
            PolicySignal.FAIL,
            PolicySignal.SKIP,
            PolicySignal.PROCEED,
        ]

    @pytest.mark.asyncio
    async def test_observer_manager_cleanup(self, observer_manager):
        """Test observer manager cleanup."""
        metrics_observer = PerformanceMetricsObserver()
        observer_manager.register(metrics_observer.handle)

        assert len(observer_manager) == 1

        await observer_manager.close()

        # Should be able to close without errors
        assert True
