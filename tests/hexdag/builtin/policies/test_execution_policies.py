"""Tests for execution policy implementations."""

import asyncio
import time

import pytest

from hexdag.builtin.policies import (
    CircuitBreakerPolicy,
    ConditionalSkipPolicy,
    ExponentialBackoffPolicy,
    FallbackPolicy,
    RateLimitPolicy,
    ResourceThrottlePolicy,
    RetryPolicy,
    TimeoutPolicy,
)
from hexdag.core.orchestration.events import NodeCompleted, NodeFailed, NodeStarted
from hexdag.core.orchestration.policies.models import PolicyContext, PolicySignal

# ==============================================================================
# RETRY POLICY TESTS
# ==============================================================================


class TestRetryPolicy:
    """Test RetryPolicy."""

    def test_retry_on_error_within_limit(self):
        """Test that policy signals RETRY when within retry limit."""
        policy = RetryPolicy(max_retries=3)

        context = PolicyContext(
            event=NodeFailed(name="node1", wave_index=0, error=Exception("error")),
            dag_id="dag1",
            node_id="node1",
            wave_index=0,
            attempt=1,
            error=Exception("error"),
        )

        response = asyncio.run(policy.evaluate(context))
        assert response.signal == PolicySignal.RETRY
        assert response.data["attempt"] == 1

    def test_no_retry_beyond_limit(self):
        """Test that policy doesn't retry beyond max attempts."""
        policy = RetryPolicy(max_retries=3)

        context = PolicyContext(
            event=NodeFailed(name="node1", wave_index=0, error=Exception("error")),
            dag_id="dag1",
            node_id="node1",
            wave_index=0,
            attempt=4,  # Beyond limit
            error=Exception("error"),
        )

        response = asyncio.run(policy.evaluate(context))
        assert response.signal == PolicySignal.PROCEED

    def test_no_retry_without_error(self):
        """Test that policy doesn't retry on success."""
        policy = RetryPolicy(max_retries=3)

        context = PolicyContext(
            event=NodeCompleted(name="node1", wave_index=0, result={}, duration_ms=100.0),
            dag_id="dag1",
            node_id="node1",
            wave_index=0,
            attempt=1,
        )

        response = asyncio.run(policy.evaluate(context))
        assert response.signal == PolicySignal.PROCEED

    def test_policy_name(self):
        """Test policy name generation."""
        policy = RetryPolicy(max_retries=5)
        assert policy.name == "retry_5"

    def test_policy_priority(self):
        """Test that retry has correct priority."""
        policy = RetryPolicy()
        assert policy.priority == 10


# ==============================================================================
# CIRCUIT BREAKER POLICY TESTS
# ==============================================================================


class TestCircuitBreakerPolicy:
    """Test CircuitBreakerPolicy."""

    def test_opens_after_threshold(self):
        """Test that circuit opens after failure threshold."""
        policy = CircuitBreakerPolicy(failure_threshold=3)

        # Trigger 3 failures
        for i in range(3):
            context = PolicyContext(
                event=NodeFailed(name="node1", wave_index=0, error=Exception("error")),
                dag_id="dag1",
                node_id="node1",
                wave_index=0,
                attempt=i + 1,
                error=Exception("error"),
            )
            asyncio.run(policy.evaluate(context))

        # Circuit should be open now
        assert policy.is_open is True

    def test_fails_when_open(self):
        """Test that policy fails all requests when circuit is open."""
        policy = CircuitBreakerPolicy(failure_threshold=1)

        # Open the circuit
        context = PolicyContext(
            event=NodeFailed(name="node1", wave_index=0, error=Exception("error")),
            dag_id="dag1",
            node_id="node1",
            wave_index=0,
            attempt=1,
            error=Exception("error"),
        )
        asyncio.run(policy.evaluate(context))

        # Next request should fail
        response = asyncio.run(policy.evaluate(context))
        assert response.signal == PolicySignal.FAIL

    def test_resets_on_success(self):
        """Test that failure count resets on success."""
        policy = CircuitBreakerPolicy(failure_threshold=3)

        # 2 failures
        for _ in range(2):
            context = PolicyContext(
                event=NodeFailed(name="node1", wave_index=0, error=Exception("error")),
                dag_id="dag1",
                node_id="node1",
                wave_index=0,
                attempt=1,
                error=Exception("error"),
            )
            asyncio.run(policy.evaluate(context))

        # Success should reset
        success_context = PolicyContext(
            event=NodeCompleted(name="node1", wave_index=0, result={}, duration_ms=100.0),
            dag_id="dag1",
            node_id="node1",
            wave_index=0,
            attempt=1,
        )
        asyncio.run(policy.evaluate(success_context))

        assert policy.failure_count == 0


# ==============================================================================
# FALLBACK POLICY TESTS
# ==============================================================================


class TestFallbackPolicy:
    """Test FallbackPolicy."""

    def test_provides_fallback_on_error(self):
        """Test that policy provides fallback value on error."""
        fallback_value = {"status": "fallback", "data": []}
        policy = FallbackPolicy(fallback_value=fallback_value)

        context = PolicyContext(
            event=NodeFailed(name="node1", wave_index=0, error=Exception("error")),
            dag_id="dag1",
            node_id="node1",
            wave_index=0,
            attempt=1,
            error=Exception("error"),
        )

        response = asyncio.run(policy.evaluate(context))
        assert response.signal == PolicySignal.FALLBACK
        assert response.data == fallback_value

    def test_no_fallback_on_success(self):
        """Test that policy doesn't provide fallback on success."""
        policy = FallbackPolicy(fallback_value={"fallback": True})

        context = PolicyContext(
            event=NodeCompleted(name="node1", wave_index=0, result={}, duration_ms=100.0),
            dag_id="dag1",
            node_id="node1",
            wave_index=0,
            attempt=1,
        )

        response = asyncio.run(policy.evaluate(context))
        assert response.signal == PolicySignal.PROCEED


# ==============================================================================
# TIMEOUT POLICY TESTS
# ==============================================================================


class TestTimeoutPolicy:
    """Test TimeoutPolicy."""

    @pytest.mark.asyncio
    async def test_tracks_start_time(self):
        """Test that policy tracks node start times."""
        policy = TimeoutPolicy(timeout_seconds=1.0)

        context = PolicyContext(
            event=NodeStarted(name="node1", wave_index=0, dependencies=[]),
            dag_id="dag1",
            node_id="node1",
            wave_index=0,
            attempt=1,
        )

        await policy.evaluate(context)
        assert "node1" in policy.start_times

    @pytest.mark.asyncio
    async def test_fails_on_timeout(self):
        """Test that policy fails when timeout is exceeded."""
        policy = TimeoutPolicy(timeout_seconds=0.1)  # 100ms timeout

        # Start tracking
        start_context = PolicyContext(
            event=NodeStarted(name="node1", wave_index=0, dependencies=[]),
            dag_id="dag1",
            node_id="node1",
            wave_index=0,
            attempt=1,
        )
        await policy.evaluate(start_context)

        # Wait for timeout
        await asyncio.sleep(0.2)

        # Check timeout
        check_context = PolicyContext(
            event=NodeCompleted(name="node1", wave_index=0, result={}, duration_ms=200.0),
            dag_id="dag1",
            node_id="node1",
            wave_index=0,
            attempt=1,
        )

        response = await policy.evaluate(check_context)
        assert response.signal == PolicySignal.FAIL
        assert "timeout" in response.data.get("reason", "")


# ==============================================================================
# RATE LIMIT POLICY TESTS
# ==============================================================================


class TestRateLimitPolicy:
    """Test RateLimitPolicy."""

    @pytest.mark.asyncio
    async def test_allows_within_limit(self):
        """Test that requests within limit are allowed."""
        policy = RateLimitPolicy(max_executions=3, window_seconds=1.0)

        # First 3 requests should be allowed
        for _ in range(3):
            context = PolicyContext(
                event=NodeStarted(name="node1", wave_index=0, dependencies=[]),
                dag_id="dag1",
                node_id="node1",
                wave_index=0,
                attempt=1,
            )
            response = await policy.evaluate(context)
            assert response.signal == PolicySignal.PROCEED

    @pytest.mark.asyncio
    async def test_skips_beyond_limit(self):
        """Test that requests beyond limit are skipped."""
        policy = RateLimitPolicy(max_executions=2, window_seconds=1.0)

        # First 2 allowed
        for _ in range(2):
            context = PolicyContext(
                event=NodeStarted(name="node1", wave_index=0, dependencies=[]),
                dag_id="dag1",
                node_id="node1",
                wave_index=0,
                attempt=1,
            )
            await policy.evaluate(context)

        # 3rd should be skipped
        context = PolicyContext(
            event=NodeStarted(name="node1", wave_index=0, dependencies=[]),
            dag_id="dag1",
            node_id="node1",
            wave_index=0,
            attempt=1,
        )
        response = await policy.evaluate(context)
        assert response.signal == PolicySignal.SKIP

    @pytest.mark.asyncio
    async def test_window_resets(self):
        """Test that rate limit window resets over time."""
        policy = RateLimitPolicy(max_executions=1, window_seconds=0.1)

        # First request
        context = PolicyContext(
            event=NodeStarted(name="node1", wave_index=0, dependencies=[]),
            dag_id="dag1",
            node_id="node1",
            wave_index=0,
            attempt=1,
        )
        response1 = await policy.evaluate(context)
        assert response1.signal == PolicySignal.PROCEED

        # Wait for window to reset
        await asyncio.sleep(0.15)

        # Should be allowed again
        response2 = await policy.evaluate(context)
        assert response2.signal == PolicySignal.PROCEED


# ==============================================================================
# CONDITIONAL SKIP POLICY TESTS
# ==============================================================================


class TestConditionalSkipPolicy:
    """Test ConditionalSkipPolicy."""

    @pytest.mark.asyncio
    async def test_skips_when_condition_met(self):
        """Test that policy skips when condition is met."""

        def should_skip(ctx):
            return "test" in ctx.node_id.lower()

        policy = ConditionalSkipPolicy(should_skip=should_skip)

        context = PolicyContext(
            event=NodeStarted(name="test_node", wave_index=0, dependencies=[]),
            dag_id="dag1",
            node_id="test_node",
            wave_index=0,
            attempt=1,
        )

        response = await policy.evaluate(context)
        assert response.signal == PolicySignal.SKIP

    @pytest.mark.asyncio
    async def test_proceeds_when_condition_not_met(self):
        """Test that policy proceeds when condition is not met."""

        def should_skip(ctx):
            return "test" in ctx.node_id.lower()

        policy = ConditionalSkipPolicy(should_skip=should_skip)

        context = PolicyContext(
            event=NodeStarted(name="production_node", wave_index=0, dependencies=[]),
            dag_id="dag1",
            node_id="production_node",
            wave_index=0,
            attempt=1,
        )

        response = await policy.evaluate(context)
        assert response.signal == PolicySignal.PROCEED

    @pytest.mark.asyncio
    async def test_handles_predicate_errors(self):
        """Test that policy handles predicate errors gracefully."""

        def failing_predicate(ctx):
            raise Exception("Predicate error")

        policy = ConditionalSkipPolicy(should_skip=failing_predicate)

        context = PolicyContext(
            event=NodeStarted(name="node1", wave_index=0, dependencies=[]),
            dag_id="dag1",
            node_id="node1",
            wave_index=0,
            attempt=1,
        )

        # Should not crash, should proceed on error
        response = await policy.evaluate(context)
        assert response.signal == PolicySignal.PROCEED


# ==============================================================================
# RESOURCE THROTTLE POLICY TESTS
# ==============================================================================


class TestResourceThrottlePolicy:
    """Test ResourceThrottlePolicy."""

    @pytest.mark.asyncio
    async def test_allows_within_resource_limit(self):
        """Test that executions within resource limit are allowed."""
        policy = ResourceThrottlePolicy(max_concurrent=2, resource_cost=1)

        # Start 2 nodes (should be allowed)
        for i in range(2):
            context = PolicyContext(
                event=NodeStarted(name=f"node{i}", wave_index=0, dependencies=[]),
                dag_id="dag1",
                node_id=f"node{i}",
                wave_index=0,
                attempt=1,
            )
            response = await policy.evaluate(context)
            assert response.signal == PolicySignal.PROCEED

    @pytest.mark.asyncio
    async def test_skips_when_resources_exhausted(self):
        """Test that executions are skipped when resources exhausted."""
        policy = ResourceThrottlePolicy(max_concurrent=1, resource_cost=1)

        # Start first node
        context1 = PolicyContext(
            event=NodeStarted(name="node1", wave_index=0, dependencies=[]),
            dag_id="dag1",
            node_id="node1",
            wave_index=0,
            attempt=1,
        )
        await policy.evaluate(context1)

        # Try to start second node (should be skipped)
        context2 = PolicyContext(
            event=NodeStarted(name="node2", wave_index=0, dependencies=[]),
            dag_id="dag1",
            node_id="node2",
            wave_index=0,
            attempt=1,
        )
        response = await policy.evaluate(context2)
        assert response.signal == PolicySignal.SKIP

    @pytest.mark.asyncio
    async def test_releases_resources_on_completion(self):
        """Test that resources are released when node completes."""
        policy = ResourceThrottlePolicy(max_concurrent=1, resource_cost=1)

        # Start node
        start_ctx = PolicyContext(
            event=NodeStarted(name="node1", wave_index=0, dependencies=[]),
            dag_id="dag1",
            node_id="node1",
            wave_index=0,
            attempt=1,
        )
        await policy.evaluate(start_ctx)

        # Complete node
        complete_ctx = PolicyContext(
            event=NodeCompleted(name="node1", wave_index=0, result={}, duration_ms=100.0),
            dag_id="dag1",
            node_id="node1",
            wave_index=0,
            attempt=1,
        )
        await policy.evaluate(complete_ctx)

        assert policy.current_usage == 0


# ==============================================================================
# EXPONENTIAL BACKOFF POLICY TESTS
# ==============================================================================


class TestExponentialBackoffPolicy:
    """Test ExponentialBackoffPolicy."""

    @pytest.mark.asyncio
    async def test_retries_with_increasing_delay(self):
        """Test that policy retries with exponentially increasing delay."""
        policy = ExponentialBackoffPolicy(
            max_retries=3, initial_delay_ms=10.0, backoff_factor=2.0, max_delay_ms=1000.0
        )

        # First retry
        context1 = PolicyContext(
            event=NodeFailed(name="node1", wave_index=0, error=Exception("error")),
            dag_id="dag1",
            node_id="node1",
            wave_index=0,
            attempt=1,
            error=Exception("error"),
        )

        start = time.time()
        response1 = await policy.evaluate(context1)
        elapsed1 = (time.time() - start) * 1000

        assert response1.signal == PolicySignal.RETRY
        assert elapsed1 >= 10.0  # Should wait at least initial delay

        # Second retry should have longer delay
        context2 = PolicyContext(
            event=NodeFailed(name="node1", wave_index=0, error=Exception("error")),
            dag_id="dag1",
            node_id="node1",
            wave_index=0,
            attempt=2,
            error=Exception("error"),
        )

        start = time.time()
        response2 = await policy.evaluate(context2)
        elapsed2 = (time.time() - start) * 1000

        assert response2.signal == PolicySignal.RETRY
        assert elapsed2 >= 20.0  # Should wait at least 2x initial delay

    @pytest.mark.asyncio
    async def test_respects_max_delay(self):
        """Test that delay doesn't exceed max_delay_ms."""
        policy = ExponentialBackoffPolicy(
            max_retries=10, initial_delay_ms=10.0, backoff_factor=2.0, max_delay_ms=50.0
        )

        # High attempt number
        context = PolicyContext(
            event=NodeFailed(name="node1", wave_index=0, error=Exception("error")),
            dag_id="dag1",
            node_id="node1",
            wave_index=0,
            attempt=10,  # Would normally be 10 * 2^9 = 5120ms
            error=Exception("error"),
        )

        start = time.time()
        response = await policy.evaluate(context)
        elapsed = (time.time() - start) * 1000

        assert response.signal == PolicySignal.RETRY
        # Should cap at max_delay_ms
        assert elapsed < 100.0  # Should not wait more than 2x max_delay

    @pytest.mark.asyncio
    async def test_stops_after_max_retries(self):
        """Test that policy stops retrying after max attempts."""
        policy = ExponentialBackoffPolicy(max_retries=2, initial_delay_ms=1.0)

        # Beyond max retries
        context = PolicyContext(
            event=NodeFailed(name="node1", wave_index=0, error=Exception("error")),
            dag_id="dag1",
            node_id="node1",
            wave_index=0,
            attempt=3,
            error=Exception("error"),
        )

        response = await policy.evaluate(context)
        assert response.signal == PolicySignal.PROCEED


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


class TestPolicyIntegration:
    """Integration tests for policies working together."""

    @pytest.mark.asyncio
    async def test_policy_priority_order(self):
        """Test that policies execute in priority order."""
        # Create policies with different priorities
        circuit = CircuitBreakerPolicy(failure_threshold=1)
        retry = RetryPolicy(max_retries=3)

        # Circuit breaker has priority 5, retry has priority 10
        assert circuit.priority < retry.priority

    @pytest.mark.asyncio
    async def test_multiple_policies_same_event(self):
        """Test multiple policies evaluating same event."""
        retry = RetryPolicy(max_retries=3)
        fallback = FallbackPolicy(fallback_value={"fallback": True})

        context = PolicyContext(
            event=NodeFailed(name="node1", wave_index=0, error=Exception("error")),
            dag_id="dag1",
            node_id="node1",
            wave_index=0,
            attempt=1,
            error=Exception("error"),
        )

        # Both should handle the event independently
        retry_resp = await retry.evaluate(context)
        fallback_resp = await fallback.evaluate(context)

        assert retry_resp.signal == PolicySignal.RETRY
        assert fallback_resp.signal == PolicySignal.FALLBACK
