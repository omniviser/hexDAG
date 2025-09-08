"""Tests for the EventBus - execution control system."""

import pytest

from hexai.core.application.events.bus import ControlHandler, EventBus
from hexai.core.application.events.context import ExecutionContext
from hexai.core.application.events.events import (
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    PipelineStarted,
)
from hexai.core.application.events.models import ControlResponse, ControlSignal


class TestEventBus:
    """Test the EventBus for execution control."""

    @pytest.mark.asyncio
    async def test_no_handlers_allows_everything(self):
        """Test that with no handlers, all events are allowed."""
        bus = EventBus()
        ctx = ExecutionContext(dag_id="test-dag")

        # All events should be allowed (PROCEED response)
        response = await bus.check(NodeStarted(name="test", wave_index=1), ctx)
        assert response.signal == ControlSignal.PROCEED

        response = await bus.check(
            NodeCompleted(name="test", wave_index=1, result={}, duration_ms=100), ctx
        )
        assert response.signal == ControlSignal.PROCEED

        response = await bus.check(NodeFailed(name="test", wave_index=1, error=RuntimeError()), ctx)
        assert response.signal == ControlSignal.PROCEED

        response = await bus.check(PipelineStarted(name="test", total_nodes=5, total_waves=2), ctx)
        assert response.signal == ControlSignal.PROCEED

    @pytest.mark.asyncio
    async def test_veto_handler_blocks_events(self):
        """Test that handlers can veto specific events."""
        bus = EventBus()

        # Handler that vetoes nodes with specific names
        async def veto_forbidden_nodes(event, context):
            if isinstance(event, NodeStarted) and event.name == "forbidden":
                return ControlResponse(signal=ControlSignal.SKIP)
            return ControlResponse()

        bus.register(veto_forbidden_nodes)

        ctx = ExecutionContext(dag_id="test-dag")

        # Normal nodes allowed
        response = await bus.check(NodeStarted(name="allowed", wave_index=1), ctx)
        assert response.signal == ControlSignal.PROCEED

        response = await bus.check(NodeStarted(name="test", wave_index=1), ctx)
        assert response.signal == ControlSignal.PROCEED

        # Forbidden node is vetoed
        response = await bus.check(NodeStarted(name="forbidden", wave_index=1), ctx)
        assert response.signal == ControlSignal.SKIP

        # Other event types still allowed
        response = await bus.check(
            NodeCompleted(name="forbidden", wave_index=1, result={}, duration_ms=100), ctx
        )
        assert response.signal == ControlSignal.PROCEED

    @pytest.mark.asyncio
    async def test_multiple_handlers_all_must_approve(self):
        """Test that all handlers must approve for event to pass."""
        bus = EventBus()

        # First handler - allows everything
        async def allow_all(event, context):
            return ControlResponse()

        # Second handler - vetoes failures
        async def veto_failures(event, context):
            if isinstance(event, NodeFailed):
                return ControlResponse(signal=ControlSignal.SKIP)
            return ControlResponse()

        # Third handler - limits node names
        async def limit_node_names(event, context):
            if hasattr(event, "name") and event.name.startswith("test_"):
                return ControlResponse()
            if hasattr(event, "name"):
                return ControlResponse(signal=ControlSignal.SKIP)
            return ControlResponse()

        bus.register(allow_all)
        bus.register(veto_failures)
        bus.register(limit_node_names)

        ctx = ExecutionContext(dag_id="test-dag")

        # Node with valid name passes all handlers
        response = await bus.check(NodeStarted(name="test_node", wave_index=1), ctx)
        assert response.signal == ControlSignal.PROCEED

        # Node with invalid name fails third handler
        response = await bus.check(NodeStarted(name="bad_node", wave_index=1), ctx)
        assert response.signal == ControlSignal.SKIP

        # Failed node fails second handler
        response = await bus.check(
            NodeFailed(name="test_node", wave_index=1, error=RuntimeError()), ctx
        )
        assert response.signal == ControlSignal.SKIP

    @pytest.mark.asyncio
    async def test_handler_class_implementation(self):
        """Test using a ControlHandler class instead of function."""
        bus = EventBus()

        class RateLimiter(ControlHandler):
            """Example rate limiter handler."""

            def __init__(self, max_events: int = 3):
                self.max_events = max_events
                self.event_count = 0

            async def handle(self, event, context):
                self.event_count += 1
                if self.event_count > self.max_events:
                    return ControlResponse(signal=ControlSignal.SKIP)  # Rate limit exceeded
                return ControlResponse()

        limiter = RateLimiter(max_events=3)
        bus.register(limiter)

        ctx = ExecutionContext(dag_id="test-dag")

        # First 3 events pass
        response = await bus.check(NodeStarted(name="node1", wave_index=1), ctx)
        assert response.signal == ControlSignal.PROCEED

        response = await bus.check(NodeStarted(name="node2", wave_index=1), ctx)
        assert response.signal == ControlSignal.PROCEED

        response = await bus.check(NodeStarted(name="node3", wave_index=1), ctx)
        assert response.signal == ControlSignal.PROCEED

        # Fourth event is blocked
        response = await bus.check(NodeStarted(name="node4", wave_index=1), ctx)
        assert response.signal == ControlSignal.SKIP

    @pytest.mark.asyncio
    async def test_circuit_breaker_pattern(self):
        """Test implementing a circuit breaker with ControlHandler."""
        bus = EventBus()

        class CircuitBreaker(ControlHandler):
            """Circuit breaker that opens after too many failures."""

            def __init__(self, failure_threshold: int = 2):
                self.failure_threshold = failure_threshold
                self.failure_count = 0
                self.is_open = False

            async def handle(self, event, context):
                if self.is_open:
                    return ControlResponse(
                        signal=ControlSignal.FAIL
                    )  # Circuit is open, block everything

                if isinstance(event, NodeFailed):
                    self.failure_count += 1
                    if self.failure_count >= self.failure_threshold:
                        self.is_open = True
                        return ControlResponse(signal=ControlSignal.FAIL)  # Open the circuit

                return ControlResponse()

        breaker = CircuitBreaker(failure_threshold=2)
        bus.register(breaker)

        ctx = ExecutionContext(dag_id="test-dag")

        # Normal events pass
        response = await bus.check(NodeStarted(name="test", wave_index=1), ctx)
        assert response.signal == ControlSignal.PROCEED

        # First failure passes
        response = await bus.check(
            NodeFailed(name="fail1", wave_index=1, error=RuntimeError()), ctx
        )
        assert response.signal == ControlSignal.PROCEED

        # Second failure triggers circuit breaker
        response = await bus.check(
            NodeFailed(name="fail2", wave_index=1, error=RuntimeError()), ctx
        )
        assert response.signal == ControlSignal.FAIL

        # Now all events are blocked (circuit is open)
        response = await bus.check(NodeStarted(name="test", wave_index=2), ctx)
        assert response.signal == ControlSignal.FAIL

    @pytest.mark.asyncio
    async def test_handler_error_does_not_crash(self):
        """Test that errors in handlers are handled gracefully."""
        bus = EventBus()

        # Handler that crashes
        async def broken_handler(event, context):
            raise RuntimeError("Handler error")

        bus.register(broken_handler)

        ctx = ExecutionContext(dag_id="test-dag")

        # Error is caught and logged, execution proceeds
        response = await bus.check(NodeStarted(name="test", wave_index=1), ctx)
        assert response.signal == ControlSignal.PROCEED

    @pytest.mark.asyncio
    async def test_clear_handlers(self):
        """Test clearing handlers from the bus."""
        bus = EventBus()

        # Handler that vetoes everything
        async def veto_all(event, context):
            return ControlResponse(signal=ControlSignal.SKIP)

        bus.register(veto_all)

        ctx = ExecutionContext(dag_id="test-dag")

        # Events are vetoed
        response = await bus.check(NodeStarted(name="test", wave_index=1), ctx)
        assert response.signal == ControlSignal.SKIP

        # Clear the handlers
        bus.clear()

        # Events are now allowed
        response = await bus.check(NodeStarted(name="test", wave_index=1), ctx)
        assert response.signal == ControlSignal.PROCEED

    def test_clear_all_handlers(self):
        """Test clearing all handlers."""
        bus = EventBus()

        # Add multiple handlers
        async def handler1(event, context):
            return ControlResponse()

        async def handler2(event, context):
            return ControlResponse()

        bus.register(handler1)
        bus.register(handler2)

        assert len(bus._handlers) == 2

        # Clear all
        bus.clear()
        assert len(bus._handlers) == 0

    @pytest.mark.asyncio
    async def test_policy_enforcement_example(self):
        """Test using EventBus for policy enforcement."""
        bus = EventBus()

        # Business policy: no processing on weekends
        async def weekday_only_policy(event, context):
            # For demo, just check if node name contains "weekend"
            if hasattr(event, "name") and "weekend" in event.name:
                return ControlResponse(signal=ControlSignal.SKIP)
            return ControlResponse()

        # Security policy: certain nodes require approval
        async def security_policy(event, context):
            if hasattr(event, "name") and event.name.startswith("sensitive_"):
                # In real scenario, might check for approval
                return ControlResponse(signal=ControlSignal.SKIP)  # Block sensitive operations
            return ControlResponse()

        bus.register(weekday_only_policy)
        bus.register(security_policy)

        ctx = ExecutionContext(dag_id="test-dag")

        # Normal operations allowed
        response = await bus.check(NodeStarted(name="normal_task", wave_index=1), ctx)
        assert response.signal == ControlSignal.PROCEED

        # Weekend operations blocked
        response = await bus.check(NodeStarted(name="weekend_job", wave_index=1), ctx)
        assert response.signal == ControlSignal.SKIP

        # Sensitive operations blocked
        response = await bus.check(NodeStarted(name="sensitive_data_export", wave_index=1), ctx)
        assert response.signal == ControlSignal.SKIP
