"""Tests for the ControlManager - execution control system."""

import pytest

from hexai.core.application.events.context import ExecutionContext
from hexai.core.application.events.control_manager import ControlHandler, ControlManager
from hexai.core.application.events.events import (
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    PipelineStarted,
)
from hexai.core.application.events.models import ControlResponse, ControlSignal


class TestControlManager:
    """Test the ControlManager for execution control."""

    @pytest.mark.asyncio
    async def test_no_handlers_allows_everything(self):
        """Test that with no handlers, all events are allowed."""
        control_manager = ControlManager()
        ctx = ExecutionContext(dag_id="test-dag")

        # All events should be allowed (PROCEED response)
        response = await control_manager.check(NodeStarted(name="test", wave_index=1), ctx)
        assert response.signal == ControlSignal.PROCEED

        response = await control_manager.check(
            NodeCompleted(name="test", wave_index=1, result={}, duration_ms=100), ctx
        )
        assert response.signal == ControlSignal.PROCEED

        response = await control_manager.check(
            NodeFailed(name="test", wave_index=1, error=RuntimeError()), ctx
        )
        assert response.signal == ControlSignal.PROCEED

        response = await control_manager.check(
            PipelineStarted(name="test", total_nodes=5, total_waves=2), ctx
        )
        assert response.signal == ControlSignal.PROCEED

    @pytest.mark.asyncio
    async def test_veto_handler_blocks_events(self):
        """Test that handlers can veto specific events."""
        control_manager = ControlManager()

        # Handler that vetoes nodes with specific names
        async def veto_forbidden_nodes(event, context):
            if isinstance(event, NodeStarted) and event.name == "forbidden":
                return ControlResponse(signal=ControlSignal.SKIP)
            return ControlResponse()

        control_manager.register(veto_forbidden_nodes)

        ctx = ExecutionContext(dag_id="test-dag")

        # Normal nodes allowed
        response = await control_manager.check(NodeStarted(name="allowed", wave_index=1), ctx)
        assert response.signal == ControlSignal.PROCEED

        response = await control_manager.check(NodeStarted(name="test", wave_index=1), ctx)
        assert response.signal == ControlSignal.PROCEED

        # Forbidden node is vetoed
        response = await control_manager.check(NodeStarted(name="forbidden", wave_index=1), ctx)
        assert response.signal == ControlSignal.SKIP

        # Other event types still allowed
        response = await control_manager.check(
            NodeCompleted(name="forbidden", wave_index=1, result={}, duration_ms=100), ctx
        )
        assert response.signal == ControlSignal.PROCEED

    @pytest.mark.asyncio
    async def test_multiple_handlers_all_must_approve(self):
        """Test that all handlers must approve for event to pass."""
        control_manager = ControlManager()

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

        control_manager.register(allow_all)
        control_manager.register(veto_failures)
        control_manager.register(limit_node_names)

        ctx = ExecutionContext(dag_id="test-dag")

        # Node with valid name passes all handlers
        response = await control_manager.check(NodeStarted(name="test_node", wave_index=1), ctx)
        assert response.signal == ControlSignal.PROCEED

        # Node with invalid name fails third handler
        response = await control_manager.check(NodeStarted(name="bad_node", wave_index=1), ctx)
        assert response.signal == ControlSignal.SKIP

        # Failed node fails second handler
        response = await control_manager.check(
            NodeFailed(name="test_node", wave_index=1, error=RuntimeError()), ctx
        )
        assert response.signal == ControlSignal.SKIP

    @pytest.mark.asyncio
    async def test_handler_class_implementation(self):
        """Test using a ControlHandler class instead of function."""
        control_manager = ControlManager()

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
        control_manager.register(limiter)

        ctx = ExecutionContext(dag_id="test-dag")

        # First 3 events pass
        response = await control_manager.check(NodeStarted(name="node1", wave_index=1), ctx)
        assert response.signal == ControlSignal.PROCEED

        response = await control_manager.check(NodeStarted(name="node2", wave_index=1), ctx)
        assert response.signal == ControlSignal.PROCEED

        response = await control_manager.check(NodeStarted(name="node3", wave_index=1), ctx)
        assert response.signal == ControlSignal.PROCEED

        # Fourth event is blocked
        response = await control_manager.check(NodeStarted(name="node4", wave_index=1), ctx)
        assert response.signal == ControlSignal.SKIP

    @pytest.mark.asyncio
    async def test_circuit_breaker_pattern(self):
        """Test implementing a circuit breaker with ControlHandler."""
        control_manager = ControlManager()

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
        control_manager.register(breaker)

        ctx = ExecutionContext(dag_id="test-dag")

        # Normal events pass
        response = await control_manager.check(NodeStarted(name="test", wave_index=1), ctx)
        assert response.signal == ControlSignal.PROCEED

        # First failure passes
        response = await control_manager.check(
            NodeFailed(name="fail1", wave_index=1, error=RuntimeError()), ctx
        )
        assert response.signal == ControlSignal.PROCEED

        # Second failure triggers circuit breaker
        response = await control_manager.check(
            NodeFailed(name="fail2", wave_index=1, error=RuntimeError()), ctx
        )
        assert response.signal == ControlSignal.FAIL

        # Now all events are blocked (circuit is open)
        response = await control_manager.check(NodeStarted(name="test", wave_index=2), ctx)
        assert response.signal == ControlSignal.FAIL

    @pytest.mark.asyncio
    async def test_handler_error_does_not_crash(self):
        """Test that errors in handlers are handled gracefully."""
        control_manager = ControlManager()

        # Handler that crashes
        async def broken_handler(event, context):
            raise RuntimeError("Handler error")

        control_manager.register(broken_handler)

        ctx = ExecutionContext(dag_id="test-dag")

        # Error is caught and logged, execution proceeds
        response = await control_manager.check(NodeStarted(name="test", wave_index=1), ctx)
        assert response.signal == ControlSignal.PROCEED

    @pytest.mark.asyncio
    async def test_clear_handlers(self):
        """Test clearing handlers from the bus."""
        control_manager = ControlManager()

        # Handler that vetoes everything
        async def veto_all(event, context):
            return ControlResponse(signal=ControlSignal.SKIP)

        control_manager.register(veto_all)

        ctx = ExecutionContext(dag_id="test-dag")

        # Events are vetoed
        response = await control_manager.check(NodeStarted(name="test", wave_index=1), ctx)
        assert response.signal == ControlSignal.SKIP

        # Clear the handlers
        control_manager.clear()

        # Events are now allowed
        response = await control_manager.check(NodeStarted(name="test", wave_index=1), ctx)
        assert response.signal == ControlSignal.PROCEED

    def test_clear_all_handlers(self):
        """Test clearing all handlers."""
        control_manager = ControlManager()

        # Add multiple handlers
        async def handler1(event, context):
            return ControlResponse()

        async def handler2(event, context):
            return ControlResponse()

        control_manager.register(handler1)
        control_manager.register(handler2)

        assert len(control_manager) == 2

        # Clear all
        control_manager.clear()
        assert len(control_manager) == 0

    @pytest.mark.asyncio
    async def test_policy_enforcement_example(self):
        """Test using ControlManager for policy enforcement."""
        control_manager = ControlManager()

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

        control_manager.register(weekday_only_policy)
        control_manager.register(security_policy)

        ctx = ExecutionContext(dag_id="test-dag")

        # Normal operations allowed
        response = await control_manager.check(NodeStarted(name="normal_task", wave_index=1), ctx)
        assert response.signal == ControlSignal.PROCEED

        # Weekend operations blocked
        response = await control_manager.check(NodeStarted(name="weekend_job", wave_index=1), ctx)
        assert response.signal == ControlSignal.SKIP

        # Sensitive operations blocked
        response = await control_manager.check(
            NodeStarted(name="sensitive_data_export", wave_index=1), ctx
        )
        assert response.signal == ControlSignal.SKIP
