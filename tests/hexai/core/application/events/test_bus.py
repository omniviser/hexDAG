"""Tests for the EventBus - execution control system."""

import pytest

from hexai.core.application.events import (
    ControlHandler,
    EventBus,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    PipelineStarted,
)


class TestEventBus:
    """Test the EventBus for execution control."""

    @pytest.mark.asyncio
    async def test_no_handlers_allows_everything(self):
        """Test that with no handlers, all events are allowed."""
        bus = EventBus()

        # All events should be allowed
        assert await bus.check(NodeStarted(name="test", wave_index=1))
        assert await bus.check(NodeCompleted(name="test", wave_index=1, result={}, duration_ms=100))
        assert await bus.check(NodeFailed(name="test", wave_index=1, error=RuntimeError()))
        assert await bus.check(PipelineStarted(name="test", total_nodes=5, total_waves=2))

    @pytest.mark.asyncio
    async def test_veto_handler_blocks_events(self):
        """Test that handlers can veto specific events."""
        bus = EventBus()

        # Handler that vetoes nodes with specific names
        async def veto_forbidden_nodes(event):
            if isinstance(event, NodeStarted) and event.name == "forbidden":
                return False
            return True

        bus.register(veto_forbidden_nodes)

        # Normal nodes allowed
        assert await bus.check(NodeStarted(name="allowed", wave_index=1))
        assert await bus.check(NodeStarted(name="test", wave_index=1))

        # Forbidden node is vetoed
        assert not await bus.check(NodeStarted(name="forbidden", wave_index=1))

        # Other event types still allowed
        assert await bus.check(
            NodeCompleted(name="forbidden", wave_index=1, result={}, duration_ms=100)
        )

    @pytest.mark.asyncio
    async def test_multiple_handlers_all_must_approve(self):
        """Test that all handlers must approve for event to pass."""
        bus = EventBus()

        # First handler - allows everything
        async def allow_all(event):
            return True

        # Second handler - vetoes failures
        async def veto_failures(event):
            if isinstance(event, NodeFailed):
                return False
            return True

        # Third handler - limits node names
        async def limit_node_names(event):
            if hasattr(event, "name") and event.name.startswith("test_"):
                return True
            if hasattr(event, "name"):
                return False
            return True

        bus.register(allow_all)
        bus.register(veto_failures)
        bus.register(limit_node_names)

        # Node with valid name passes all handlers
        assert await bus.check(NodeStarted(name="test_node", wave_index=1))

        # Node with invalid name fails third handler
        assert not await bus.check(NodeStarted(name="bad_node", wave_index=1))

        # Failed node fails second handler
        assert not await bus.check(NodeFailed(name="test_node", wave_index=1, error=RuntimeError()))

    @pytest.mark.asyncio
    async def test_handler_class_implementation(self):
        """Test using a ControlHandler class instead of function."""
        bus = EventBus()

        class RateLimiter(ControlHandler):
            """Example rate limiter handler."""

            def __init__(self, max_events: int = 3):
                self.max_events = max_events
                self.event_count = 0

            async def check(self, event):
                self.event_count += 1
                if self.event_count > self.max_events:
                    return False  # Rate limit exceeded
                return True

        limiter = RateLimiter(max_events=3)
        bus.register(limiter)

        # First 3 events pass
        assert await bus.check(NodeStarted(name="node1", wave_index=1))
        assert await bus.check(NodeStarted(name="node2", wave_index=1))
        assert await bus.check(NodeStarted(name="node3", wave_index=1))

        # Fourth event is blocked
        assert not await bus.check(NodeStarted(name="node4", wave_index=1))

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

            async def check(self, event):
                if self.is_open:
                    return False  # Circuit is open, block everything

                if isinstance(event, NodeFailed):
                    self.failure_count += 1
                    if self.failure_count >= self.failure_threshold:
                        self.is_open = True
                        return False  # Open the circuit

                return True

        breaker = CircuitBreaker(failure_threshold=2)
        bus.register(breaker)

        # Normal events pass
        assert await bus.check(NodeStarted(name="test", wave_index=1))

        # First failure passes
        assert await bus.check(NodeFailed(name="fail1", wave_index=1, error=RuntimeError()))

        # Second failure triggers circuit breaker
        assert not await bus.check(NodeFailed(name="fail2", wave_index=1, error=RuntimeError()))

        # Now all events are blocked (circuit is open)
        assert not await bus.check(NodeStarted(name="test", wave_index=2))

    @pytest.mark.asyncio
    async def test_handler_error_results_in_veto(self):
        """Test that errors in handlers result in veto for safety."""
        bus = EventBus()

        # Handler that crashes
        async def broken_handler(event):
            raise RuntimeError("Handler error")

        bus.register(broken_handler)

        # Error means veto
        assert not await bus.check(NodeStarted(name="test", wave_index=1))

    @pytest.mark.asyncio
    async def test_unregister_handler(self):
        """Test removing handlers from the bus."""
        bus = EventBus()

        # Handler that vetoes everything
        async def veto_all(event):
            return False

        bus.register(veto_all)

        # Events are vetoed
        assert not await bus.check(NodeStarted(name="test", wave_index=1))

        # Unregister the handler
        bus.unregister(veto_all)

        # Events are now allowed
        assert await bus.check(NodeStarted(name="test", wave_index=1))

    def test_clear_all_handlers(self):
        """Test clearing all handlers."""
        bus = EventBus()

        # Add multiple handlers
        async def handler1(event):
            return True

        async def handler2(event):
            return True

        bus.register(handler1)
        bus.register(handler2)

        assert bus.has_handlers()

        # Clear all
        bus.clear()
        assert not bus.has_handlers()

    @pytest.mark.asyncio
    async def test_policy_enforcement_example(self):
        """Test using EventBus for policy enforcement."""
        bus = EventBus()

        # Business policy: no processing on weekends
        async def weekday_only_policy(event):
            # For demo, just check if node name contains "weekend"
            if hasattr(event, "name") and "weekend" in event.name:
                return False
            return True

        # Security policy: certain nodes require approval
        async def security_policy(event):
            if hasattr(event, "name") and event.name.startswith("sensitive_"):
                # In real scenario, might check for approval
                return False  # Block sensitive operations
            return True

        bus.register(weekday_only_policy)
        bus.register(security_policy)

        # Normal operations allowed
        assert await bus.check(NodeStarted(name="normal_task", wave_index=1))

        # Weekend operations blocked
        assert not await bus.check(NodeStarted(name="weekend_job", wave_index=1))

        # Sensitive operations blocked
        assert not await bus.check(NodeStarted(name="sensitive_data_export", wave_index=1))
