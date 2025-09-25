"""Tests for the ControlManager - execution control system."""

import pytest

from hexai.core.application.events import (
    ControlHandler,
    ControlManager,
    ControlResponse,
    ControlSignal,
    ExecutionContext,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    PipelineStarted,
    control_handler,
)


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
        async def veto_forbidden_nodes(event, context) -> ControlResponse:
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
        async def allow_all(event, context) -> ControlResponse:
            return ControlResponse()

        # Second handler - vetoes failures
        async def veto_failures(event, context) -> ControlResponse:
            if isinstance(event, NodeFailed):
                return ControlResponse(signal=ControlSignal.SKIP)
            return ControlResponse()

        # Third handler - limits node names
        async def limit_node_names(event, context) -> ControlResponse:
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

            async def handle(self, event, context) -> ControlResponse:
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

            async def handle(self, event, context) -> ControlResponse:
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
        async def broken_handler(event, context) -> ControlResponse:
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
        async def veto_all(event, context) -> ControlResponse:
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

    @pytest.mark.asyncio
    async def test_handler_only_checks_registered_events(self):
        """Test that handlers only process events they registered for."""
        manager = ControlManager()
        context = ExecutionContext(dag_id="test")

        from hexai.core.application.events import (
            NodeCompleted,
            PipelineStarted,
            ToolCalled,
            ToolCompleted,
        )

        # Create handlers with different filters
        async def node_handler(event, ctx) -> ControlResponse:
            return ControlResponse(signal=ControlSignal.RETRY)

        async def tool_handler(event, ctx) -> ControlResponse:
            return ControlResponse(signal=ControlSignal.SKIP)

        manager.register(node_handler, name="node_policy", event_types=[NodeStarted, NodeCompleted])
        manager.register(tool_handler, name="tool_policy", event_types=[ToolCalled, ToolCompleted])

        # Check NodeStarted - should trigger retry
        response = await manager.check(NodeStarted(name="test", wave_index=1), context)
        assert response.signal == ControlSignal.RETRY

        # Check ToolCalled - should trigger skip
        response = await manager.check(
            ToolCalled(node_name="test", tool_name="api", params={}), context
        )
        assert response.signal == ControlSignal.SKIP

        # Check PipelineStarted - no handlers registered, should proceed
        response = await manager.check(
            PipelineStarted(name="pipeline", total_waves=1, total_nodes=1), context
        )
        assert response.signal == ControlSignal.PROCEED

    @pytest.mark.asyncio
    async def test_handler_no_filter_checks_all(self):
        """Test that handler without filter checks all events."""
        manager = ControlManager()
        context = ExecutionContext(dag_id="test")

        from hexai.core.application.events import PipelineCompleted, ToolCalled

        call_count = 0

        async def universal_handler(event, ctx) -> ControlResponse:
            nonlocal call_count
            call_count += 1
            return ControlResponse()

        manager.register(universal_handler, name="universal")  # No event_types

        # Check various events
        await manager.check(NodeStarted(name="test", wave_index=1), context)
        await manager.check(ToolCalled(node_name="test", tool_name="api", params={}), context)
        await manager.check(PipelineCompleted(name="pipeline", duration_ms=1000), context)

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_filtering_with_priority(self):
        """Test that filtering works correctly with priority ordering."""
        manager = ControlManager()
        context = ExecutionContext(dag_id="test")

        from hexai.core.application.events import ToolCalled

        # High priority handler for nodes only
        async def high_priority_node_handler(event, ctx) -> ControlResponse:
            return ControlResponse(signal=ControlSignal.SKIP)

        # Low priority universal handler
        async def low_priority_handler(event, ctx) -> ControlResponse:
            return ControlResponse(signal=ControlSignal.RETRY)

        manager.register(
            high_priority_node_handler,
            priority=10,
            event_types=[NodeStarted],  # High priority
        )
        manager.register(
            low_priority_handler,
            priority=100,  # Low priority
            # No filter - handles all
        )

        # NodeStarted should be skipped by high priority handler
        response = await manager.check(NodeStarted(name="test", wave_index=1), context)
        assert response.signal == ControlSignal.SKIP

        # ToolCalled should be retried by low priority handler (high priority doesn't match)
        response = await manager.check(
            ToolCalled(node_name="test", tool_name="api", params={}), context
        )
        assert response.signal == ControlSignal.RETRY

    @pytest.mark.asyncio
    async def test_empty_filter_list_never_called(self):
        """Test that handler with empty event list is never called."""
        manager = ControlManager()
        context = ExecutionContext(dag_id="test")

        from hexai.core.application.events import PipelineCompleted

        async def never_handler(event, ctx) -> ControlResponse:
            return ControlResponse(signal=ControlSignal.FAIL)

        manager.register(never_handler, event_types=[])  # Empty list

        # Check various events - should all proceed
        response = await manager.check(NodeStarted(name="test", wave_index=1), context)
        assert response.signal == ControlSignal.PROCEED

        response = await manager.check(
            PipelineCompleted(name="pipeline", duration_ms=1000), context
        )
        assert response.signal == ControlSignal.PROCEED

    @pytest.mark.asyncio
    async def test_policy_enforcement_example(self):
        """Test using ControlManager for policy enforcement."""
        control_manager = ControlManager()

        # Business policy: no processing on weekends
        async def weekday_only_policy(event, context) -> ControlResponse:
            # For demo, just check if node name contains "weekend"
            if hasattr(event, "name") and "weekend" in event.name:
                return ControlResponse(signal=ControlSignal.SKIP)
            return ControlResponse()

        # Security policy: certain nodes require approval
        async def security_policy(event, context) -> ControlResponse:
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

    @pytest.mark.asyncio
    async def test_control_weak_ref_cleanup(self):
        """Test that control handlers are cleaned up when appropriate."""
        manager = ControlManager(use_weak_refs=True)

        # Create a handler object
        class TestHandler:
            async def handle(self, event, context) -> ControlResponse:
                return ControlResponse()

        handler = TestHandler()

        # Register with keep_alive=False - but handlers are kept in heap which maintains strong ref
        # This is expected behavior for ControlManager since handlers need to
        # stay alive for priority ordering
        manager.register(handler, priority=10, name="test_handler")

        # Handler should be in heap
        assert len(manager._handler_heap) == 1

        # For ControlManager, the handler stays alive because it's stored in the heap entry
        # This is different from ObserverManager which truly uses weak refs
        # We can still test that explicit unregister works

        # Unregister the handler
        manager.unregister("test_handler")

        # Handler should be marked as deleted (lazy deletion)
        assert manager._handler_heap[0].deleted is True

        # Should return default PROCEED since handler is deleted
        context = ExecutionContext(dag_id="test")
        response = await manager.check(NodeStarted(name="test", wave_index=0), context)
        assert response.signal.value == "proceed"

    @pytest.mark.asyncio
    async def test_control_function_kept_alive(self):
        """Test that wrapped control functions are kept alive."""
        import gc

        manager = ControlManager(use_weak_refs=True)

        called = []

        # Register a function (will be wrapped)
        def control_func(event, context) -> ControlResponse:
            called.append(event)
            return ControlResponse()

        manager.register(control_func, priority=10, name="test_func")

        # Function should be in strong refs
        assert "test_func" in manager._strong_refs

        # Delete original function
        del control_func
        gc.collect()

        # Should still work
        context = ExecutionContext(dag_id="test")
        response = await manager.check(NodeStarted(name="test", wave_index=0), context)

        # Verify it was called
        assert len(called) == 1
        assert response.signal.value == "proceed"

    def test_control_no_weak_refs_mode(self):
        """Test that control manager works without weak refs."""
        ctrl_manager = ControlManager(use_weak_refs=False)

        class TestHandler:
            async def handle(self, event, context) -> ControlResponse:
                return ControlResponse()

        handler = TestHandler()
        ctrl_manager.register(handler, name="test")

        # Should be in normal handlers dict
        assert "test" in ctrl_manager._handlers
        assert isinstance(ctrl_manager._handlers["test"], TestHandler)


@pytest.mark.asyncio
async def test_control_handler_decorator_applies_metadata():
    """Decorated control handlers should populate metadata defaults."""
    manager = ControlManager()
    calls: list[NodeFailed] = []

    @control_handler(
        "retry_on_failure",
        priority=5,
        event_types=[NodeFailed],
        description="Retry failing nodes",
    )
    async def retry_on_failure(event, context) -> ControlResponse:
        calls.append(event)
        return ControlResponse(signal=ControlSignal.RETRY)

    manager.register(retry_on_failure)

    ctx = ExecutionContext(dag_id="decorated")
    failed_event = NodeFailed(name="node", wave_index=1, error=RuntimeError("boom"))
    other_event = NodeStarted(name="node", wave_index=1)

    response = await manager.check(failed_event, ctx)
    assert response.signal == ControlSignal.RETRY
    assert calls == [failed_event]

    response = await manager.check(other_event, ctx)
    assert response.signal == ControlSignal.PROCEED

    entry = manager._handler_index["retry_on_failure"]
    assert entry.priority == 5
    assert entry.event_types == {NodeFailed}
    assert entry.metadata.description == "Retry failing nodes"

@pytest.mark.asyncio
async def test_control_handler_metadata_overrides():
    """Explicit kwargs should override decorator metadata."""
    manager = ControlManager()

    @control_handler("decorated_handler", priority=30, event_types=[NodeFailed])
    async def decorated_handler(event, context) -> ControlResponse:
        return ControlResponse(signal=ControlSignal.FAIL)

    manager.register(
        decorated_handler,
        name="override_name",
        priority=15,
        event_types=[NodeStarted],
        description="overridden",
    )

    entry = manager._handler_index["override_name"]
    assert entry.priority == 15
    assert entry.event_types == {NodeStarted}
    assert entry.metadata.description == "overridden"

def test_control_handler_missing_annotation_raises():
    """Handlers must declare ControlResponse return type."""
    manager = ControlManager()

    async def missing_annotation(event, context):
        return ControlResponse()

    with pytest.raises(TypeError):
        manager.register(missing_annotation)

def test_control_handler_wrong_annotation_raises():
    """Return annotations other than ControlResponse are rejected."""
    manager = ControlManager()

    async def wrong_return(event, context) -> str:
        return "nope"

    with pytest.raises(TypeError):
        manager.register(wrong_return)

@pytest.mark.asyncio
async def test_control_handler_runtime_wrong_type_raises():
    """Handlers that return the wrong type at runtime should raise."""
    manager = ControlManager()

    @control_handler("bad_runtime")
    async def bad_runtime(event, context) -> ControlResponse:  # type: ignore[return-value]
        return "oops"  # type: ignore[return-value]

    manager.register(bad_runtime)

    ctx = ExecutionContext(dag_id="runtime")
    with pytest.raises(TypeError):
        await manager.check(NodeStarted(name="node", wave_index=0), ctx)
