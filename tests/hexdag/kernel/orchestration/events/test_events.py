"""Tests for event data classes and decorator integrations."""

import asyncio
from datetime import datetime
from typing import Any

import pytest

from hexdag.drivers.observer_manager import LocalObserverManager
from hexdag.kernel.orchestration.events.decorators import (
    observer,
)
from hexdag.kernel.orchestration.events.events import (
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    PipelineCompleted,
    PipelineStarted,
    WaveCompleted,
)
from hexdag.kernel.ports.llm import LLMGeneration
from hexdag.kernel.ports.tool_router import ToolRouterEvent


class TestNodeEvents:
    """Test node-level event data classes."""

    def test_node_started_creation(self):
        """Test NodeStarted event creation."""
        event = NodeStarted(name="test_node", wave_index=1, dependencies=["dep1", "dep2"])

        assert event.name == "test_node"
        assert event.wave_index == 1
        assert event.dependencies == ["dep1", "dep2"]
        assert isinstance(event.timestamp, datetime)

    def test_node_started_defaults(self):
        """Test NodeStarted with default values."""
        event = NodeStarted(name="test", wave_index=0)

        assert event.name == "test"
        assert event.wave_index == 0
        assert event.dependencies == ()  # Empty tuple (performance optimization)
        assert isinstance(event.timestamp, datetime)

    def test_node_completed_creation(self):
        """Test NodeCompleted event creation."""
        result = {"output": "test_output", "score": 0.95}
        event = NodeCompleted(name="test_node", wave_index=1, result=result, duration_ms=1500.5)

        assert event.name == "test_node"
        assert event.wave_index == 1
        assert event.result == result
        assert event.duration_ms == 1500.5
        assert isinstance(event.timestamp, datetime)

    def test_node_failed_creation(self):
        """Test NodeFailed event creation."""
        error = RuntimeError("Test error message")
        event = NodeFailed(name="failing_node", wave_index=2, error=error)

        assert event.name == "failing_node"
        assert event.wave_index == 2
        assert event.error is error
        assert isinstance(event.timestamp, datetime)


class TestWaveEvents:
    """Test wave-level event data classes."""

    def test_wave_completed_creation(self):
        """Test WaveCompleted event creation."""
        event = WaveCompleted(wave_index=2, duration_ms=3000.0)

        assert event.wave_index == 2
        assert event.duration_ms == 3000.0
        assert event.nodes == []
        assert isinstance(event.timestamp, datetime)

    def test_wave_completed_with_nodes(self):
        """Test WaveCompleted event creation with nodes list."""
        nodes = ["node1", "node2", "node3"]
        event = WaveCompleted(wave_index=1, duration_ms=1500.0, nodes=nodes)

        assert event.wave_index == 1
        assert event.duration_ms == 1500.0
        assert event.nodes == nodes
        assert isinstance(event.timestamp, datetime)


class TestPipelineEvents:
    """Test pipeline-level event data classes."""

    def test_pipeline_started_creation(self):
        """Test PipelineStarted event creation."""
        event = PipelineStarted(name="test_pipeline", total_waves=3, total_nodes=10)

        assert event.name == "test_pipeline"
        assert event.total_waves == 3
        assert event.total_nodes == 10
        assert isinstance(event.timestamp, datetime)

    def test_pipeline_completed_creation(self):
        """Test PipelineCompleted event creation."""
        node_results = {"node1": {"output": 1}, "node2": {"output": 2}}
        event = PipelineCompleted(
            name="test_pipeline", duration_ms=5000.0, node_results=node_results
        )

        assert event.name == "test_pipeline"
        assert event.duration_ms == 5000.0
        assert event.node_results == node_results
        assert event.status == "completed"
        assert event.reason is None
        assert isinstance(event.timestamp, datetime)

    def test_pipeline_completed_defaults(self):
        """Test PipelineCompleted with default values."""
        event = PipelineCompleted(name="test", duration_ms=1000.0)

        assert event.name == "test"
        assert event.duration_ms == 1000.0
        assert event.node_results == {}
        assert event.status == "completed"
        assert isinstance(event.timestamp, datetime)

    def test_pipeline_completed_cancelled(self):
        """Test PipelineCompleted with cancelled status."""
        event = PipelineCompleted(
            name="test",
            duration_ms=500.0,
            node_results={"a": 1},
            status="cancelled",
            reason="timeout",
        )

        assert event.status == "cancelled"
        assert event.reason == "timeout"
        assert event.node_results == {"a": 1}


class TestPortEvents:
    """Test port-level event data classes."""

    def test_llm_generation_creation(self):
        """Test LLMGeneration event creation."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ]
        event = LLMGeneration(
            node_name="llm_node",
            duration_ms=2500.0,
            messages=messages,
            response="This is the LLM response",
        )

        assert event.node_name == "llm_node"
        assert event.messages == messages
        assert event.response == "This is the LLM response"
        assert event.duration_ms == 2500.0
        assert isinstance(event.timestamp, datetime)

    def test_tool_router_event_creation(self):
        """Test ToolRouterEvent event creation."""
        params = {"param1": "value1", "param2": 42}
        result = {"calculation": 84}
        event = ToolRouterEvent(
            node_name="tool_node",
            tool_name="calculator",
            params=params,
            result=result,
            duration_ms=100.5,
        )

        assert event.node_name == "tool_node"
        assert event.tool_name == "calculator"
        assert event.params == params
        assert event.result == result
        assert event.duration_ms == 100.5
        assert isinstance(event.timestamp, datetime)


class TestEventComparison:
    """Test event comparison and equality."""

    def test_events_are_dataclasses(self):
        """Test that events behave like dataclasses."""
        import time

        event1 = NodeStarted(name="test", wave_index=1)
        time.sleep(0.001)  # Ensure different timestamp
        event2 = NodeStarted(name="test", wave_index=1)
        event3 = NodeStarted(name="other", wave_index=1)

        # Same data with different timestamps should not be equal
        assert event1 != event2

        # Different data definitely not equal
        assert event1 != event3

    def test_event_fields_accessible(self):
        """Test that all event fields are accessible."""
        event = NodeCompleted(name="test", wave_index=1, result={"test": True}, duration_ms=500)

        # All fields should be accessible
        assert hasattr(event, "name")
        assert hasattr(event, "wave_index")
        assert hasattr(event, "result")
        assert hasattr(event, "duration_ms")
        assert hasattr(event, "timestamp")


class RecordingErrorHandler:
    """Error handler used to capture manager error calls in tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[Exception, dict[str, Any]]] = []

    def handle_error(self, error: Exception, context: dict[str, Any]) -> None:
        self.calls.append((error, context))


class TestDecoratorIntegrations:
    """Tests covering function-level decorator metadata usage."""

    @pytest.mark.asyncio
    async def test_observer_metadata_filters_and_respects_timeout(self):
        """Observer manager should use decorator metadata for filtering and timeouts."""

        events_seen: list[str] = []

        @observer(event_types={NodeStarted}, timeout=0.01, max_concurrency=2, id="observer-1")
        async def record_node_started(event) -> None:
            events_seen.append(type(event).__name__)

        error_handler = RecordingErrorHandler()
        manager = LocalObserverManager(error_handler=error_handler, batching_enabled=False)

        observer_id = manager.register(record_node_started)

        assert observer_id == "observer-1"
        assert manager._event_filters[observer_id] == {NodeStarted}
        assert manager._observer_timeouts[observer_id] == 0.01
        per_observer_semaphore = manager._observer_semaphores[observer_id]
        assert isinstance(per_observer_semaphore, asyncio.Semaphore)
        assert per_observer_semaphore._value == 2  # noqa: SLF001 - inspect concurrency budget

        await manager.notify(NodeFailed(name="fail", wave_index=1, error=RuntimeError("x")))
        await manager.notify(NodeStarted(name="start", wave_index=1))

        assert events_seen == ["NodeStarted"]

        @observer(event_types={NodeStarted}, timeout=0.01, id="slow-observer")
        async def slow_observer(event) -> None:
            await asyncio.sleep(0.05)

        slow_id = manager.register(slow_observer)
        assert slow_id == "slow-observer"

        await manager.notify(NodeStarted(name="slow", wave_index=3))

        assert events_seen == ["NodeStarted", "NodeStarted"]

        assert any(isinstance(error, TimeoutError) for error, _ in error_handler.calls)
        timeout_call = next(
            (call for call in error_handler.calls if isinstance(call[0], TimeoutError)),
            None,
        )
        assert timeout_call is not None
        error, context = timeout_call
        assert context["handler_name"] == "slow_observer"
        assert context["event_type"] == "NodeStarted"

        await manager.close()
