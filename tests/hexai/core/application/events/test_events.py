"""Tests for event data classes."""

from datetime import datetime

from hexai.core.application.events import (
    LLMPromptSent,
    LLMResponseReceived,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    PipelineCompleted,
    PipelineStarted,
    ToolCalled,
    ToolCompleted,
    WaveCompleted,
    WaveStarted,
)


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
        assert event.dependencies == []
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

    def test_wave_started_creation(self):
        """Test WaveStarted event creation."""
        nodes = ["node1", "node2", "node3"]
        event = WaveStarted(wave_index=1, nodes=nodes)

        assert event.wave_index == 1
        assert event.nodes == nodes
        assert isinstance(event.timestamp, datetime)

    def test_wave_completed_creation(self):
        """Test WaveCompleted event creation."""
        event = WaveCompleted(wave_index=2, duration_ms=3000.0)

        assert event.wave_index == 2
        assert event.duration_ms == 3000.0
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
        assert isinstance(event.timestamp, datetime)

    def test_pipeline_completed_defaults(self):
        """Test PipelineCompleted with default values."""
        event = PipelineCompleted(name="test", duration_ms=1000.0)

        assert event.name == "test"
        assert event.duration_ms == 1000.0
        assert event.node_results == {}
        assert isinstance(event.timestamp, datetime)


class TestLLMEvents:
    """Test LLM-related event data classes."""

    def test_llm_prompt_sent_creation(self):
        """Test LLMPromptSent event creation."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ]
        event = LLMPromptSent(node_name="llm_node", messages=messages)

        assert event.node_name == "llm_node"
        assert event.messages == messages
        assert isinstance(event.timestamp, datetime)

    def test_llm_response_received_creation(self):
        """Test LLMResponseReceived event creation."""
        event = LLMResponseReceived(
            node_name="llm_node", response="This is the LLM response", duration_ms=2500.0
        )

        assert event.node_name == "llm_node"
        assert event.response == "This is the LLM response"
        assert event.duration_ms == 2500.0
        assert isinstance(event.timestamp, datetime)


class TestToolEvents:
    """Test tool-related event data classes."""

    def test_tool_called_creation(self):
        """Test ToolCalled event creation."""
        params = {"param1": "value1", "param2": 42}
        event = ToolCalled(node_name="tool_node", tool_name="calculator", params=params)

        assert event.node_name == "tool_node"
        assert event.tool_name == "calculator"
        assert event.params == params
        assert isinstance(event.timestamp, datetime)

    def test_tool_completed_creation(self):
        """Test ToolCompleted event creation."""
        result = {"calculation": 84}
        event = ToolCompleted(
            node_name="tool_node", tool_name="calculator", result=result, duration_ms=100.5
        )

        assert event.node_name == "tool_node"
        assert event.tool_name == "calculator"
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
