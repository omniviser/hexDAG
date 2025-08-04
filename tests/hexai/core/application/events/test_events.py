"""Tests for core event classes."""

from datetime import datetime

from hexai.core.application.events import (
    LLMPromptGeneratedEvent,
    LLMResponseReceivedEvent,
    NodeCompletedEvent,
    NodeFailedEvent,
    NodeStartedEvent,
    PipelineCompletedEvent,
    PipelineStartedEvent,
    ValidationWarningEvent,
)


class MockEventManager:
    """Mock event manager for testing."""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or "test-session-123"
        self.events = []

    async def emit(self, event):
        """Mock emit that just stores events."""
        self.events.append(event)


class TestEventClasses:
    """Test cases for event data classes."""

    def test_pipeline_started_event(self):
        """Test PipelineStartedEvent creation and properties."""
        event = PipelineStartedEvent(
            pipeline_name="test_pipeline",
            total_waves=3,
            total_nodes=10,
        )

        assert event.pipeline_name == "test_pipeline"
        assert event.total_waves == 3
        assert event.total_nodes == 10
        assert isinstance(event.timestamp, datetime)

    def test_pipeline_completed_event(self):
        """Test PipelineCompletedEvent creation and properties."""
        node_results = {"node1": {"output": "test"}, "node2": {"output": "test2"}}
        event = PipelineCompletedEvent(
            pipeline_name="test_pipeline",
            total_execution_time=1.5,
            node_results=node_results,
        )

        assert event.pipeline_name == "test_pipeline"
        assert event.total_execution_time == 1.5
        assert event.node_results == node_results
        assert isinstance(event.timestamp, datetime)

    def test_node_started_event(self):
        """Test NodeStartedEvent creation and properties."""
        event = NodeStartedEvent(
            node_name="test_node",
            wave_index=1,
            dependencies=["dep1", "dep2"],
        )

        assert event.node_name == "test_node"
        assert event.wave_index == 1
        assert event.dependencies == ["dep1", "dep2"]
        assert isinstance(event.timestamp, datetime)

    def test_node_completed_event(self):
        """Test NodeCompletedEvent creation and properties."""
        result = {"output": "test_result"}
        event = NodeCompletedEvent(
            node_name="test_node",
            wave_index=1,
            result=result,
            execution_time=0.5,
        )

        assert event.node_name == "test_node"
        assert event.result == result
        assert event.execution_time == 0.5
        assert event.wave_index == 1
        assert isinstance(event.timestamp, datetime)

    def test_node_failed_event(self):
        """Test NodeFailedEvent creation and properties."""
        error = ValueError("Test error message")
        event = NodeFailedEvent(
            node_name="test_node",
            wave_index=1,
            error=error,
        )

        assert event.node_name == "test_node"
        assert event.error == error
        assert event.wave_index == 1
        assert isinstance(event.timestamp, datetime)

    def test_llm_prompt_generated_event(self):
        """Test LLMPromptGeneratedEvent creation and properties."""
        messages = [{"role": "user", "content": "test prompt"}]
        template_vars = {"variable": "value"}
        event = LLMPromptGeneratedEvent(
            node_name="llm_node",
            messages=messages,
            template="Test template: {{variable}}",
            template_vars=template_vars,
        )

        assert event.node_name == "llm_node"
        assert event.messages == messages
        assert event.template == "Test template: {{variable}}"
        assert event.template_vars == template_vars
        assert isinstance(event.timestamp, datetime)

    def test_llm_response_received_event(self):
        """Test LLMResponseReceivedEvent creation and properties."""
        response = "Test LLM response"
        event = LLMResponseReceivedEvent(
            node_name="llm_node",
            response=response,
        )

        assert event.node_name == "llm_node"
        assert event.response == response
        assert isinstance(event.timestamp, datetime)

    def test_validation_warning_event(self):
        """Test ValidationWarningEvent creation and properties."""
        warnings = ["Test validation warning"]
        event = ValidationWarningEvent(
            pipeline_name="test_pipeline",
            warnings=warnings,
        )

        assert event.pipeline_name == "test_pipeline"
        assert event.warnings == warnings
        assert isinstance(event.timestamp, datetime)

    def test_event_extra_fields(self):
        """Test that events include expected extra fields in their representation."""
        event = NodeStartedEvent(
            node_name="test_node",
            wave_index=1,
            dependencies=["dep1"],
        )

        # Check that extra fields method exists and returns expected data
        extra = event._extra_fields()
        assert "node_name" in extra
        assert "wave_index" in extra
        assert "dependencies" in extra
        assert extra["node_name"] == "test_node"

    def test_event_manager_integration(self):
        """Test event integration with MockEventManager."""
        event_manager = MockEventManager("test-session")

        # Test that we can create events and they have proper structure
        event = PipelineStartedEvent(
            pipeline_name="test_pipeline",
            total_waves=2,
            total_nodes=5,
        )

        # Mock emitting the event
        event_manager.events.append(event)

        assert len(event_manager.events) == 1
        assert event_manager.events[0].pipeline_name == "test_pipeline"
