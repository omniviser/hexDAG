"""Tests for consolidated event classes."""

from datetime import datetime

from hexai.core.application.events import (
    ExecutionEvent,
    ExecutionLevel,
    ExecutionPhase,
    HookEvent,
    LLMEvent,
    MetaEvent,
)


class MockEventManager:
    """Mock event manager for testing."""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or "test-session-123"
        self.events = []

    async def emit(self, event):
        """Mock emit that just stores events."""
        self.events.append(event)


class TestConsolidatedEvents:
    """Test cases for consolidated event data classes."""

    def test_execution_event_dag_started(self):
        """Test ExecutionEvent creation for DAG started."""
        event = ExecutionEvent(
            level=ExecutionLevel.DAG,
            phase=ExecutionPhase.STARTED,
            name="test_pipeline",
            total_waves=3,
            total_nodes=10,
        )

        assert event.level == ExecutionLevel.DAG
        assert event.phase == ExecutionPhase.STARTED
        assert event.name == "test_pipeline"
        assert event.total_waves == 3
        assert event.total_nodes == 10
        assert isinstance(event.timestamp, datetime)

    def test_execution_event_dag_completed(self):
        """Test ExecutionEvent creation for DAG completed."""
        node_results = {"node1": {"output": "test"}, "node2": {"output": "test2"}}
        event = ExecutionEvent(
            level=ExecutionLevel.DAG,
            phase=ExecutionPhase.COMPLETED,
            name="test_pipeline",
            execution_time_ms=1500,
            node_results=node_results,
        )

        assert event.level == ExecutionLevel.DAG
        assert event.phase == ExecutionPhase.COMPLETED
        assert event.name == "test_pipeline"
        assert event.execution_time_ms == 1500
        assert event.node_results == node_results
        assert isinstance(event.timestamp, datetime)

    def test_execution_event_node_started(self):
        """Test ExecutionEvent creation for node started."""
        event = ExecutionEvent(
            level=ExecutionLevel.NODE,
            phase=ExecutionPhase.STARTED,
            name="test_node",
            wave_index=1,
            dependencies=["dep1", "dep2"],
        )

        assert event.level == ExecutionLevel.NODE
        assert event.phase == ExecutionPhase.STARTED
        assert event.name == "test_node"
        assert event.wave_index == 1
        assert event.dependencies == ["dep1", "dep2"]
        assert isinstance(event.timestamp, datetime)

    def test_execution_event_node_completed(self):
        """Test ExecutionEvent creation for node completed."""
        result = {"output": "test_result"}
        event = ExecutionEvent(
            level=ExecutionLevel.NODE,
            phase=ExecutionPhase.COMPLETED,
            name="test_node",
            wave_index=1,
            result=result,
            execution_time_ms=500,
        )

        assert event.level == ExecutionLevel.NODE
        assert event.phase == ExecutionPhase.COMPLETED
        assert event.name == "test_node"
        assert event.result == result
        assert event.execution_time_ms == 500
        assert event.wave_index == 1
        assert isinstance(event.timestamp, datetime)

    def test_execution_event_node_failed(self):
        """Test ExecutionEvent creation for node failed."""
        error = ValueError("Test error message")
        event = ExecutionEvent(
            level=ExecutionLevel.NODE,
            phase=ExecutionPhase.FAILED,
            name="test_node",
            wave_index=1,
            error=error,
        )

        assert event.level == ExecutionLevel.NODE
        assert event.phase == ExecutionPhase.FAILED
        assert event.name == "test_node"
        assert event.error == error
        assert event.wave_index == 1
        assert isinstance(event.timestamp, datetime)

    def test_execution_event_wave(self):
        """Test ExecutionEvent creation for wave events."""
        event = ExecutionEvent(
            level=ExecutionLevel.WAVE,
            phase=ExecutionPhase.STARTED,
            name="wave_1",
            wave_index=1,
            nodes=["node1", "node2", "node3"],
        )

        assert event.level == ExecutionLevel.WAVE
        assert event.phase == ExecutionPhase.STARTED
        assert event.name == "wave_1"
        assert event.wave_index == 1
        assert event.nodes == ["node1", "node2", "node3"]

    def test_llm_event_prompt(self):
        """Test LLMEvent creation for prompts."""
        messages = [{"role": "user", "content": "Hello"}]
        event = LLMEvent(
            event_class="llm",
            action="prompt",
            node_name="test_node",
            tool_name="gpt-4",
            input_data=messages,
            messages=messages,
            template_vars={"user": "test"},
        )

        assert event.event_class == "llm"
        assert event.action == "prompt"
        assert event.node_name == "test_node"
        assert event.tool_name == "gpt-4"
        assert event.messages == messages
        assert event.template_vars == {"user": "test"}
        assert isinstance(event.timestamp, datetime)

    def test_llm_event_response(self):
        """Test LLMEvent creation for responses."""
        event = LLMEvent(
            event_class="llm",
            action="response",
            node_name="test_node",
            tool_name="gpt-4",
            output_data="Generated response",
        )

        assert event.event_class == "llm"
        assert event.action == "response"
        assert event.node_name == "test_node"
        assert event.output_data == "Generated response"
        assert isinstance(event.timestamp, datetime)

    def test_llm_event_tool_called(self):
        """Test LLMEvent for tool calls."""
        tool_params = {"query": "test search"}
        event = LLMEvent(
            event_class="tool",
            action="called",
            node_name="test_node",
            tool_name="search_tool",
            input_data=tool_params,
        )

        assert event.event_class == "tool"
        assert event.action == "called"
        assert event.node_name == "test_node"
        assert event.tool_name == "search_tool"
        assert event.input_data == tool_params

    def test_llm_event_tool_completed(self):
        """Test LLMEvent for tool completion."""
        event = LLMEvent(
            event_class="tool",
            action="completed",
            node_name="test_node",
            tool_name="search_tool",
            output_data=["result1", "result2"],
            execution_time_ms=250,
        )

        assert event.event_class == "tool"
        assert event.action == "completed"
        assert event.tool_name == "search_tool"
        assert event.output_data == ["result1", "result2"]
        assert event.execution_time_ms == 250

    def test_hook_event(self):
        """Test HookEvent creation."""
        event = HookEvent(
            hook_type="pre_node",
            hook_name="validation_hook",
            target_name="test_node",
            metadata={"input": "test"},
        )

        assert event.hook_type == "pre_node"
        assert event.hook_name == "validation_hook"
        assert event.target_name == "test_node"
        assert event.metadata == {"input": "test"}
        assert isinstance(event.timestamp, datetime)

    def test_hook_event_mid_node(self):
        """Test HookEvent for mid-node LLM retries."""
        event = HookEvent(
            hook_type="mid_node",
            hook_name="retry_hook",
            target_name="agent_node",
            metadata={"retry_count": 2, "reason": "Invalid response format"},
        )

        assert event.hook_type == "mid_node"
        assert event.target_name == "agent_node"
        assert event.metadata["retry_count"] == 2

    def test_meta_event(self):
        """Test MetaEvent creation."""
        event = MetaEvent(
            category="validation",
            pipeline_name="test_pipeline",
            warnings=["Warning 1", "Warning 2"],
        )

        assert event.category == "validation"
        assert event.pipeline_name == "test_pipeline"
        assert event.warnings == ["Warning 1", "Warning 2"]
        assert isinstance(event.timestamp, datetime)

    def test_meta_event_build(self):
        """Test MetaEvent for build events."""
        event = MetaEvent(
            category="build",
            pipeline_name="test_pipeline",
            message="Building pipeline nodes",
            details={
                "nodes_added": 10,
                "waves_created": 3,
            },
        )

        assert event.category == "build"
        assert event.pipeline_name == "test_pipeline"
        assert event.message == "Building pipeline nodes"
        assert event.details["nodes_added"] == 10

    def test_event_extra_fields(self):
        """Test that events include expected extra fields in their representation."""
        event = ExecutionEvent(
            level=ExecutionLevel.NODE,
            phase=ExecutionPhase.STARTED,
            name="test_node",
            wave_index=1,
            dependencies=["dep1"],
        )

        extra_fields = event._extra_fields()
        assert extra_fields["level"] == "node"  # String value of the enum
        assert extra_fields["phase"] == "started"  # String value of the enum
        assert extra_fields["name"] == "test_node"
        assert extra_fields["wave_index"] == 1
        # dependencies is not included in _extra_fields()

    def test_execution_event_optional_fields(self):
        """Test ExecutionEvent with optional fields."""
        metadata = {"custom": "data"}
        event = ExecutionEvent(
            level=ExecutionLevel.NODE,
            phase=ExecutionPhase.COMPLETED,
            name="test_node",
            wave_index=1,
            result={"output": "test"},
            execution_time_ms=100,
            metadata=metadata,
        )

        assert event.metadata == metadata
        assert event.execution_time_ms == 100
        assert event.result == {"output": "test"}
