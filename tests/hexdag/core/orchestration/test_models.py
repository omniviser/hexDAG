"""Tests for orchestration models (CheckpointState, NodeExecutionContext)."""

from datetime import datetime

from hexdag.core.orchestration.models import CheckpointState, NodeExecutionContext


class TestCheckpointState:
    """Tests for CheckpointState model."""

    def test_checkpoint_state_creation(self):
        """Test creating a basic CheckpointState."""
        now = datetime.now()
        state = CheckpointState(
            run_id="test-run-123",
            dag_id="test-dag",
            graph_snapshot={"nodes": {"node1": {}}, "edges": []},
            initial_input="test input",
            node_results={"node1": "result1"},
            completed_node_ids=["node1"],
            created_at=now,
            updated_at=now,
        )

        assert state.run_id == "test-run-123"
        assert state.dag_id == "test-dag"
        assert state.initial_input == "test input"
        assert state.node_results == {"node1": "result1"}
        assert state.completed_node_ids == ["node1"]
        assert state.failed_node_ids == []  # Default
        assert state.metadata == {}  # Default

    def test_checkpoint_state_with_failures(self):
        """Test CheckpointState with failed nodes."""
        now = datetime.now()
        state = CheckpointState(
            run_id="test-run-123",
            dag_id="test-dag",
            graph_snapshot={"nodes": {}, "edges": []},
            initial_input="test",
            node_results={},
            completed_node_ids=["node1"],
            failed_node_ids=["node2", "node3"],
            created_at=now,
            updated_at=now,
        )

        assert state.failed_node_ids == ["node2", "node3"]

    def test_checkpoint_state_preserves_order(self):
        """Test that completed_node_ids preserves execution order."""
        now = datetime.now()
        state = CheckpointState(
            run_id="test-run",
            dag_id="test-dag",
            graph_snapshot={},
            initial_input=None,
            node_results={},
            completed_node_ids=["node3", "node1", "node2"],  # Order matters
            created_at=now,
            updated_at=now,
        )

        assert state.completed_node_ids == ["node3", "node1", "node2"]

    def test_checkpoint_state_with_metadata(self):
        """Test CheckpointState with custom metadata."""
        now = datetime.now()
        state = CheckpointState(
            run_id="test-run",
            dag_id="test-dag",
            graph_snapshot={},
            initial_input=None,
            node_results={},
            completed_node_ids=[],
            created_at=now,
            updated_at=now,
            metadata={"user": "test_user", "tags": ["prod", "v2"]},
        )

        assert state.metadata["user"] == "test_user"
        assert state.metadata["tags"] == ["prod", "v2"]

    def test_checkpoint_state_arbitrary_types(self):
        """Test that CheckpointState accepts arbitrary types for results."""
        now = datetime.now()

        class CustomResult:
            def __init__(self, value):
                self.value = value

        custom_obj = CustomResult(42)
        state = CheckpointState(
            run_id="test-run",
            dag_id="test-dag",
            graph_snapshot={},
            initial_input=custom_obj,
            node_results={"node1": custom_obj},
            completed_node_ids=[],
            created_at=now,
            updated_at=now,
        )

        assert state.initial_input.value == 42
        assert state.node_results["node1"].value == 42


class TestNodeExecutionContext:
    """Tests for NodeExecutionContext model."""

    def test_context_creation(self):
        """Test creating a basic NodeExecutionContext."""
        ctx = NodeExecutionContext(dag_id="test-dag")

        assert ctx.dag_id == "test-dag"
        assert ctx.node_id is None
        assert ctx.wave_index == 0
        assert ctx.attempt == 1
        assert ctx.metadata == {}

    def test_context_with_node(self):
        """Test with_node creates new context with updated node info."""
        ctx = NodeExecutionContext(dag_id="test-dag")
        new_ctx = ctx.with_node("node1", wave_index=2)

        # Original unchanged
        assert ctx.node_id is None
        assert ctx.wave_index == 0

        # New context updated
        assert new_ctx.dag_id == "test-dag"
        assert new_ctx.node_id == "node1"
        assert new_ctx.wave_index == 2
        assert new_ctx.attempt == 1

    def test_context_with_attempt(self):
        """Test with_attempt creates new context with updated attempt."""
        ctx = NodeExecutionContext(dag_id="test-dag", node_id="node1")
        new_ctx = ctx.with_attempt(3)

        # Original unchanged
        assert ctx.attempt == 1

        # New context updated
        assert new_ctx.dag_id == "test-dag"
        assert new_ctx.node_id == "node1"
        assert new_ctx.attempt == 3

    def test_context_metadata_isolation(self):
        """Test that metadata is copied, not shared between contexts."""
        ctx = NodeExecutionContext(dag_id="test-dag", metadata={"key": "value"})
        new_ctx = ctx.with_node("node1", wave_index=1)

        # Modify new context metadata
        new_ctx.metadata["new_key"] = "new_value"

        # Original should be unchanged
        assert "new_key" not in ctx.metadata
        assert ctx.metadata == {"key": "value"}

    def test_context_full_workflow(self):
        """Test context evolution through a typical workflow."""
        # Start with DAG-level context
        ctx = NodeExecutionContext(dag_id="research-pipeline")
        assert ctx.node_id is None

        # Execute first node
        ctx_node1 = ctx.with_node("fetch_data", wave_index=1)
        assert ctx_node1.node_id == "fetch_data"
        assert ctx_node1.wave_index == 1

        # Retry the node
        ctx_retry = ctx_node1.with_attempt(2)
        assert ctx_retry.node_id == "fetch_data"
        assert ctx_retry.attempt == 2

        # Execute next node in different wave
        ctx_node2 = ctx.with_node("process_data", wave_index=2)
        assert ctx_node2.node_id == "process_data"
        assert ctx_node2.wave_index == 2
        assert ctx_node2.attempt == 1  # Reset to 1
