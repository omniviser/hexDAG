"""Tests for CheckpointManager component."""

from datetime import datetime

import pytest

from hexdag.builtin.adapters.memory import InMemoryMemory
from hexdag.core.domain.dag import DirectedGraph, NodeSpec
from hexdag.core.orchestration.components import CheckpointManager
from hexdag.core.orchestration.models import CheckpointState


class TestCheckpointManager:
    """Tests for CheckpointManager with Memory Port."""

    @pytest.fixture
    def storage(self):
        """Provide in-memory storage for testing."""
        return InMemoryMemory()

    @pytest.fixture
    def manager(self, storage):
        """Provide a CheckpointManager instance."""
        return CheckpointManager(storage=storage)

    @pytest.mark.asyncio
    async def test_save_and_load_checkpoint(self, manager):
        """Test saving and loading a checkpoint."""
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

        # Save checkpoint
        await manager.save(state)

        # Load checkpoint
        loaded = await manager.load("test-run-123")

        assert loaded is not None
        assert loaded.run_id == "test-run-123"
        assert loaded.dag_id == "test-dag"
        assert loaded.initial_input == "test input"
        assert loaded.node_results == {"node1": "result1"}
        assert loaded.completed_node_ids == ["node1"]

    @pytest.mark.asyncio
    async def test_load_nonexistent_checkpoint(self, manager):
        """Test loading a checkpoint that doesn't exist."""
        loaded = await manager.load("nonexistent-run-id")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_checkpoint_update(self, manager):
        """Test updating an existing checkpoint."""
        now = datetime.now()
        state = CheckpointState(
            run_id="test-run-123",
            dag_id="test-dag",
            graph_snapshot={},
            initial_input="test",
            node_results={"node1": "result1"},
            completed_node_ids=["node1"],
            created_at=now,
            updated_at=now,
        )

        # Save initial checkpoint
        await manager.save(state)

        # Update checkpoint with more completed nodes
        updated_state = CheckpointState(
            run_id="test-run-123",
            dag_id="test-dag",
            graph_snapshot={},
            initial_input="test",
            node_results={"node1": "result1", "node2": "result2"},
            completed_node_ids=["node1", "node2"],
            created_at=now,
            updated_at=datetime.now(),
        )
        await manager.save(updated_state)

        # Load and verify
        loaded = await manager.load("test-run-123")
        assert len(loaded.completed_node_ids) == 2
        assert "node2" in loaded.completed_node_ids
        assert loaded.node_results["node2"] == "result2"

    @pytest.mark.asyncio
    async def test_filter_completed_nodes(self, manager):
        """Test filtering completed nodes from a graph."""

        def node_fn(x):
            return x

        # Create a simple graph
        graph = DirectedGraph()
        graph.add(NodeSpec(name="node1", fn=node_fn))
        graph.add(NodeSpec(name="node2", fn=node_fn, deps={"node1"}))
        graph.add(NodeSpec(name="node3", fn=node_fn, deps={"node1"}))
        graph.add(NodeSpec(name="node4", fn=node_fn, deps={"node2", "node3"}))

        # Mark node1 and node2 as completed
        completed = {"node1", "node2"}

        # Filter graph
        filtered = manager.filter_completed(graph, completed)

        # Should only have node3 and node4 remaining
        assert len(filtered.nodes) == 2
        assert "node3" in filtered.nodes
        assert "node4" in filtered.nodes
        assert "node1" not in filtered.nodes
        assert "node2" not in filtered.nodes

    @pytest.mark.asyncio
    async def test_filter_completed_preserves_dependencies(self, manager):
        """Test that filtering preserves correct dependencies."""

        def node_fn(x):
            return x

        graph = DirectedGraph()
        graph.add(NodeSpec(name="node1", fn=node_fn))
        graph.add(NodeSpec(name="node2", fn=node_fn, deps={"node1"}))
        graph.add(NodeSpec(name="node3", fn=node_fn, deps={"node2"}))

        # Mark node1 as completed
        completed = {"node1"}

        filtered = manager.filter_completed(graph, completed)

        # node2 should have empty deps (its dependency was completed)
        assert filtered.nodes["node2"].deps == set()
        # node3 should still depend on node2
        assert filtered.nodes["node3"].deps == {"node2"}

    @pytest.mark.asyncio
    async def test_checkpoint_with_failed_nodes(self, manager):
        """Test checkpoint with failed nodes tracking."""
        now = datetime.now()
        state = CheckpointState(
            run_id="test-run-123",
            dag_id="test-dag",
            graph_snapshot={},
            initial_input="test",
            node_results={"node1": "result1"},
            completed_node_ids=["node1"],
            failed_node_ids=["node2"],
            created_at=now,
            updated_at=now,
        )

        await manager.save(state)
        loaded = await manager.load("test-run-123")

        assert loaded.failed_node_ids == ["node2"]

    @pytest.mark.asyncio
    async def test_auto_checkpoint_flag(self, storage):
        """Test that auto_checkpoint flag is respected."""
        manager_auto = CheckpointManager(storage=storage, auto_checkpoint=True)
        assert manager_auto.auto_checkpoint is True

        manager_manual = CheckpointManager(storage=storage, auto_checkpoint=False)
        assert manager_manual.auto_checkpoint is False

    @pytest.mark.asyncio
    async def test_key_prefix(self, storage):
        """Test custom key prefix."""
        manager = CheckpointManager(storage=storage, key_prefix="custom:")

        now = datetime.now()
        state = CheckpointState(
            run_id="test-123",
            dag_id="test-dag",
            graph_snapshot={},
            initial_input="test",
            node_results={},
            completed_node_ids=[],
            created_at=now,
            updated_at=now,
        )

        await manager.save(state)

        # Verify key was stored with custom prefix
        key = manager._make_key("test-123")
        assert key == "custom:test-123"

        # Verify we can load it back
        loaded = await manager.load("test-123")
        assert loaded is not None
        assert loaded.run_id == "test-123"
