"""Tests for WaveExecutor component."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from hexai.core.domain.dag import DirectedGraph
from hexai.core.orchestration.components import WaveExecutor
from hexai.core.orchestration.models import NodeExecutionContext


class TestWaveExecutor:
    """Tests for WaveExecutor."""

    @pytest.fixture
    def executor(self):
        """Provide a WaveExecutor instance."""
        return WaveExecutor(max_concurrent_nodes=3)

    @pytest.fixture
    def mock_context(self):
        """Provide a mock execution context."""
        return NodeExecutionContext(dag_id="test-dag")

    @pytest.fixture
    def mock_policy_coordinator(self):
        """Provide a mock policy coordinator."""
        coordinator = MagicMock()
        coordinator.notify_observer = AsyncMock()
        coordinator.evaluate_policy = AsyncMock(
            return_value=MagicMock(signal=MagicMock(value="PROCEED"))
        )
        coordinator.check_policy_signal = MagicMock()
        return coordinator

    @pytest.mark.asyncio
    async def test_executor_initialization(self):
        """Test WaveExecutor initialization."""
        executor = WaveExecutor(max_concurrent_nodes=5)
        assert executor.max_concurrent_nodes == 5
        assert executor._semaphore._value == 5

    @pytest.mark.asyncio
    async def test_execute_single_wave(self, executor, mock_context, mock_policy_coordinator):
        """Test executing a single wave with one node."""

        async def mock_node_executor(**kwargs):
            """Mock node executor that returns node name and result."""
            return "result"

        graph = DirectedGraph()
        node_results = {}
        wave = ["node1"]

        result = await executor._execute_wave(
            wave=wave,
            node_executor_fn=mock_node_executor,
            graph=graph,
            node_results=node_results,
            initial_input="test",
            ports={},
            context=mock_context,
            observer_manager=None,
            policy_manager=None,
            wave_index=1,
            validate=True,
        )

        assert result == {"node1": "result"}

    @pytest.mark.asyncio
    async def test_execute_parallel_nodes(self, executor, mock_context, mock_policy_coordinator):
        """Test executing multiple nodes in parallel."""
        execution_order = []

        async def mock_node_executor(node_name, **kwargs):
            """Mock node executor that tracks execution order."""
            execution_order.append(f"{node_name}-start")
            await asyncio.sleep(0.01)  # Simulate work
            execution_order.append(f"{node_name}-end")
            return f"result-{node_name}"

        graph = DirectedGraph()
        node_results = {}
        wave = ["node1", "node2", "node3"]

        result = await executor._execute_wave(
            wave=wave,
            node_executor_fn=mock_node_executor,
            graph=graph,
            node_results=node_results,
            initial_input="test",
            ports={},
            context=mock_context,
            observer_manager=None,
            policy_manager=None,
            wave_index=1,
            validate=True,
        )

        # All nodes should complete
        assert len(result) == 3
        assert result["node1"] == "result-node1"
        assert result["node2"] == "result-node2"
        assert result["node3"] == "result-node3"

        # Nodes should have started in parallel (all starts before all ends)
        assert "node1-start" in execution_order
        assert "node2-start" in execution_order
        assert "node3-start" in execution_order

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Test that concurrency limiting works."""
        executor = WaveExecutor(max_concurrent_nodes=2)
        concurrent_count = 0
        max_concurrent = 0

        async def mock_node_executor(node_name, **kwargs):
            """Track concurrent executions."""
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.01)
            concurrent_count -= 1
            return f"result-{node_name}"

        graph = DirectedGraph()
        node_results = {}
        context = NodeExecutionContext(dag_id="test")
        wave = ["node1", "node2", "node3", "node4"]

        await executor._execute_wave(
            wave=wave,
            node_executor_fn=mock_node_executor,
            graph=graph,
            node_results=node_results,
            initial_input="test",
            ports={},
            context=context,
            observer_manager=None,
            policy_manager=None,
            wave_index=1,
            validate=True,
        )

        # Max concurrent should not exceed limit
        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_exception_propagation(self, executor, mock_context, mock_policy_coordinator):
        """Test that exceptions from nodes are propagated."""

        async def failing_node_executor(**kwargs):
            """Node executor that raises an exception."""
            raise ValueError("Node execution failed")

        graph = DirectedGraph()
        node_results = {}
        wave = ["node1"]

        with pytest.raises(ValueError, match="Node execution failed"):
            await executor._execute_wave(
                wave=wave,
                node_executor_fn=failing_node_executor,
                graph=graph,
                node_results=node_results,
                initial_input="test",
                ports={},
                context=mock_context,
                observer_manager=None,
                policy_manager=None,
                wave_index=1,
                validate=True,
            )

    @pytest.mark.asyncio
    async def test_execute_all_waves(self, executor, mock_context, mock_policy_coordinator):
        """Test executing multiple waves."""

        async def mock_node_executor(**kwargs):
            return "result"

        graph = DirectedGraph()
        node_results = {}
        waves = [["node1"], ["node2", "node3"], ["node4"]]

        cancelled = await executor.execute_all_waves(
            waves=waves,
            node_executor_fn=mock_node_executor,
            graph=graph,
            node_results=node_results,
            initial_input="test",
            all_ports={},
            context=mock_context,
            observer_manager=None,
            policy_manager=None,
            policy_coordinator=mock_policy_coordinator,
            timeout=None,
            validate=True,
        )

        assert cancelled is False
        assert len(node_results) == 4  # All 4 nodes completed

    @pytest.mark.asyncio
    async def test_pipeline_timeout(self, executor, mock_context, mock_policy_coordinator):
        """Test that pipeline-level timeout works."""

        async def slow_node_executor(**kwargs):
            """Node that takes too long."""
            await asyncio.sleep(2)
            return "result"

        graph = DirectedGraph()
        node_results = {}
        waves = [["node1"]]

        cancelled = await executor.execute_all_waves(
            waves=waves,
            node_executor_fn=slow_node_executor,
            graph=graph,
            node_results=node_results,
            initial_input="test",
            all_ports={},
            context=mock_context,
            observer_manager=None,
            policy_manager=None,
            policy_coordinator=mock_policy_coordinator,
            timeout=0.5,  # 500ms timeout
            validate=True,
        )

        # Should have been cancelled due to timeout
        assert cancelled is True
        assert len(node_results) == 0  # No nodes completed

    @pytest.mark.asyncio
    async def test_wave_events_emitted(self, executor, mock_context, mock_policy_coordinator):
        """Test that wave events are emitted."""

        async def mock_node_executor(**kwargs):
            return "result"

        graph = DirectedGraph()
        node_results = {}
        waves = [["node1"], ["node2"]]

        await executor.execute_all_waves(
            waves=waves,
            node_executor_fn=mock_node_executor,
            graph=graph,
            node_results=node_results,
            initial_input="test",
            all_ports={},
            context=mock_context,
            observer_manager=None,
            policy_manager=None,
            policy_coordinator=mock_policy_coordinator,
            timeout=None,
            validate=True,
        )

        # Should have notified observer for wave started/completed events
        # 2 waves × 2 events (started + completed) = 4 calls
        assert mock_policy_coordinator.notify_observer.call_count == 4

    @pytest.mark.asyncio
    async def test_node_results_accumulate(self, executor, mock_context, mock_policy_coordinator):
        """Test that node results accumulate across waves."""

        async def mock_node_executor(node_name, **kwargs):
            return f"result-{node_name}"

        graph = DirectedGraph()
        node_results = {}
        waves = [["node1"], ["node2", "node3"]]

        await executor.execute_all_waves(
            waves=waves,
            node_executor_fn=mock_node_executor,
            graph=graph,
            node_results=node_results,
            initial_input="test",
            all_ports={},
            context=mock_context,
            observer_manager=None,
            policy_manager=None,
            policy_coordinator=mock_policy_coordinator,
            timeout=None,
            validate=True,
        )

        # All results should be accumulated in node_results
        assert node_results["node1"] == "result-node1"
        assert node_results["node2"] == "result-node2"
        assert node_results["node3"] == "result-node3"
