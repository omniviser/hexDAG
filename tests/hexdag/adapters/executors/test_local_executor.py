"""Tests for LocalExecutor adapter."""

import asyncio

import pytest

from hexdag.adapters.executors import LocalExecutor
from hexdag.core.domain.dag import DirectedGraph, NodeSpec
from hexdag.core.orchestration.orchestrator import Orchestrator


# Test helper functions
async def async_add_one(x: int, **kwargs) -> int:
    """Async function that adds 1 to input."""
    await asyncio.sleep(0.01)
    return x + 1


async def async_multiply_two(x: int, **kwargs) -> int:
    """Async function that multiplies by 2."""
    await asyncio.sleep(0.01)
    return x * 2


async def async_combine(inputs: dict, **kwargs) -> int:
    """Async function that sums multiple inputs."""
    return sum(inputs.values())


class TestLocalExecutor:
    """Tests for LocalExecutor implementation."""

    def test_create_local_executor(self):
        """Test creating a LocalExecutor instance."""
        executor = LocalExecutor(max_concurrent_nodes=5)

        assert executor.config.max_concurrent_nodes == 5
        assert executor.config.strict_validation is False
        assert executor.config.default_node_timeout is None

    def test_local_executor_capabilities(self):
        """Test that LocalExecutor reports correct capabilities."""
        executor = LocalExecutor(max_concurrent_nodes=10)

        caps = executor.get_capabilities()

        assert caps.supports_timeout is True
        assert caps.supports_cancellation is True
        assert caps.max_concurrent == 10
        assert caps.is_distributed is False
        assert caps.requires_serialization is False

    @pytest.mark.asyncio
    async def test_local_executor_lifecycle(self):
        """Test LocalExecutor setup and cleanup."""
        executor = LocalExecutor()

        assert executor._initialized is False

        await executor.asetup()
        assert executor._initialized is True

        await executor.aclose()
        assert executor._initialized is False

    @pytest.mark.asyncio
    async def test_orchestrator_with_local_executor_simple(self):
        """Test simple sequential execution with LocalExecutor."""
        # Create executor
        executor = LocalExecutor(max_concurrent_nodes=5)

        # Build DAG
        graph = DirectedGraph()
        graph.add(NodeSpec("add", async_add_one))

        # Create orchestrator with LocalExecutor
        orchestrator = Orchestrator(executor=executor)

        # Execute
        results = await orchestrator.run(graph, 5)

        # Verify
        assert "add" in results
        assert results["add"] == 6

    @pytest.mark.asyncio
    async def test_orchestrator_with_local_executor_parallel(self):
        """Test parallel execution with LocalExecutor."""
        executor = LocalExecutor(max_concurrent_nodes=5)

        # Build DAG with parallel nodes
        graph = DirectedGraph()
        graph.add(NodeSpec("add", async_add_one))
        graph.add(NodeSpec("mult", async_multiply_two))

        orchestrator = Orchestrator(executor=executor)

        # Execute
        results = await orchestrator.run(graph, 5)

        # Verify both nodes executed
        assert "add" in results
        assert "mult" in results
        assert results["add"] == 6  # 5 + 1
        assert results["mult"] == 10  # 5 * 2

    @pytest.mark.asyncio
    async def test_orchestrator_with_local_executor_dependencies(self):
        """Test DAG with dependencies using LocalExecutor."""
        executor = LocalExecutor(max_concurrent_nodes=5)

        # Build DAG with dependencies
        graph = DirectedGraph()
        graph.add(NodeSpec("add", async_add_one))
        graph.add(NodeSpec("mult", async_multiply_two).after("add"))

        orchestrator = Orchestrator(executor=executor)

        # Execute
        results = await orchestrator.run(graph, 5)

        # Verify sequential execution
        assert "add" in results
        assert "mult" in results
        assert results["add"] == 6  # 5 + 1
        assert results["mult"] == 12  # 6 * 2

    @pytest.mark.asyncio
    async def test_orchestrator_with_local_executor_complex_dag(self):
        """Test complex DAG execution with LocalExecutor."""
        executor = LocalExecutor(max_concurrent_nodes=10)

        # Build complex DAG: two parallel paths that merge
        graph = DirectedGraph()
        graph.add(NodeSpec("add", async_add_one))  # Wave 1
        graph.add(NodeSpec("mult", async_multiply_two))  # Wave 1
        graph.add(NodeSpec("combine", async_combine).after("add", "mult"))  # Wave 2

        orchestrator = Orchestrator(executor=executor)

        # Execute
        results = await orchestrator.run(graph, 5)

        # Verify
        assert "add" in results
        assert "mult" in results
        assert "combine" in results
        assert results["add"] == 6  # 5 + 1
        assert results["mult"] == 10  # 5 * 2
        assert results["combine"] == 16  # 6 + 10

    @pytest.mark.asyncio
    async def test_local_executor_equivalence_to_default(self):
        """Test that LocalExecutor produces same results as default execution."""
        # Build same DAG
        graph = DirectedGraph()
        graph.add(NodeSpec("add", async_add_one))
        graph.add(NodeSpec("mult", async_multiply_two).after("add"))

        # Execute with default (no executor)
        orchestrator_default = Orchestrator()
        results_default = await orchestrator_default.run(graph, 10)

        # Execute with LocalExecutor
        executor = LocalExecutor()
        orchestrator_local = Orchestrator(executor=executor)
        results_local = await orchestrator_local.run(graph, 10)

        # Results should be identical
        assert results_default == results_local
        assert results_default["add"] == 11
        assert results_default["mult"] == 22
        assert results_local["add"] == 11
        assert results_local["mult"] == 22

    @pytest.mark.asyncio
    async def test_local_executor_with_validation(self):
        """Test LocalExecutor with validation enabled."""
        from pydantic import BaseModel

        class InputModel(BaseModel):
            value: int

        class OutputModel(BaseModel):
            result: int

        async def validated_add(input_data: InputModel, **kwargs) -> OutputModel:
            return OutputModel(result=input_data.value + 1)

        executor = LocalExecutor(strict_validation=True)

        graph = DirectedGraph()
        graph.add(
            NodeSpec(
                "validated",
                validated_add,
                in_model=InputModel,
                out_model=OutputModel,
            )
        )

        orchestrator = Orchestrator(executor=executor)

        # Execute with dict (will be validated)
        results = await orchestrator.run(graph, {"value": 5})

        assert "validated" in results
        assert results["validated"].result == 6

    @pytest.mark.asyncio
    async def test_local_executor_error_handling(self):
        """Test that LocalExecutor handles errors properly."""

        async def failing_node(x: int, **kwargs) -> int:
            raise ValueError("Intentional failure")

        executor = LocalExecutor()

        graph = DirectedGraph()
        graph.add(NodeSpec("fail", failing_node))

        orchestrator = Orchestrator(executor=executor)

        # Should raise OrchestratorError
        from hexdag.core.orchestration.orchestrator import OrchestratorError

        with pytest.raises(OrchestratorError, match="failed"):
            await orchestrator.run(graph, 0)

    @pytest.mark.asyncio
    async def test_local_executor_with_timeout(self):
        """Test LocalExecutor with node timeout."""

        async def slow_node(x: int, **kwargs) -> int:
            await asyncio.sleep(2.0)
            return x + 1

        executor = LocalExecutor(default_node_timeout=0.1)

        graph = DirectedGraph()
        graph.add(NodeSpec("slow", slow_node))

        orchestrator = Orchestrator(executor=executor)

        # Should timeout - will raise TimeoutError or wrapped in OrchestratorError
        with pytest.raises((TimeoutError, Exception)):
            await orchestrator.run(graph, 0)
