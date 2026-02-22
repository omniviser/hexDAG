"""Tests for Orchestrator integration with ExecutorPort."""

import asyncio

import pytest

from hexdag.kernel.domain.dag import DirectedGraph, NodeSpec
from hexdag.kernel.orchestration.orchestrator import Orchestrator
from hexdag.kernel.ports.executor import (
    ExecutionResult,
    ExecutionTask,
    ExecutorPort,
)
from hexdag.kernel.ports_builder import PortsBuilder


# Test helper functions
async def async_add_one(x: int, **kwargs) -> int:
    """Async function that adds 1 to input."""
    await asyncio.sleep(0.01)  # Simulate async work
    return x + 1


async def async_multiply_two(x: int, **kwargs) -> int:
    """Async function that multiplies by 2."""
    await asyncio.sleep(0.01)
    return x * 2


async def async_combine(inputs: dict, **kwargs) -> int:
    """Async function that sums multiple inputs."""
    return sum(inputs.values())


class MockExecutor:
    """Mock executor that tracks execution for testing."""

    def __init__(self):
        self.setup_called = False
        self.close_called = False
        self.executed_tasks: list[ExecutionTask] = []
        self.node_functions = {}

    def register_node(self, name: str, fn):
        """Register a node function for execution."""
        self.node_functions[name] = fn

    async def aexecute_node(self, task: ExecutionTask) -> ExecutionResult:
        """Execute a single node (mock implementation)."""
        self.executed_tasks.append(task)

        # Look up the function for this node
        if task.node_name not in self.node_functions:
            return ExecutionResult(
                node_name=task.node_name,
                status="failed",
                error=f"Node '{task.node_name}' not found",
                error_type="KeyError",
            )

        try:
            fn = self.node_functions[task.node_name]
            # Execute the actual function
            if asyncio.iscoroutinefunction(fn):
                output = await fn(task.node_input, **task.params)
            else:
                output = fn(task.node_input, **task.params)

            return ExecutionResult(
                node_name=task.node_name,
                output=output,
                duration_ms=10.0,
                status="success",
            )
        except Exception as e:
            return ExecutionResult(
                node_name=task.node_name,
                status="failed",
                error=str(e),
                error_type=type(e).__name__,
            )

    async def aexecute_wave(self, tasks: list[ExecutionTask]) -> dict[str, ExecutionResult]:
        """Execute multiple nodes concurrently (mock implementation)."""
        # Execute all tasks concurrently
        results_list = await asyncio.gather(
            *[self.aexecute_node(task) for task in tasks],
            return_exceptions=True,
        )

        results = {}
        for result in results_list:
            if isinstance(result, ExecutionResult):
                results[result.node_name] = result
            elif isinstance(result, Exception):
                # Should not happen in mock, but handle gracefully
                raise result

        return results

    async def asetup(self) -> None:
        """Initialize executor resources."""
        self.setup_called = True

    async def aclose(self) -> None:
        """Cleanup executor resources."""
        self.close_called = True


class TestOrchestratorWithExecutor:
    """Tests for Orchestrator with custom executor implementation."""

    @pytest.mark.asyncio
    async def test_orchestrator_with_none_executor_uses_default(self):
        """Test that orchestrator with executor=None uses default execution."""
        # Build a simple DAG
        graph = DirectedGraph()
        graph.add(NodeSpec("add", async_add_one))

        # Create orchestrator without executor (default behavior)
        orchestrator = Orchestrator(executor=None)

        # Execute
        results = await orchestrator.run(graph, 5)

        # Verify default execution worked
        assert "add" in results
        assert results["add"] == 6

    @pytest.mark.asyncio
    async def test_orchestrator_with_mock_executor(self):
        """Test orchestrator execution with mock executor."""
        # Note: This test demonstrates the interface but we haven't implemented
        # the delegation logic yet (that's Phase 2). For now, this shows that
        # the executor parameter is accepted and lifecycle methods are called.

        # Create mock executor
        executor = MockExecutor()
        executor.register_node("add", async_add_one)

        # Build a simple DAG
        graph = DirectedGraph()
        graph.add(NodeSpec("add", async_add_one))

        # Create orchestrator with custom executor
        orchestrator = Orchestrator(executor=executor)

        # The executor should be set
        assert orchestrator.executor is executor

        # Verify executor lifecycle would be managed
        # (actual delegation to executor is Phase 2)
        assert executor.setup_called is False
        assert executor.close_called is False

    @pytest.mark.asyncio
    async def test_executor_lifecycle_managed(self):
        """Test that executor setup/cleanup is called during execution."""
        # Now that we actually delegate to executor, use LocalExecutor
        from hexdag.drivers.executors import LocalExecutor

        executor = LocalExecutor()

        graph = DirectedGraph()
        graph.add(NodeSpec("add", async_add_one))

        orchestrator = Orchestrator(executor=executor)

        # Execute pipeline
        results = await orchestrator.run(graph, 5)

        # Verify lifecycle methods were called and cleanup happened
        # Executor is cleaned up after run (aclose called by context manager)
        assert executor._initialized is False
        assert "add" in results
        assert results["add"] == 6

    @pytest.mark.asyncio
    async def test_from_builder_with_executor(self):
        """Test creating orchestrator from PortsBuilder with executor."""
        executor = MockExecutor()

        # Create orchestrator using from_builder
        orchestrator = Orchestrator.from_builder(
            PortsBuilder(),
            max_concurrent_nodes=5,
            executor=executor,
        )

        assert orchestrator.executor is executor
        assert orchestrator.max_concurrent_nodes == 5

    @pytest.mark.asyncio
    async def test_executor_setup_failure_propagates(self):
        """Test that executor setup failures are propagated."""

        class FailingExecutor:
            """Executor that fails during setup."""

            async def aexecute_node(self, task: ExecutionTask) -> ExecutionResult:
                return ExecutionResult(node_name=task.node_name, status="success")

            async def aexecute_wave(self, tasks: list[ExecutionTask]) -> dict[str, ExecutionResult]:
                return {}

            async def asetup(self) -> None:
                raise RuntimeError("Setup failed!")

            async def aclose(self) -> None:
                pass

        executor = FailingExecutor()
        orchestrator = Orchestrator(executor=executor)

        graph = DirectedGraph()
        graph.add(NodeSpec("dummy", async_add_one))

        # Setup failure should propagate
        with pytest.raises(RuntimeError, match="Setup failed!"):
            await orchestrator.run(graph, 0)

    @pytest.mark.asyncio
    async def test_executor_cleanup_called_even_on_error(self):
        """Test that executor cleanup is called even when execution fails."""
        # Use LocalExecutor which properly handles cleanup
        from hexdag.drivers.executors import LocalExecutor

        executor = LocalExecutor()

        # Create a DAG with failing node
        async def failing_node(x: int, **kwargs) -> int:
            raise ValueError("Intentional failure")

        graph = DirectedGraph()
        graph.add(NodeSpec("fail", failing_node))

        orchestrator = Orchestrator(executor=executor)

        # Execution should fail with NodeExecutionError (propagated from executor)
        from hexdag.kernel.orchestration.components import NodeExecutionError

        with pytest.raises(NodeExecutionError):
            await orchestrator.run(graph, 0)

        # Executor was cleaned up (context manager closed it)
        assert executor._initialized is False

    @pytest.mark.asyncio
    async def test_multiple_executors_independent(self):
        """Test that multiple orchestrator instances with different executors are independent."""
        executor1 = MockExecutor()
        executor2 = MockExecutor()

        orchestrator1 = Orchestrator(executor=executor1)
        orchestrator2 = Orchestrator(executor=executor2)

        assert orchestrator1.executor is executor1
        assert orchestrator2.executor is executor2
        assert orchestrator1.executor is not orchestrator2.executor

    @pytest.mark.asyncio
    async def test_executor_protocol_validation(self):
        """Test that executor implements ExecutorPort protocol."""
        executor = MockExecutor()

        # Protocol check
        assert isinstance(executor, ExecutorPort)

        # Verify all required methods exist
        assert hasattr(executor, "aexecute_node")
        assert hasattr(executor, "aexecute_wave")
        assert callable(executor.aexecute_node)
        assert callable(executor.aexecute_wave)


# ============================================================================
# Skip propagation integration tests
# ============================================================================


async def passthrough(input_data, **ports):
    """Pass input through unchanged."""
    return input_data


async def transform(input_data, **ports):
    """Transform input data."""
    if isinstance(input_data, dict):
        return {**input_data, "transformed": True}
    return {"value": input_data, "transformed": True}


class TestSkipPropagation:
    """Test that skip propagates through dependency chains."""

    @pytest.mark.asyncio()
    async def test_linear_chain_skip_propagation(self) -> None:
        """A -> B -> C: if A is skipped, B and C should also be skipped."""
        graph = DirectedGraph()

        # Node A: skipped by when clause (condition always false)
        graph += NodeSpec("node_a", passthrough, when="False")
        # Node B: depends on A, no when clause
        graph += NodeSpec("node_b", transform, deps={"node_a"})
        # Node C: depends on B, no when clause
        graph += NodeSpec("node_c", transform, deps={"node_b"})

        orchestrator = Orchestrator()
        results = await orchestrator.run(graph, {"data": "input"})

        # All three should be skipped
        assert results["node_a"]["_skipped"] is True
        assert results["node_b"]["_skipped"] is True
        assert results["node_b"]["_upstream_skipped"] is True
        assert results["node_c"]["_skipped"] is True
        assert results["node_c"]["_upstream_skipped"] is True

    @pytest.mark.asyncio()
    async def test_partial_skip_with_two_branches(self) -> None:
        """Branch A skipped, branch B runs. Node C depends on both -> runs with B's result.

        Graph:
          A (skipped) -+
                        +- C
          B (runs)    -+
        """
        graph = DirectedGraph()

        # Branch A: skipped
        graph += NodeSpec("branch_a", passthrough, when="False")
        # Branch B: runs normally
        graph += NodeSpec("branch_b", transform)
        # Node C: depends on both A and B
        graph += NodeSpec("node_c", passthrough, deps={"branch_a", "branch_b"})

        orchestrator = Orchestrator()
        results = await orchestrator.run(graph, {"data": "input"})

        # A is skipped
        assert results["branch_a"]["_skipped"] is True
        # B ran normally
        assert results["branch_b"]["transformed"] is True
        # C should NOT be skipped (one dep is alive)
        assert "_skipped" not in results["node_c"] or results["node_c"].get("_skipped") is not True

    @pytest.mark.asyncio()
    async def test_both_branches_skipped_propagates(self) -> None:
        """Both branches skipped -> downstream is also skipped.

        Graph:
          A (skipped) -+
                        +- C (should be skipped)
          B (skipped) -+
        """
        graph = DirectedGraph()

        graph += NodeSpec("branch_a", passthrough, when="False")
        graph += NodeSpec("branch_b", passthrough, when="False")
        graph += NodeSpec("node_c", passthrough, deps={"branch_a", "branch_b"})

        orchestrator = Orchestrator()
        results = await orchestrator.run(graph, {"data": "input"})

        assert results["branch_a"]["_skipped"] is True
        assert results["branch_b"]["_skipped"] is True
        assert results["node_c"]["_skipped"] is True
        assert results["node_c"]["_upstream_skipped"] is True

    @pytest.mark.asyncio()
    async def test_when_condition_skip_does_not_affect_independent_nodes(self) -> None:
        """Skipping one node shouldn't affect independent (non-dependent) nodes.

        Graph:
          A (skipped)     B (runs independently)
        """
        graph = DirectedGraph()

        graph += NodeSpec("node_a", passthrough, when="False")
        graph += NodeSpec("node_b", transform)  # No deps -- independent

        orchestrator = Orchestrator()
        results = await orchestrator.run(graph, {"data": "input"})

        assert results["node_a"]["_skipped"] is True
        assert results["node_b"]["transformed"] is True
