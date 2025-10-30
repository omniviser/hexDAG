"""Tests for ExecutorPort interface and execution models."""

import pytest

from hexdag.core.ports.executor import (
    ExecutionResult,
    ExecutionTask,
    ExecutorPort,
)


class TestExecutionTask:
    """Tests for ExecutionTask model."""

    def test_create_basic_task(self):
        """Test creating a basic execution task."""
        task = ExecutionTask(
            node_name="test_node",
            node_input={"data": "value"},
            wave_index=0,
        )

        assert task.node_name == "test_node"
        assert task.node_input == {"data": "value"}
        assert task.wave_index == 0
        assert task.should_validate is True
        assert task.context_data == {}
        assert task.params == {}

    def test_create_task_with_context(self):
        """Test creating a task with execution context."""
        task = ExecutionTask(
            node_name="task_with_context",
            node_input="input_data",
            wave_index=2,
            should_validate=False,
            context_data={"dag_id": "my_dag", "run_id": "abc123"},
            params={"timeout": 30},
        )

        assert task.node_name == "task_with_context"
        assert task.should_validate is False
        assert task.context_data["dag_id"] == "my_dag"
        assert task.context_data["run_id"] == "abc123"
        assert task.params["timeout"] == 30

    def test_task_serialization(self):
        """Test that ExecutionTask can be serialized to dict."""
        task = ExecutionTask(
            node_name="serializable",
            node_input={"x": 1, "y": 2},
            wave_index=1,
        )

        task_dict = task.model_dump()
        assert task_dict["node_name"] == "serializable"
        assert task_dict["node_input"] == {"x": 1, "y": 2}
        assert task_dict["wave_index"] == 1

    def test_task_deserialization(self):
        """Test that ExecutionTask can be deserialized from dict."""
        task_dict = {
            "node_name": "deserializable",
            "node_input": [1, 2, 3],
            "wave_index": 3,
            "should_validate": False,
            "context_data": {"test": "data"},
            "params": {"key": "value"},
        }

        task = ExecutionTask.model_validate(task_dict)
        assert task.node_name == "deserializable"
        assert task.node_input == [1, 2, 3]
        assert task.wave_index == 3
        assert task.should_validate is False


class TestExecutionResult:
    """Tests for ExecutionResult model."""

    def test_create_success_result(self):
        """Test creating a successful execution result."""
        result = ExecutionResult(
            node_name="success_node",
            output={"result": "data"},
            duration_ms=150.5,
            status="success",
        )

        assert result.node_name == "success_node"
        assert result.output == {"result": "data"}
        assert result.duration_ms == 150.5
        assert result.status == "success"
        assert result.error is None
        assert result.error_type is None

    def test_create_failure_result(self):
        """Test creating a failed execution result."""
        result = ExecutionResult(
            node_name="failed_node",
            duration_ms=50.0,
            status="failed",
            error="Division by zero",
            error_type="ZeroDivisionError",
        )

        assert result.node_name == "failed_node"
        assert result.output is None
        assert result.status == "failed"
        assert result.error == "Division by zero"
        assert result.error_type == "ZeroDivisionError"

    def test_create_timeout_result(self):
        """Test creating a timeout/cancelled execution result."""
        from hexdag.core.orchestration.hook_context import PipelineStatus

        result = ExecutionResult(
            node_name="timeout_node",
            duration_ms=30000.0,
            status=PipelineStatus.CANCELLED,
            error="Task exceeded timeout of 30s",
            error_type="TimeoutError",
        )

        assert result.status == PipelineStatus.CANCELLED
        assert "timeout" in result.error.lower()

    def test_result_serialization(self):
        """Test that ExecutionResult can be serialized to dict."""
        result = ExecutionResult(
            node_name="test",
            output=[1, 2, 3],
            duration_ms=100.0,
            status="success",
        )

        result_dict = result.model_dump()
        assert result_dict["node_name"] == "test"
        assert result_dict["output"] == [1, 2, 3]
        assert result_dict["status"] == "success"


class MockExecutor:
    """Mock executor for testing ExecutorPort protocol."""

    def __init__(self):
        self.setup_called = False
        self.close_called = False
        self.executed_tasks: list[ExecutionTask] = []

    async def aexecute_node(self, task: ExecutionTask) -> ExecutionResult:
        """Execute a single node (mock implementation)."""
        self.executed_tasks.append(task)
        return ExecutionResult(
            node_name=task.node_name,
            output=f"result_{task.node_name}",
            duration_ms=10.0,
            status="success",
        )

    async def aexecute_wave(self, tasks: list[ExecutionTask]) -> dict[str, ExecutionResult]:
        """Execute multiple nodes concurrently (mock implementation)."""
        results = {}
        for task in tasks:
            result = await self.aexecute_node(task)
            results[task.node_name] = result
        return results

    async def asetup(self) -> None:
        """Initialize executor resources."""
        self.setup_called = True

    async def aclose(self) -> None:
        """Cleanup executor resources."""
        self.close_called = True


class TestExecutorPort:
    """Tests for ExecutorPort protocol interface."""

    @pytest.mark.asyncio
    async def test_executor_implements_protocol(self):
        """Test that mock executor implements ExecutorPort protocol."""
        executor = MockExecutor()

        # Check that it implements the protocol
        assert isinstance(executor, ExecutorPort)

    @pytest.mark.asyncio
    async def test_execute_single_node(self):
        """Test executing a single node through executor."""
        executor = MockExecutor()

        task = ExecutionTask(
            node_name="test_node",
            node_input={"data": "value"},
            wave_index=0,
        )

        result = await executor.aexecute_node(task)

        assert result.node_name == "test_node"
        assert result.status == "success"
        assert result.output == "result_test_node"
        assert len(executor.executed_tasks) == 1

    @pytest.mark.asyncio
    async def test_execute_wave(self):
        """Test executing multiple nodes in a wave."""
        executor = MockExecutor()

        tasks = [
            ExecutionTask(node_name="node_a", node_input="input_a", wave_index=0),
            ExecutionTask(node_name="node_b", node_input="input_b", wave_index=0),
            ExecutionTask(node_name="node_c", node_input="input_c", wave_index=0),
        ]

        results = await executor.aexecute_wave(tasks)

        assert len(results) == 3
        assert "node_a" in results
        assert "node_b" in results
        assert "node_c" in results
        assert results["node_a"].status == "success"
        assert len(executor.executed_tasks) == 3

    @pytest.mark.asyncio
    async def test_executor_lifecycle(self):
        """Test executor setup and cleanup lifecycle."""
        executor = MockExecutor()

        assert executor.setup_called is False
        assert executor.close_called is False

        await executor.asetup()
        assert executor.setup_called is True

        await executor.aclose()
        assert executor.close_called is True
