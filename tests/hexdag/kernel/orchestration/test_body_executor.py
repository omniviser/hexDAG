"""Tests for BodyExecutor -- shared body execution logic for control flow nodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hexdag.kernel.exceptions import BodyExecutorError
from hexdag.kernel.orchestration.body_executor import BodyExecutor
from hexdag.kernel.orchestration.models import NodeExecutionContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(node_id: str = "test_node") -> NodeExecutionContext:
    """Create a minimal NodeExecutionContext for testing."""
    return NodeExecutionContext(
        node_id=node_id,
        dag_id="test_dag",
    )


def sync_add(input_data: dict[str, Any], **ports: Any) -> dict[str, Any]:
    """Synchronous test function."""
    return {"result": input_data.get("x", 0) + input_data.get("y", 0)}


async def async_multiply(input_data: dict[str, Any], **ports: Any) -> dict[str, Any]:
    """Asynchronous test function."""
    return {"result": input_data.get("x", 0) * input_data.get("y", 0)}


# ---------------------------------------------------------------------------
# TestBodyExecutorInit
# ---------------------------------------------------------------------------


class TestBodyExecutorInit:
    """Tests for BodyExecutor initialization."""

    def test_default_init(self) -> None:
        """Default init sets sensible defaults."""
        executor = BodyExecutor()
        assert executor.base_path == Path.cwd()
        assert executor.max_concurrent_nodes == 10
        assert executor.strict_validation is False
        assert executor.default_node_timeout is None

    def test_custom_init(self) -> None:
        """Custom parameters are stored."""
        executor = BodyExecutor(
            base_path=Path("/tmp"),
            max_concurrent_nodes=5,
            strict_validation=True,
            default_node_timeout=30.0,
        )
        assert executor.base_path == Path("/tmp")
        assert executor.max_concurrent_nodes == 5
        assert executor.strict_validation is True
        assert executor.default_node_timeout == 30.0


# ---------------------------------------------------------------------------
# TestBodyExecutorDispatch
# ---------------------------------------------------------------------------


class TestBodyExecutorDispatch:
    """Tests for execute() dispatch logic."""

    def setup_method(self) -> None:
        self.executor = BodyExecutor()
        self.context = _make_context()

    @pytest.mark.asyncio
    async def test_dispatches_to_function_by_string(self) -> None:
        """String body dispatches to _execute_function."""
        with patch.object(self.executor, "_execute_function", new_callable=AsyncMock) as mock:
            mock.return_value = {"result": 42}
            await self.executor.execute(
                body="mymodule.my_func",
                body_pipeline=None,
                input_data={"x": 1},
                context=self.context,
                ports={},
            )
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatches_to_callable(self) -> None:
        """Callable body dispatches to _execute_callable."""

        async def my_fn(data: Any, **kw: Any) -> str:
            return "ok"

        with patch.object(self.executor, "_execute_callable", new_callable=AsyncMock) as mock:
            mock.return_value = "ok"
            await self.executor.execute(
                body=my_fn,
                body_pipeline=None,
                input_data={},
                context=self.context,
                ports={},
            )
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatches_to_inline_nodes(self) -> None:
        """List body dispatches to _execute_inline_nodes."""
        with patch.object(self.executor, "_execute_inline_nodes", new_callable=AsyncMock) as mock:
            mock.return_value = {"result": "done"}
            await self.executor.execute(
                body=[{"kind": "expression_node", "metadata": {"name": "n1"}, "spec": {}}],
                body_pipeline=None,
                input_data={},
                context=self.context,
                ports={},
            )
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatches_to_pipeline(self) -> None:
        """body_pipeline dispatches to _execute_pipeline."""
        with patch.object(self.executor, "_execute_pipeline", new_callable=AsyncMock) as mock:
            mock.return_value = {"result": "ok"}
            await self.executor.execute(
                body=None,
                body_pipeline="sub_pipeline.yaml",
                input_data={},
                context=self.context,
                ports={},
            )
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_when_no_body(self) -> None:
        """Raises BodyExecutorError when no body or pipeline specified."""
        with pytest.raises(BodyExecutorError, match="No body specified"):
            await self.executor.execute(
                body=None,
                body_pipeline=None,
                input_data={},
                context=self.context,
                ports={},
            )


# ---------------------------------------------------------------------------
# TestBodyNameDerivation
# ---------------------------------------------------------------------------


class TestBodyNameDerivation:
    """Tests for body name derivation in execute()."""

    def setup_method(self) -> None:
        self.executor = BodyExecutor()
        self.context = _make_context()

    @pytest.mark.asyncio
    async def test_uses_body_pipeline_as_name(self) -> None:
        """Pipeline reference is used as body_name."""
        with patch.object(self.executor, "_execute_pipeline", new_callable=AsyncMock) as mock:
            mock.return_value = {}
            await self.executor.execute(
                body=None,
                body_pipeline="my_pipeline.yaml",
                input_data={},
                context=self.context,
                ports={},
            )
            # Check that body_name passed to _execute_pipeline contains the pipeline name
            call_args = mock.call_args
            assert "my_pipeline.yaml" in str(call_args)

    @pytest.mark.asyncio
    async def test_uses_inline_with_index(self) -> None:
        """Iteration context $index produces inline[N] name."""
        with patch.object(self.executor, "_execute_inline_nodes", new_callable=AsyncMock) as mock:
            mock.return_value = {}
            await self.executor.execute(
                body=[{"kind": "expression_node", "metadata": {"name": "n1"}, "spec": {}}],
                body_pipeline=None,
                input_data={},
                context=self.context,
                ports={},
                iteration_context={"$index": 3, "$item": "x"},
            )
            call_args = mock.call_args
            # body_name should be "inline[3]"
            assert "inline[3]" in str(call_args)


# ---------------------------------------------------------------------------
# TestExecuteFunction
# ---------------------------------------------------------------------------


class TestExecuteFunction:
    """Tests for _execute_function."""

    def setup_method(self) -> None:
        self.executor = BodyExecutor()
        self.context = _make_context()

    @pytest.mark.asyncio
    async def test_resolves_and_calls_sync_function(self) -> None:
        """Resolves a sync function and calls it."""
        with patch("hexdag.kernel.orchestration.body_executor.resolve_function") as mock_resolve:
            mock_resolve.return_value = sync_add
            result = await self.executor._execute_function(
                body="test_module.sync_add",
                input_data={"x": 3, "y": 4},
                context=self.context,
                ports={},
            )
            assert result == {"result": 7}

    @pytest.mark.asyncio
    async def test_resolves_and_calls_async_function(self) -> None:
        """Resolves an async function and awaits it."""
        with patch("hexdag.kernel.orchestration.body_executor.resolve_function") as mock_resolve:
            mock_resolve.return_value = async_multiply
            result = await self.executor._execute_function(
                body="test_module.async_multiply",
                input_data={"x": 3, "y": 4},
                context=self.context,
                ports={},
            )
            assert result == {"result": 12}

    @pytest.mark.asyncio
    async def test_raises_on_resolution_failure(self) -> None:
        """Raises BodyExecutorError when function cannot be resolved."""
        with patch("hexdag.kernel.orchestration.body_executor.resolve_function") as mock_resolve:
            mock_resolve.side_effect = ImportError("Module not found")
            with pytest.raises(BodyExecutorError, match="Failed to resolve function"):
                await self.executor._execute_function(
                    body="nonexistent.module.func",
                    input_data={},
                    context=self.context,
                    ports={},
                )

    @pytest.mark.asyncio
    async def test_wraps_execution_exceptions(self) -> None:
        """Function execution errors are wrapped in BodyExecutorError."""

        def bad_func(data: Any, **kw: Any) -> None:
            raise ValueError("bad input")

        with patch("hexdag.kernel.orchestration.body_executor.resolve_function") as mock_resolve:
            mock_resolve.return_value = bad_func
            with pytest.raises(BodyExecutorError, match="execution failed"):
                await self.executor._execute_function(
                    body="test.bad_func",
                    input_data={},
                    context=self.context,
                    ports={},
                )


# ---------------------------------------------------------------------------
# TestExecutePyFunction
# ---------------------------------------------------------------------------


class TestExecutePyFunction:
    """Tests for _execute_py_function."""

    def setup_method(self) -> None:
        self.executor = BodyExecutor()
        self.context = _make_context()

    @pytest.mark.asyncio
    async def test_extracts_item_index_state(self) -> None:
        """Extracts $item, $index, state from input_data."""

        def py_fn(item: Any, index: int, state: dict[str, Any], **ports: Any) -> Any:
            return {"item": item, "index": index}

        result = await self.executor._execute_py_function(
            body=py_fn,
            input_data={"$item": "hello", "$index": 5, "state": {"count": 1}},
            context=self.context,
            ports={},
        )
        assert result == {"item": "hello", "index": 5}

    @pytest.mark.asyncio
    async def test_handles_async_py_function(self) -> None:
        """Handles async !py functions."""

        async def async_py_fn(item: Any, index: int, state: dict[str, Any], **ports: Any) -> Any:
            return {"doubled": item * 2}

        result = await self.executor._execute_py_function(
            body=async_py_fn,
            input_data={"$item": 5, "$index": 0, "state": {}},
            context=self.context,
            ports={},
        )
        assert result == {"doubled": 10}

    @pytest.mark.asyncio
    async def test_wraps_exceptions(self) -> None:
        """!py function errors are wrapped in BodyExecutorError."""

        def failing_fn(item: Any, index: int, state: Any, **kw: Any) -> None:
            raise RuntimeError("bad")

        with pytest.raises(BodyExecutorError, match="execution failed"):
            await self.executor._execute_py_function(
                body=failing_fn,
                input_data={"$item": None, "$index": 0},
                context=self.context,
                ports={},
            )


# ---------------------------------------------------------------------------
# TestExecuteCallable
# ---------------------------------------------------------------------------


class TestExecuteCallable:
    """Tests for _execute_callable."""

    def setup_method(self) -> None:
        self.executor = BodyExecutor()
        self.context = _make_context()

    @pytest.mark.asyncio
    async def test_calls_with_input_data_and_ports(self) -> None:
        """Callable receives (input_data, **ports)."""
        received = {}

        def capture_fn(data: Any, **ports: Any) -> str:
            received["data"] = data
            received["ports"] = ports
            return "ok"

        result = await self.executor._execute_callable(
            body=capture_fn,
            input_data={"key": "value"},
            context=self.context,
            ports={"db": "mock_db"},
        )
        assert result == "ok"
        assert received["data"] == {"key": "value"}
        assert received["ports"] == {"db": "mock_db"}

    @pytest.mark.asyncio
    async def test_handles_async_callable(self) -> None:
        """Async callable is awaited."""

        async def async_fn(data: Any, **kw: Any) -> dict[str, int]:
            return {"sum": data.get("a", 0) + data.get("b", 0)}

        result = await self.executor._execute_callable(
            body=async_fn,
            input_data={"a": 10, "b": 20},
            context=self.context,
            ports={},
        )
        assert result == {"sum": 30}


# ---------------------------------------------------------------------------
# TestExecutePipeline
# ---------------------------------------------------------------------------


class TestExecutePipeline:
    """Tests for _execute_pipeline."""

    def setup_method(self) -> None:
        self.executor = BodyExecutor(base_path=Path("/tmp"))
        self.context = _make_context()

    @pytest.mark.asyncio
    async def test_raises_when_pipeline_not_found(self) -> None:
        """Raises BodyExecutorError for missing pipeline file."""
        with pytest.raises(BodyExecutorError, match="Pipeline file not found"):
            await self.executor._execute_pipeline(
                body_pipeline="nonexistent_pipeline.yaml",
                input_data={},
                context=self.context,
                ports={},
            )


# ---------------------------------------------------------------------------
# TestNotify
# ---------------------------------------------------------------------------


class TestNotify:
    """Tests for _notify event emission."""

    @pytest.mark.asyncio
    async def test_notify_suppresses_exceptions(self) -> None:
        """_notify does not raise even if observer fails."""
        executor = BodyExecutor()
        # Should not raise even with no observer set up
        await executor._notify(MagicMock())
