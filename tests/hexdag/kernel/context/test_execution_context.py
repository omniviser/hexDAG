"""Tests for ExecutionContext and pipeline_name context propagation."""

from __future__ import annotations

import pytest

from hexdag.kernel.context import (
    ExecutionContext,
    clear_execution_context,
    get_pipeline_name,
    get_ports,
    get_run_id,
    set_pipeline_name,
)


class TestPipelineNameContext:
    """Test pipeline_name ContextVar get/set functions."""

    def setup_method(self):
        clear_execution_context()

    def teardown_method(self):
        clear_execution_context()

    def test_default_is_none(self) -> None:
        assert get_pipeline_name() is None

    def test_set_and_get(self) -> None:
        set_pipeline_name("my-pipeline")
        assert get_pipeline_name() == "my-pipeline"

    def test_set_none_clears(self) -> None:
        set_pipeline_name("pipeline-1")
        set_pipeline_name(None)
        assert get_pipeline_name() is None

    def test_clear_execution_context_resets(self) -> None:
        set_pipeline_name("pipeline-2")
        clear_execution_context()
        assert get_pipeline_name() is None


class TestExecutionContextManager:
    """Test ExecutionContext as sync and async context manager."""

    def setup_method(self):
        clear_execution_context()

    def teardown_method(self):
        clear_execution_context()

    def test_sync_context_sets_pipeline_name(self) -> None:
        with ExecutionContext(pipeline_name="sync-pipeline"):
            assert get_pipeline_name() == "sync-pipeline"
        # Cleaned up after exit
        assert get_pipeline_name() is None

    def test_sync_context_sets_run_id(self) -> None:
        with ExecutionContext(run_id="run-123"):
            assert get_run_id() == "run-123"
        assert get_run_id() is None

    def test_sync_context_sets_ports(self) -> None:
        ports = {"llm": "mock_llm"}
        with ExecutionContext(ports=ports):
            ctx_ports = get_ports()
            assert ctx_ports is not None
            assert ctx_ports["llm"] == "mock_llm"
        assert get_ports() is None

    def test_sync_context_cleans_up_on_exception(self) -> None:
        with pytest.raises(ValueError, match="test error"):
            with ExecutionContext(pipeline_name="error-pipeline"):
                assert get_pipeline_name() == "error-pipeline"
                raise ValueError("test error")
        assert get_pipeline_name() is None

    @pytest.mark.asyncio()
    async def test_async_context_sets_pipeline_name(self) -> None:
        async with ExecutionContext(pipeline_name="async-pipeline"):
            assert get_pipeline_name() == "async-pipeline"
        assert get_pipeline_name() is None

    @pytest.mark.asyncio()
    async def test_async_context_sets_all_fields(self) -> None:
        async with ExecutionContext(
            run_id="run-456",
            ports={"db": "mock_db"},
            pipeline_name="full-pipeline",
        ):
            assert get_run_id() == "run-456"
            assert get_pipeline_name() == "full-pipeline"
            ports = get_ports()
            assert ports is not None
            assert ports["db"] == "mock_db"

        assert get_run_id() is None
        assert get_pipeline_name() is None
        assert get_ports() is None

    @pytest.mark.asyncio()
    async def test_async_context_cleans_up_on_exception(self) -> None:
        with pytest.raises(RuntimeError, match="async error"):
            async with ExecutionContext(pipeline_name="error-async"):
                assert get_pipeline_name() == "error-async"
                raise RuntimeError("async error")
        assert get_pipeline_name() is None

    def test_pipeline_name_none_by_default(self) -> None:
        """ExecutionContext without pipeline_name leaves it as None."""
        with ExecutionContext(run_id="run-789"):
            assert get_pipeline_name() is None
            assert get_run_id() == "run-789"

    def test_ports_are_immutable(self) -> None:
        """Ports stored via ExecutionContext should be immutable (MappingProxyType)."""
        with ExecutionContext(ports={"llm": "mock"}):
            ports = get_ports()
            assert ports is not None
            with pytest.raises(TypeError):
                ports["new_key"] = "value"  # type: ignore[index]
