"""Tests for LocalPipelineSpawner."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from hexdag.drivers.pipeline_spawner.local import LocalPipelineSpawner, RunStatus
from hexdag.kernel.ports.pipeline_spawner import PipelineSpawner

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeRunner:
    """Minimal PipelineRunner substitute for testing."""

    def __init__(
        self,
        result: dict[str, Any] | None = None,
        delay: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        self._result = result or {"output": "done"}
        self._delay = delay
        self._error = error
        self.calls: list[tuple[str, Any]] = []

    async def run(
        self,
        pipeline_path: str,
        input_data: Any = None,
        *,
        environment: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append((pipeline_path, input_data))
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error:
            raise self._error
        return self._result


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_local_spawner_satisfies_protocol(self) -> None:
        runner = FakeRunner()
        spawner = LocalPipelineSpawner(runner=runner)
        assert isinstance(spawner, PipelineSpawner)


# ---------------------------------------------------------------------------
# aspawn
# ---------------------------------------------------------------------------


class TestAspawn:
    @pytest.mark.asyncio()
    async def test_spawn_wait_returns_run_id(self) -> None:
        runner = FakeRunner(result={"answer": 42})
        spawner = LocalPipelineSpawner(runner=runner)
        run_id = await spawner.aspawn("my_pipeline.yaml", {"q": "hi"}, wait=True)
        assert isinstance(run_id, str)
        assert len(run_id) > 0

    @pytest.mark.asyncio()
    async def test_spawn_wait_calls_runner(self) -> None:
        runner = FakeRunner()
        spawner = LocalPipelineSpawner(runner=runner)
        await spawner.aspawn("pipeline.yaml", {"x": 1}, wait=True)
        assert len(runner.calls) == 1
        assert runner.calls[0] == ("pipeline.yaml", {"x": 1})

    @pytest.mark.asyncio()
    async def test_spawn_wait_with_pipeline_dir(self) -> None:
        runner = FakeRunner()
        spawner = LocalPipelineSpawner(runner=runner, pipeline_dir="/pipelines")
        await spawner.aspawn("my.yaml", {}, wait=True)
        assert runner.calls[0][0] == "/pipelines/my.yaml"

    @pytest.mark.asyncio()
    async def test_spawn_fire_and_forget(self) -> None:
        runner = FakeRunner(delay=0.01)
        spawner = LocalPipelineSpawner(runner=runner)
        run_id = await spawner.aspawn("pipeline.yaml", {})
        assert spawner.get_run_status(run_id) == RunStatus.RUNNING
        # Wait for it to complete
        await asyncio.sleep(0.05)
        assert spawner.get_run_status(run_id) == RunStatus.COMPLETED

    @pytest.mark.asyncio()
    async def test_spawn_stores_ref_metadata(self) -> None:
        runner = FakeRunner()
        spawner = LocalPipelineSpawner(runner=runner)
        run_id = await spawner.aspawn(
            "p.yaml",
            {},
            ref_id="order-123",
            ref_type="order",
            parent_run_id="parent-1",
            wait=True,
        )
        state = spawner._runs[run_id]
        assert state.ref_id == "order-123"
        assert state.ref_type == "order"
        assert state.parent_run_id == "parent-1"


# ---------------------------------------------------------------------------
# aspawn_many
# ---------------------------------------------------------------------------


class TestAspawnMany:
    @pytest.mark.asyncio()
    async def test_spawn_many_returns_run_ids(self) -> None:
        runner = FakeRunner(delay=0.01)
        spawner = LocalPipelineSpawner(runner=runner)
        run_ids = await spawner.aspawn_many(
            "pipeline.yaml",
            [{"x": 1}, {"x": 2}, {"x": 3}],
        )
        assert len(run_ids) == 3
        assert len(set(run_ids)) == 3  # all unique


# ---------------------------------------------------------------------------
# await_result
# ---------------------------------------------------------------------------


class TestAwaitResult:
    @pytest.mark.asyncio()
    async def test_await_completed_run(self) -> None:
        runner = FakeRunner(result={"answer": 42})
        spawner = LocalPipelineSpawner(runner=runner)
        run_id = await spawner.aspawn("p.yaml", {}, wait=True)
        result = await spawner.await_result(run_id)
        assert result == {"answer": 42}

    @pytest.mark.asyncio()
    async def test_await_background_run(self) -> None:
        runner = FakeRunner(result={"val": "ok"}, delay=0.01)
        spawner = LocalPipelineSpawner(runner=runner)
        run_id = await spawner.aspawn("p.yaml", {})
        result = await spawner.await_result(run_id, timeout=5.0)
        assert result == {"val": "ok"}

    @pytest.mark.asyncio()
    async def test_await_unknown_run_raises(self) -> None:
        runner = FakeRunner()
        spawner = LocalPipelineSpawner(runner=runner)
        with pytest.raises(KeyError, match="Unknown run ID"):
            await spawner.await_result("nonexistent")

    @pytest.mark.asyncio()
    async def test_await_failed_run_raises(self) -> None:
        runner = FakeRunner(error=ValueError("bad input"), delay=0.01)
        spawner = LocalPipelineSpawner(runner=runner)
        run_id = await spawner.aspawn("p.yaml", {})
        await asyncio.sleep(0.05)
        with pytest.raises(RuntimeError, match="failed"):
            await spawner.await_result(run_id)


# ---------------------------------------------------------------------------
# acancel
# ---------------------------------------------------------------------------


class TestAcancel:
    @pytest.mark.asyncio()
    async def test_cancel_running_task(self) -> None:
        runner = FakeRunner(delay=10.0)  # long running
        spawner = LocalPipelineSpawner(runner=runner)
        run_id = await spawner.aspawn("p.yaml", {})
        await spawner.acancel(run_id)
        assert spawner.get_run_status(run_id) == RunStatus.CANCELLED

    @pytest.mark.asyncio()
    async def test_cancel_nonexistent_is_noop(self) -> None:
        runner = FakeRunner()
        spawner = LocalPipelineSpawner(runner=runner)
        # Should not raise
        await spawner.acancel("nonexistent")


# ---------------------------------------------------------------------------
# get_run_status
# ---------------------------------------------------------------------------


class TestGetRunStatus:
    @pytest.mark.asyncio()
    async def test_status_after_wait_spawn(self) -> None:
        runner = FakeRunner()
        spawner = LocalPipelineSpawner(runner=runner)
        run_id = await spawner.aspawn("p.yaml", {}, wait=True)
        assert spawner.get_run_status(run_id) == RunStatus.COMPLETED

    def test_status_unknown_raises(self) -> None:
        runner = FakeRunner()
        spawner = LocalPipelineSpawner(runner=runner)
        with pytest.raises(KeyError):
            spawner.get_run_status("bogus")
