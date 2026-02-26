"""Tests for ProcessRegistry lib and ProcessRegistryObserver."""

from __future__ import annotations

import pytest

from hexdag.kernel.domain.pipeline_run import PipelineRun, RunStatus
from hexdag.kernel.orchestration.events.events import (
    NodeFailed,
    PipelineCompleted,
    PipelineStarted,
)
from hexdag.stdlib.adapters.memory.collection_memory import InMemoryCollectionStorage
from hexdag.stdlib.lib.process_registry import ProcessRegistry
from hexdag.stdlib.lib.process_registry_observer import ProcessRegistryObserver
from hexdag.stdlib.lib_base import HexDAGLib

# ---------------------------------------------------------------------------
# ProcessRegistry — direct API
# ---------------------------------------------------------------------------


class TestProcessRegistryIsLib:
    def test_is_hexdag_lib(self) -> None:
        assert isinstance(ProcessRegistry(), HexDAGLib)

    def test_exposed_tools(self) -> None:
        reg = ProcessRegistry()
        tools = reg.get_tools()
        assert "aget" in tools
        assert "alist" in tools
        assert "alist_by_ref" in tools
        assert len(tools) == 3


class TestRegisterAndGet:
    @pytest.mark.asyncio()
    async def test_register_and_aget(self) -> None:
        reg = ProcessRegistry()
        run = PipelineRun(run_id="r1", pipeline_name="p1", status=RunStatus.RUNNING)
        await reg.register(run)
        result = await reg.aget("r1")
        assert result is not None
        assert result["run_id"] == "r1"
        assert result["pipeline_name"] == "p1"
        assert result["status"] == "running"

    @pytest.mark.asyncio()
    async def test_aget_missing_returns_none(self) -> None:
        reg = ProcessRegistry()
        assert await reg.aget("nonexistent") is None


class TestUpdateStatus:
    @pytest.mark.asyncio()
    async def test_update_to_completed(self) -> None:
        reg = ProcessRegistry()
        run = PipelineRun(run_id="r1", pipeline_name="p1")
        await reg.register(run)
        await reg.update_status("r1", RunStatus.COMPLETED, duration_ms=1234.5)
        result = await reg.aget("r1")
        assert result is not None
        assert result["status"] == "completed"
        assert result["duration_ms"] == 1234.5

    @pytest.mark.asyncio()
    async def test_update_with_error(self) -> None:
        reg = ProcessRegistry()
        run = PipelineRun(run_id="r1", pipeline_name="p1")
        await reg.register(run)
        await reg.update_status("r1", RunStatus.FAILED, error="bad stuff")
        result = await reg.aget("r1")
        assert result is not None
        assert result["status"] == "failed"
        assert result["error"] == "bad stuff"

    @pytest.mark.asyncio()
    async def test_update_unknown_run_is_noop(self) -> None:
        reg = ProcessRegistry()
        await reg.update_status("nonexistent", RunStatus.COMPLETED)


class TestAlist:
    @pytest.mark.asyncio()
    async def test_list_all(self) -> None:
        reg = ProcessRegistry()
        for i in range(5):
            await reg.register(PipelineRun(run_id=f"r{i}", pipeline_name="p", created_at=float(i)))
        result = await reg.alist()
        assert len(result) == 5
        # Newest first
        assert result[0]["run_id"] == "r4"

    @pytest.mark.asyncio()
    async def test_list_by_status(self) -> None:
        reg = ProcessRegistry()
        await reg.register(PipelineRun(run_id="r1", pipeline_name="p", status=RunStatus.RUNNING))
        await reg.register(PipelineRun(run_id="r2", pipeline_name="p", status=RunStatus.COMPLETED))
        await reg.register(PipelineRun(run_id="r3", pipeline_name="p", status=RunStatus.RUNNING))
        result = await reg.alist(status="running")
        assert len(result) == 2

    @pytest.mark.asyncio()
    async def test_list_with_limit(self) -> None:
        reg = ProcessRegistry()
        for i in range(10):
            await reg.register(PipelineRun(run_id=f"r{i}", pipeline_name="p", created_at=float(i)))
        result = await reg.alist(limit=3)
        assert len(result) == 3

    @pytest.mark.asyncio()
    async def test_list_empty(self) -> None:
        reg = ProcessRegistry()
        assert await reg.alist() == []


class TestAlistByRef:
    @pytest.mark.asyncio()
    async def test_by_ref_id(self) -> None:
        reg = ProcessRegistry()
        await reg.register(PipelineRun(run_id="r1", pipeline_name="p", ref_id="order-1"))
        await reg.register(PipelineRun(run_id="r2", pipeline_name="p", ref_id="order-2"))
        await reg.register(PipelineRun(run_id="r3", pipeline_name="p", ref_id="order-1"))
        result = await reg.alist_by_ref("order-1")
        assert len(result) == 2

    @pytest.mark.asyncio()
    async def test_by_ref_id_and_type(self) -> None:
        reg = ProcessRegistry()
        await reg.register(
            PipelineRun(run_id="r1", pipeline_name="p", ref_id="123", ref_type="order")
        )
        await reg.register(
            PipelineRun(run_id="r2", pipeline_name="p", ref_id="123", ref_type="customer")
        )
        result = await reg.alist_by_ref("123", ref_type="order")
        assert len(result) == 1
        assert result[0]["ref_type"] == "order"

    @pytest.mark.asyncio()
    async def test_by_ref_no_match(self) -> None:
        reg = ProcessRegistry()
        assert await reg.alist_by_ref("missing") == []


# ---------------------------------------------------------------------------
# ProcessRegistryObserver — event-driven population
# ---------------------------------------------------------------------------


class TestProcessRegistryObserver:
    @pytest.mark.asyncio()
    async def test_pipeline_started(self) -> None:
        reg = ProcessRegistry()
        obs = ProcessRegistryObserver(reg)
        event = PipelineStarted(name="my_pipeline", total_waves=3, total_nodes=5)
        await obs.handle(event)
        run_id = obs.get_active_run_id("my_pipeline")
        assert run_id is not None
        result = await reg.aget(run_id)
        assert result is not None
        assert result["pipeline_name"] == "my_pipeline"
        assert result["status"] == "running"

    @pytest.mark.asyncio()
    async def test_pipeline_completed(self) -> None:
        reg = ProcessRegistry()
        obs = ProcessRegistryObserver(reg)
        await obs.handle(PipelineStarted(name="p", total_waves=1, total_nodes=1))
        run_id = obs.get_active_run_id("p")
        assert run_id is not None
        await obs.handle(PipelineCompleted(name="p", duration_ms=500.0, node_results={"n1": "ok"}))
        result = await reg.aget(run_id)
        assert result is not None
        assert result["status"] == "completed"
        assert result["duration_ms"] == 500.0

    @pytest.mark.asyncio()
    async def test_node_failed_marks_run_as_failed(self) -> None:
        reg = ProcessRegistry()
        obs = ProcessRegistryObserver(reg)
        await obs.handle(PipelineStarted(name="p", total_waves=1, total_nodes=1))
        run_id = obs.get_active_run_id("p")
        assert run_id is not None
        await obs.handle(NodeFailed(name="bad_node", wave_index=0, error=ValueError("bad input")))
        result = await reg.aget(run_id)
        assert result is not None
        assert result["status"] == "failed"
        assert "bad input" in result["error"]

    @pytest.mark.asyncio()
    async def test_pipeline_cancelled(self) -> None:
        reg = ProcessRegistry()
        obs = ProcessRegistryObserver(reg)
        await obs.handle(PipelineStarted(name="p", total_waves=1, total_nodes=1))
        run_id = obs.get_active_run_id("p")
        assert run_id is not None
        await obs.handle(
            PipelineCompleted(
                name="p", duration_ms=100.0, status="cancelled", reason="user request"
            )
        )
        result = await reg.aget(run_id)
        assert result is not None
        assert result["status"] == "cancelled"

    @pytest.mark.asyncio()
    async def test_ignores_unknown_events(self) -> None:
        """Observer should not crash on events it doesn't handle."""
        from hexdag.kernel.orchestration.events.events import NodeStarted

        reg = ProcessRegistry()
        obs = ProcessRegistryObserver(reg)
        event = NodeStarted(name="n1", wave_index=0, dependencies=[])
        await obs.handle(event)  # Should not raise
        assert await reg.alist() == []


# ---------------------------------------------------------------------------
# ProcessRegistry — with SupportsCollectionStorage
# ---------------------------------------------------------------------------


class TestProcessRegistryWithStorage:
    @pytest.mark.asyncio()
    async def test_register_persists(self) -> None:
        storage = InMemoryCollectionStorage()
        reg = ProcessRegistry(storage=storage)
        run = PipelineRun(run_id="r1", pipeline_name="p1", status=RunStatus.RUNNING)
        await reg.register(run)
        doc = await storage.aload("pipeline_runs", "r1")
        assert doc is not None
        assert doc["run_id"] == "r1"
        assert doc["status"] == "running"

    @pytest.mark.asyncio()
    async def test_update_persists(self) -> None:
        storage = InMemoryCollectionStorage()
        reg = ProcessRegistry(storage=storage)
        run = PipelineRun(run_id="r1", pipeline_name="p1")
        await reg.register(run)
        await reg.update_status("r1", RunStatus.COMPLETED, duration_ms=500.0)
        doc = await storage.aload("pipeline_runs", "r1")
        assert doc is not None
        assert doc["status"] == "completed"
        assert doc["duration_ms"] == 500.0

    @pytest.mark.asyncio()
    async def test_aget_falls_back_to_storage(self) -> None:
        storage = InMemoryCollectionStorage()
        await storage.asave(
            "pipeline_runs",
            "r1",
            {
                "run_id": "r1",
                "pipeline_name": "p1",
                "status": "completed",
                "ref_id": None,
                "ref_type": None,
                "parent_run_id": None,
                "created_at": 1000.0,
                "started_at": 1001.0,
                "completed_at": 1005.0,
                "duration_ms": 4000.0,
                "error": None,
            },
        )
        reg = ProcessRegistry(storage=storage)
        result = await reg.aget("r1")
        assert result is not None
        assert result["run_id"] == "r1"
        assert result["status"] == "completed"

    @pytest.mark.asyncio()
    async def test_alist_uses_storage(self) -> None:
        storage = InMemoryCollectionStorage()
        reg = ProcessRegistry(storage=storage)
        await reg.register(
            PipelineRun(
                run_id="r1",
                pipeline_name="p1",
                status=RunStatus.RUNNING,
                created_at=1.0,
            )
        )
        await reg.register(
            PipelineRun(
                run_id="r2",
                pipeline_name="p2",
                status=RunStatus.COMPLETED,
                created_at=2.0,
            )
        )
        result = await reg.alist(status="running")
        assert len(result) == 1
        assert result[0]["run_id"] == "r1"

    @pytest.mark.asyncio()
    async def test_alist_by_ref_uses_storage(self) -> None:
        storage = InMemoryCollectionStorage()
        reg = ProcessRegistry(storage=storage)
        await reg.register(PipelineRun(run_id="r1", pipeline_name="p1", ref_id="order-1"))
        await reg.register(PipelineRun(run_id="r2", pipeline_name="p2", ref_id="order-2"))
        result = await reg.alist_by_ref("order-1")
        assert len(result) == 1
        assert result[0]["ref_id"] == "order-1"

    @pytest.mark.asyncio()
    async def test_backward_compat_no_storage(self) -> None:
        """ProcessRegistry without storage works exactly as before."""
        reg = ProcessRegistry()
        run = PipelineRun(run_id="r1", pipeline_name="p1")
        await reg.register(run)
        assert (await reg.aget("r1")) is not None
        assert len(await reg.alist()) == 1
