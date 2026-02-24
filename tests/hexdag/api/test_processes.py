"""Tests for the processes API module."""

from __future__ import annotations

import pytest

from hexdag.api import processes
from hexdag.kernel.domain.entity_state import StateMachineConfig
from hexdag.kernel.domain.pipeline_run import PipelineRun, RunStatus
from hexdag.stdlib.lib.entity_state import EntityState, InvalidTransitionError
from hexdag.stdlib.lib.process_registry import ProcessRegistry
from hexdag.stdlib.lib.scheduler import Scheduler

# ---------------------------------------------------------------------------
# Mock spawner
# ---------------------------------------------------------------------------


class _MockSpawner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._counter = 0

    async def aspawn(
        self,
        pipeline_name,
        initial_input,
        *,
        ref_id=None,
        ref_type=None,
        parent_run_id=None,
        wait=False,
        timeout=None,
    ) -> str:
        self._counter += 1
        self.calls.append((pipeline_name, initial_input))
        return f"run-{self._counter}"

    async def aspawn_many(self, *_a, **_kw) -> list[str]:
        return []

    async def await_result(self, *_a, **_kw) -> dict:
        return {}

    async def acancel(self, *_a, **_kw) -> None:
        pass


# ---------------------------------------------------------------------------
# ProcessRegistry API
# ---------------------------------------------------------------------------


class TestListPipelineRuns:
    @pytest.mark.asyncio()
    async def test_list_all(self) -> None:
        reg = ProcessRegistry()
        await reg.register(PipelineRun(run_id="r1", pipeline_name="p1"))
        await reg.register(PipelineRun(run_id="r2", pipeline_name="p2"))
        result = await processes.list_pipeline_runs(reg)
        assert len(result) == 2

    @pytest.mark.asyncio()
    async def test_list_by_status(self) -> None:
        reg = ProcessRegistry()
        await reg.register(PipelineRun(run_id="r1", pipeline_name="p", status=RunStatus.RUNNING))
        await reg.register(PipelineRun(run_id="r2", pipeline_name="p", status=RunStatus.COMPLETED))
        result = await processes.list_pipeline_runs(reg, status="running")
        assert len(result) == 1

    @pytest.mark.asyncio()
    async def test_list_by_ref_id(self) -> None:
        reg = ProcessRegistry()
        await reg.register(PipelineRun(run_id="r1", pipeline_name="p", ref_id="order-1"))
        await reg.register(PipelineRun(run_id="r2", pipeline_name="p", ref_id="order-2"))
        result = await processes.list_pipeline_runs(reg, ref_id="order-1")
        assert len(result) == 1


class TestGetPipelineRun:
    @pytest.mark.asyncio()
    async def test_get_existing(self) -> None:
        reg = ProcessRegistry()
        await reg.register(PipelineRun(run_id="r1", pipeline_name="p1"))
        result = await processes.get_pipeline_run(reg, "r1")
        assert result is not None
        assert result["run_id"] == "r1"

    @pytest.mark.asyncio()
    async def test_get_missing(self) -> None:
        reg = ProcessRegistry()
        assert await processes.get_pipeline_run(reg, "nope") is None


# ---------------------------------------------------------------------------
# PipelineSpawner API
# ---------------------------------------------------------------------------


class TestSpawnPipeline:
    @pytest.mark.asyncio()
    async def test_spawn(self) -> None:
        spawner = _MockSpawner()
        result = await processes.spawn_pipeline(spawner, "my_pipe", {"x": 1})
        assert result["run_id"] == "run-1"
        assert result["pipeline_name"] == "my_pipe"
        assert result["status"] == "spawned"
        assert spawner.calls == [("my_pipe", {"x": 1})]

    @pytest.mark.asyncio()
    async def test_spawn_with_ref(self) -> None:
        spawner = _MockSpawner()
        result = await processes.spawn_pipeline(spawner, "p", ref_id="order-1", ref_type="order")
        assert result["run_id"] == "run-1"


# ---------------------------------------------------------------------------
# Scheduler API
# ---------------------------------------------------------------------------


class TestSchedulePipeline:
    @pytest.mark.asyncio()
    async def test_schedule_once(self) -> None:
        sched = Scheduler()
        result = await processes.schedule_pipeline(sched, "p1", delay_seconds=60.0)
        assert result["pipeline_name"] == "p1"
        assert result["schedule_type"] == "once"
        await sched.ateardown()

    @pytest.mark.asyncio()
    async def test_schedule_recurring(self) -> None:
        sched = Scheduler()
        result = await processes.schedule_pipeline(sched, "p1", interval_seconds=300.0)
        assert result["schedule_type"] == "recurring"
        await sched.ateardown()

    @pytest.mark.asyncio()
    async def test_recurring_takes_precedence(self) -> None:
        sched = Scheduler()
        result = await processes.schedule_pipeline(
            sched, "p1", delay_seconds=10.0, interval_seconds=300.0
        )
        assert result["schedule_type"] == "recurring"
        await sched.ateardown()


class TestCancelScheduled:
    @pytest.mark.asyncio()
    async def test_cancel(self) -> None:
        sched = Scheduler()
        task = await processes.schedule_pipeline(sched, "p1", delay_seconds=100.0)
        result = await processes.cancel_scheduled(sched, task["task_id"])
        assert result["cancelled"] is True
        await sched.ateardown()

    @pytest.mark.asyncio()
    async def test_cancel_missing(self) -> None:
        sched = Scheduler()
        result = await processes.cancel_scheduled(sched, "nope")
        assert result["cancelled"] is False


class TestListScheduled:
    @pytest.mark.asyncio()
    async def test_list_all(self) -> None:
        sched = Scheduler()
        await processes.schedule_pipeline(sched, "p1", delay_seconds=100.0)
        await processes.schedule_pipeline(sched, "p2", delay_seconds=100.0)
        result = await processes.list_scheduled(sched)
        assert len(result) == 2
        await sched.ateardown()

    @pytest.mark.asyncio()
    async def test_list_by_ref(self) -> None:
        sched = Scheduler()
        await processes.schedule_pipeline(sched, "p1", delay_seconds=100.0, ref_id="o1")
        await processes.schedule_pipeline(sched, "p2", delay_seconds=100.0, ref_id="o2")
        result = await processes.list_scheduled(sched, ref_id="o1")
        assert len(result) == 1
        await sched.ateardown()


# ---------------------------------------------------------------------------
# EntityState API
# ---------------------------------------------------------------------------


class TestGetEntityState:
    @pytest.mark.asyncio()
    async def test_get_state(self) -> None:
        es = EntityState()
        await es.aregister_entity("order", "o1")
        result = await processes.get_entity_state(es, "order", "o1")
        assert result is not None
        assert result["state"] == "created"

    @pytest.mark.asyncio()
    async def test_get_missing(self) -> None:
        es = EntityState()
        assert await processes.get_entity_state(es, "order", "nope") is None


class TestTransitionEntity:
    @pytest.mark.asyncio()
    async def test_transition(self) -> None:
        es = EntityState()
        es.register_machine(
            StateMachineConfig(
                entity_type="order",
                states={"new", "processing"},
                initial_state="new",
                transitions={"new": {"processing"}},
            )
        )
        await es.aregister_entity("order", "o1")
        result = await processes.transition_entity(es, "order", "o1", "processing")
        assert result["from_state"] == "new"
        assert result["to_state"] == "processing"

    @pytest.mark.asyncio()
    async def test_transition_with_reason(self) -> None:
        es = EntityState()
        await es.aregister_entity("custom", "c1")
        result = await processes.transition_entity(es, "custom", "c1", "active", reason="approved")
        assert result["to_state"] == "active"

    @pytest.mark.asyncio()
    async def test_invalid_transition(self) -> None:
        es = EntityState()
        es.register_machine(
            StateMachineConfig(
                entity_type="order",
                states={"new", "done"},
                initial_state="new",
                transitions={},
            )
        )
        await es.aregister_entity("order", "o1")
        with pytest.raises(InvalidTransitionError):
            await processes.transition_entity(es, "order", "o1", "done")


class TestGetEntityHistory:
    @pytest.mark.asyncio()
    async def test_history(self) -> None:
        es = EntityState()
        await es.aregister_entity("order", "o1")
        await es.atransition("order", "o1", "active")
        result = await processes.get_entity_history(es, "order", "o1")
        assert len(result) == 2

    @pytest.mark.asyncio()
    async def test_history_empty(self) -> None:
        es = EntityState()
        result = await processes.get_entity_history(es, "order", "nope")
        assert result == []
