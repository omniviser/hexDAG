"""Tests for Scheduler lib and ScheduledTask domain model."""

from __future__ import annotations

import asyncio

import pytest

from hexdag.kernel.domain.scheduled_task import (
    ScheduledTask,
    ScheduleType,
    TaskStatus,
)
from hexdag.stdlib.lib.scheduler import Scheduler
from hexdag.stdlib.lib_base import HexDAGLib

# ---------------------------------------------------------------------------
# Domain — ScheduledTask
# ---------------------------------------------------------------------------


class TestScheduledTask:
    def test_defaults(self) -> None:
        task = ScheduledTask(
            task_id="t1",
            pipeline_name="p1",
            schedule_type=ScheduleType.ONCE,
        )
        assert task.task_id == "t1"
        assert task.status == TaskStatus.PENDING
        assert task.run_count == 0
        assert task.initial_input == {}

    def test_schedule_types(self) -> None:
        assert ScheduleType.ONCE == "once"
        assert ScheduleType.RECURRING == "recurring"

    def test_task_statuses(self) -> None:
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.CANCELLED == "cancelled"
        assert TaskStatus.COMPLETED == "completed"


# ---------------------------------------------------------------------------
# Mock spawner for testing
# ---------------------------------------------------------------------------


class _MockSpawner:
    """Minimal spawner that records calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._counter = 0

    async def aspawn(
        self,
        pipeline_name: str,
        initial_input: dict,
        *,
        ref_id: str | None = None,
        ref_type: str | None = None,
        parent_run_id: str | None = None,
        wait: bool = False,
        timeout: float | None = None,
    ) -> str:
        self._counter += 1
        self.calls.append((pipeline_name, initial_input))
        return f"run-{self._counter}"

    async def aspawn_many(self, *_args, **_kwargs) -> list[str]:
        return []

    async def await_result(self, *_args, **_kwargs) -> dict:
        return {}

    async def acancel(self, *_args, **_kwargs) -> None:
        pass


# ---------------------------------------------------------------------------
# Scheduler lib — tool surface
# ---------------------------------------------------------------------------


class TestSchedulerIsLib:
    def test_is_hexdag_lib(self) -> None:
        assert isinstance(Scheduler(), HexDAGLib)

    def test_exposed_tools(self) -> None:
        sched = Scheduler()
        tools = sched.get_tools()
        assert "aschedule_once" in tools
        assert "aschedule_recurring" in tools
        assert "acancel" in tools
        assert "alist_scheduled" in tools
        assert len(tools) == 4


class TestScheduleOnce:
    @pytest.mark.asyncio()
    async def test_schedule_returns_task_info(self) -> None:
        sched = Scheduler()
        result = await sched.aschedule_once("my_pipeline", {"key": "val"}, delay_seconds=100)
        assert result["pipeline_name"] == "my_pipeline"
        assert result["schedule_type"] == "once"
        assert result["status"] == "pending"
        assert result["task_id"] is not None
        await sched.ateardown()

    @pytest.mark.asyncio()
    async def test_schedule_once_executes_after_delay(self) -> None:
        spawner = _MockSpawner()
        sched = Scheduler(spawner=spawner)
        await sched.aschedule_once("p1", {"x": 1}, delay_seconds=0.05)
        await asyncio.sleep(0.15)
        assert len(spawner.calls) == 1
        assert spawner.calls[0] == ("p1", {"x": 1})
        await sched.ateardown()

    @pytest.mark.asyncio()
    async def test_schedule_once_without_spawner_is_noop(self) -> None:
        sched = Scheduler()
        await sched.aschedule_once("p1", delay_seconds=0.05)
        await asyncio.sleep(0.15)
        # Should not raise — just logs
        tasks = await sched.alist_scheduled()
        assert len(tasks) == 1
        await sched.ateardown()

    @pytest.mark.asyncio()
    async def test_schedule_once_completes_status(self) -> None:
        spawner = _MockSpawner()
        sched = Scheduler(spawner=spawner)
        result = await sched.aschedule_once("p1", delay_seconds=0.05)
        task_id = result["task_id"]
        await asyncio.sleep(0.15)
        tasks = await sched.alist_scheduled()
        task = next(t for t in tasks if t["task_id"] == task_id)
        assert task["status"] == "completed"
        await sched.ateardown()


class TestScheduleRecurring:
    @pytest.mark.asyncio()
    async def test_schedule_recurring_returns_task_info(self) -> None:
        sched = Scheduler()
        result = await sched.aschedule_recurring("p1", interval_seconds=100)
        assert result["schedule_type"] == "recurring"
        assert result["interval_seconds"] == 100
        await sched.ateardown()

    @pytest.mark.asyncio()
    async def test_recurring_executes_multiple_times(self) -> None:
        spawner = _MockSpawner()
        sched = Scheduler(spawner=spawner)
        await sched.aschedule_recurring("p1", {"x": 1}, interval_seconds=0.05)
        await asyncio.sleep(0.18)
        # Should have run at least 2 times (0.05, 0.10, maybe 0.15)
        assert len(spawner.calls) >= 2
        await sched.ateardown()


class TestCancel:
    @pytest.mark.asyncio()
    async def test_cancel_existing(self) -> None:
        sched = Scheduler()
        result = await sched.aschedule_once("p1", delay_seconds=100)
        task_id = result["task_id"]
        cancel_result = await sched.acancel(task_id)
        assert cancel_result["cancelled"] is True
        await sched.ateardown()

    @pytest.mark.asyncio()
    async def test_cancel_nonexistent(self) -> None:
        sched = Scheduler()
        result = await sched.acancel("missing")
        assert result["cancelled"] is False

    @pytest.mark.asyncio()
    async def test_cancel_prevents_execution(self) -> None:
        spawner = _MockSpawner()
        sched = Scheduler(spawner=spawner)
        result = await sched.aschedule_once("p1", {"x": 1}, delay_seconds=0.1)
        await sched.acancel(result["task_id"])
        await asyncio.sleep(0.2)
        assert len(spawner.calls) == 0
        await sched.ateardown()

    @pytest.mark.asyncio()
    async def test_cancel_recurring_stops_execution(self) -> None:
        spawner = _MockSpawner()
        sched = Scheduler(spawner=spawner)
        result = await sched.aschedule_recurring("p1", interval_seconds=0.05)
        # Let it run once
        await asyncio.sleep(0.08)
        count_before = len(spawner.calls)
        await sched.acancel(result["task_id"])
        await asyncio.sleep(0.15)
        # No additional runs after cancel
        assert len(spawner.calls) <= count_before + 1
        await sched.ateardown()


class TestListScheduled:
    @pytest.mark.asyncio()
    async def test_list_all(self) -> None:
        sched = Scheduler()
        await sched.aschedule_once("p1", delay_seconds=100)
        await sched.aschedule_once("p2", delay_seconds=100)
        tasks = await sched.alist_scheduled()
        assert len(tasks) == 2
        await sched.ateardown()

    @pytest.mark.asyncio()
    async def test_list_by_ref_id(self) -> None:
        sched = Scheduler()
        await sched.aschedule_once("p1", delay_seconds=100, ref_id="order-1")
        await sched.aschedule_once("p2", delay_seconds=100, ref_id="order-2")
        await sched.aschedule_once("p3", delay_seconds=100, ref_id="order-1")
        tasks = await sched.alist_scheduled(ref_id="order-1")
        assert len(tasks) == 2
        await sched.ateardown()

    @pytest.mark.asyncio()
    async def test_list_empty(self) -> None:
        sched = Scheduler()
        assert await sched.alist_scheduled() == []


class TestTeardown:
    @pytest.mark.asyncio()
    async def test_teardown_cancels_all(self) -> None:
        spawner = _MockSpawner()
        sched = Scheduler(spawner=spawner)
        await sched.aschedule_recurring("p1", interval_seconds=0.05)
        await sched.aschedule_recurring("p2", interval_seconds=0.05)
        await sched.ateardown()
        await asyncio.sleep(0.15)
        # After teardown, no more executions
        assert len(spawner.calls) == 0
