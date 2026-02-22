"""Scheduler lib — delayed and recurring pipeline execution.

Uses ``asyncio`` timers to trigger pipeline spawns at specified times.
Designed so the API maps cleanly to APScheduler if users want a
production-grade backend.

Usage in YAML::

    spec:
      libs:
        scheduler:
          class: hexdag.stdlib.lib.Scheduler

Programmatic::

    from hexdag.stdlib.lib.scheduler import Scheduler

    scheduler = Scheduler(spawner=my_spawner)
    task_id = await scheduler.aschedule_once("my_pipeline", {"key": "val"}, delay_seconds=60)
    task_id = await scheduler.aschedule_recurring(
        "my_pipeline", {"key": "val"}, interval_seconds=300,
    )
    await scheduler.acancel(task_id)
    tasks = await scheduler.alist_scheduled()
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from hexdag.kernel.domain.scheduled_task import ScheduleType, TaskStatus
from hexdag.stdlib.lib_base import HexDAGLib

if TYPE_CHECKING:
    from hexdag.kernel.domain.scheduled_task import ScheduledTask
    from hexdag.kernel.ports.pipeline_spawner import PipelineSpawner

logger = logging.getLogger(__name__)


class Scheduler(HexDAGLib):
    """In-memory task scheduler with asyncio timers.

    Exposed tools
    -------------
    - ``aschedule_once(pipeline_name, input, delay_seconds)`` — one-shot delayed execution
    - ``aschedule_recurring(pipeline_name, input, interval_seconds)`` — repeating execution
    - ``acancel(task_id)`` — cancel a scheduled task
    - ``alist_scheduled(ref_id?)`` — list scheduled tasks
    """

    def __init__(self, spawner: PipelineSpawner | None = None) -> None:
        """Initialise the scheduler with an optional pipeline spawner."""
        self._spawner = spawner
        self._tasks: dict[str, ScheduledTask] = {}
        self._handles: dict[str, asyncio.Task[None]] = {}

    async def ateardown(self) -> None:
        """Cancel all pending asyncio tasks on shutdown."""
        for handle in self._handles.values():
            handle.cancel()
        self._handles.clear()

    # ------------------------------------------------------------------
    # Agent-callable tools
    # ------------------------------------------------------------------

    async def aschedule_once(
        self,
        pipeline_name: str,
        initial_input: dict[str, Any] | None = None,
        delay_seconds: float = 0.0,
        ref_id: str | None = None,
        ref_type: str | None = None,
    ) -> dict[str, Any]:
        """Schedule a pipeline to run once after a delay.

        Args
        ----
            pipeline_name: Name of the pipeline to execute.
            initial_input: Input data for the pipeline.
            delay_seconds: Seconds to wait before execution.
            ref_id: Business reference ID.
            ref_type: Type of business reference.

        Returns
        -------
            Dict with task_id and schedule details.
        """
        from hexdag.kernel.domain.scheduled_task import ScheduledTask

        task_id = str(uuid4())
        now = time.time()
        task = ScheduledTask(
            task_id=task_id,
            pipeline_name=pipeline_name,
            schedule_type=ScheduleType.ONCE,
            initial_input=initial_input or {},
            delay_seconds=delay_seconds,
            ref_id=ref_id,
            ref_type=ref_type,
            next_run_at=now + delay_seconds,
        )
        self._tasks[task_id] = task
        self._handles[task_id] = asyncio.create_task(self._run_once(task_id, delay_seconds))
        return _task_to_dict(task)

    async def aschedule_recurring(
        self,
        pipeline_name: str,
        initial_input: dict[str, Any] | None = None,
        interval_seconds: float = 60.0,
        ref_id: str | None = None,
        ref_type: str | None = None,
    ) -> dict[str, Any]:
        """Schedule a pipeline to run on a recurring interval.

        Args
        ----
            pipeline_name: Name of the pipeline to execute.
            initial_input: Input data for the pipeline.
            interval_seconds: Seconds between executions.
            ref_id: Business reference ID.
            ref_type: Type of business reference.

        Returns
        -------
            Dict with task_id and schedule details.
        """
        from hexdag.kernel.domain.scheduled_task import ScheduledTask

        task_id = str(uuid4())
        now = time.time()
        task = ScheduledTask(
            task_id=task_id,
            pipeline_name=pipeline_name,
            schedule_type=ScheduleType.RECURRING,
            initial_input=initial_input or {},
            interval_seconds=interval_seconds,
            ref_id=ref_id,
            ref_type=ref_type,
            next_run_at=now + interval_seconds,
        )
        self._tasks[task_id] = task
        self._handles[task_id] = asyncio.create_task(self._run_recurring(task_id, interval_seconds))
        return _task_to_dict(task)

    async def acancel(self, task_id: str) -> dict[str, Any]:
        """Cancel a scheduled task.

        Args
        ----
            task_id: The task ID to cancel.

        Returns
        -------
            Dict with task_id and updated status.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return {"task_id": task_id, "cancelled": False, "error": "not found"}

        handle = self._handles.pop(task_id, None)
        if handle is not None:
            handle.cancel()

        task.status = TaskStatus.CANCELLED
        return {"task_id": task_id, "cancelled": True}

    async def alist_scheduled(self, ref_id: str | None = None) -> list[dict[str, Any]]:
        """List scheduled tasks, optionally filtered by ref_id.

        Args
        ----
            ref_id: Optional business reference ID to filter by.

        Returns
        -------
            List of task detail dicts.
        """
        tasks = list(self._tasks.values())
        if ref_id:
            tasks = [t for t in tasks if t.ref_id == ref_id]
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return [_task_to_dict(t) for t in tasks]

    # ------------------------------------------------------------------
    # Internal scheduling logic
    # ------------------------------------------------------------------

    async def _run_once(self, task_id: str, delay: float) -> None:
        """Execute a one-shot task after delay."""
        await asyncio.sleep(delay)
        task = self._tasks.get(task_id)
        if task is None or task.status == TaskStatus.CANCELLED:
            return
        await self._execute_task(task)
        task.status = TaskStatus.COMPLETED
        self._handles.pop(task_id, None)

    async def _run_recurring(self, task_id: str, interval: float) -> None:
        """Execute a recurring task on interval."""
        try:
            while True:
                await asyncio.sleep(interval)
                task = self._tasks.get(task_id)
                if task is None or task.status == TaskStatus.CANCELLED:
                    return
                await self._execute_task(task)
                task.next_run_at = time.time() + interval
        except asyncio.CancelledError:
            return

    async def _execute_task(self, task: ScheduledTask) -> None:
        """Run the pipeline via spawner (if available)."""
        task.status = TaskStatus.RUNNING
        task.last_run_at = time.time()
        task.run_count += 1

        if self._spawner is None:
            logger.debug(
                "Scheduler: no spawner configured, skipping execution of %s",
                task.pipeline_name,
            )
            return

        try:
            run_id = await self._spawner.aspawn(
                task.pipeline_name,
                task.initial_input,
                ref_id=task.ref_id,
                ref_type=task.ref_type,
            )
            task.last_run_id = run_id
            task.error = None
        except Exception as exc:  # noqa: BLE001
            task.status = TaskStatus.FAILED
            task.error = str(exc)
            logger.warning(
                "Scheduler: failed to spawn %s: %s",
                task.pipeline_name,
                exc,
            )


def _task_to_dict(task: ScheduledTask) -> dict[str, Any]:
    """Serialise a ScheduledTask to a plain dict for tool output."""
    return {
        "task_id": task.task_id,
        "pipeline_name": task.pipeline_name,
        "schedule_type": task.schedule_type,
        "status": task.status,
        "delay_seconds": task.delay_seconds,
        "interval_seconds": task.interval_seconds,
        "ref_id": task.ref_id,
        "ref_type": task.ref_type,
        "created_at": task.created_at,
        "next_run_at": task.next_run_at,
        "last_run_at": task.last_run_at,
        "run_count": task.run_count,
        "last_run_id": task.last_run_id,
    }
