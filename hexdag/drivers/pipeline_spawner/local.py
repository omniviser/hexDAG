"""Local pipeline spawner — runs child pipelines in-process.

This adapter uses :class:`~hexdag.kernel.pipeline_runner.PipelineRunner`
to execute child pipelines within the same Python process.  It is the
default spawner for development and single-process deployments.
"""

from __future__ import annotations

import asyncio
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from hexdag.kernel.logging import get_logger

if TYPE_CHECKING:
    from hexdag.kernel.pipeline_runner import PipelineRunner

logger = get_logger(__name__)


class RunStatus(StrEnum):
    """Status of a spawned pipeline run."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LocalPipelineSpawner:
    """PipelineSpawner adapter that runs pipelines locally via PipelineRunner.

    Parameters
    ----------
    runner : PipelineRunner
        The pipeline runner to use for executing child pipelines.
    pipeline_dir : str | None
        Base directory for resolving pipeline names to files.
        If None, pipeline_name is treated as a full path.
    """

    def __init__(
        self,
        runner: PipelineRunner,
        pipeline_dir: str | None = None,
    ) -> None:
        self._runner = runner
        self._pipeline_dir = pipeline_dir
        self._runs: dict[str, _RunState] = {}
        self._tasks: dict[str, asyncio.Task[dict[str, Any]]] = {}

    async def aspawn(
        self,
        pipeline_name: str,
        initial_input: dict[str, Any],
        *,
        ref_id: str | None = None,
        ref_type: str | None = None,
        parent_run_id: str | None = None,
        wait: bool = False,
        timeout: float | None = None,
    ) -> str:
        """Spawn a pipeline run.

        If ``wait=True``, blocks until the pipeline completes (or times out).
        Otherwise, runs in the background and returns immediately.
        """
        run_id = str(uuid4())
        self._runs[run_id] = _RunState(
            run_id=run_id,
            pipeline_name=pipeline_name,
            status=RunStatus.CREATED,
            ref_id=ref_id,
            ref_type=ref_type,
            parent_run_id=parent_run_id,
        )

        if wait:
            result = await self._execute(run_id, pipeline_name, initial_input, timeout)
            self._runs[run_id].result = result
            self._runs[run_id].status = RunStatus.COMPLETED
            return run_id

        # Fire-and-forget (background task)
        task = asyncio.create_task(self._execute(run_id, pipeline_name, initial_input, timeout))
        self._tasks[run_id] = task
        task.add_done_callback(lambda t: self._on_task_done(run_id, t))
        self._runs[run_id].status = RunStatus.RUNNING

        logger.debug(f"Spawned pipeline '{pipeline_name}' as run {run_id}")
        return run_id

    async def aspawn_many(
        self,
        pipeline_name: str,
        inputs: list[dict[str, Any]],
        *,
        ref_id: str | None = None,
        ref_type: str | None = None,
        parent_run_id: str | None = None,
    ) -> list[str]:
        """Spawn multiple pipeline runs concurrently."""
        run_ids: list[str] = []
        for input_data in inputs:
            run_id = await self.aspawn(
                pipeline_name,
                input_data,
                ref_id=ref_id,
                ref_type=ref_type,
                parent_run_id=parent_run_id,
            )
            run_ids.append(run_id)
        return run_ids

    async def await_result(self, run_id: str, timeout: float | None = None) -> dict[str, Any]:
        """Wait for a spawned pipeline to complete."""
        if run_id not in self._runs:
            msg = f"Unknown run ID: {run_id}"
            raise KeyError(msg)

        state = self._runs[run_id]

        # Already completed
        if state.status == RunStatus.COMPLETED and state.result is not None:
            return state.result

        if state.status == RunStatus.FAILED:
            msg = f"Run {run_id} failed: {state.error}"
            raise RuntimeError(msg)

        # Wait for the background task
        if run_id in self._tasks:
            try:
                return await asyncio.wait_for(self._tasks[run_id], timeout=timeout)
            except TimeoutError:
                msg = f"Timed out waiting for run {run_id}"
                raise TimeoutError(msg) from None

        msg = f"Run {run_id} has no associated task"
        raise RuntimeError(msg)

    async def acancel(self, run_id: str) -> None:
        """Cancel a running pipeline."""
        if run_id in self._tasks and not self._tasks[run_id].done():
            self._tasks[run_id].cancel()
            self._runs[run_id].status = RunStatus.CANCELLED
            logger.debug(f"Cancelled run {run_id}")

    def get_run_status(self, run_id: str) -> RunStatus:
        """Get the status of a pipeline run (non-protocol utility)."""
        if run_id not in self._runs:
            msg = f"Unknown run ID: {run_id}"
            raise KeyError(msg)
        return self._runs[run_id].status

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _execute(
        self,
        run_id: str,
        pipeline_name: str,
        initial_input: dict[str, Any],
        timeout: float | None,
    ) -> dict[str, Any]:
        """Execute the pipeline via PipelineRunner."""
        pipeline_path = pipeline_name
        if self._pipeline_dir:
            pipeline_path = f"{self._pipeline_dir}/{pipeline_name}"

        self._runs[run_id].status = RunStatus.RUNNING

        coro = self._runner.run(pipeline_path, input_data=initial_input)
        if timeout:
            return await asyncio.wait_for(coro, timeout=timeout)
        return await coro

    def _on_task_done(self, run_id: str, task: asyncio.Task[dict[str, Any]]) -> None:
        """Callback when a background task finishes."""
        if task.cancelled():
            self._runs[run_id].status = RunStatus.CANCELLED
        elif task.exception():
            self._runs[run_id].status = RunStatus.FAILED
            self._runs[run_id].error = str(task.exception())
        else:
            self._runs[run_id].status = RunStatus.COMPLETED
            self._runs[run_id].result = task.result()
        self._tasks.pop(run_id, None)


class _RunState:
    """Internal tracking state for a pipeline run."""

    __slots__ = (
        "error",
        "parent_run_id",
        "pipeline_name",
        "ref_id",
        "ref_type",
        "result",
        "run_id",
        "status",
    )

    def __init__(
        self,
        run_id: str,
        pipeline_name: str,
        status: RunStatus,
        ref_id: str | None = None,
        ref_type: str | None = None,
        parent_run_id: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.pipeline_name = pipeline_name
        self.status = status
        self.ref_id = ref_id
        self.ref_type = ref_type
        self.parent_run_id = parent_run_id
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
