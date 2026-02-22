"""PipelineSpawner port — fork/exec for child pipelines.

This port allows pipelines (and their agent nodes) to spawn other
pipelines, wait for results, or cancel running pipelines.  It is
the hexDAG equivalent of ``fork``/``exec`` in Linux.

Adapters
--------
- ``LocalPipelineSpawner`` — runs child pipelines in the same process
  using :class:`~hexdag.kernel.pipeline_runner.PipelineRunner`.
- Future: ``DistributedPipelineSpawner`` — submits to a task queue
  (Celery, Temporal, etc.).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PipelineSpawner(Protocol):
    """Port for spawning and managing child pipeline runs."""

    @abstractmethod
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
        """Spawn a new pipeline run.

        Args
        ----
            pipeline_name: Name or path of the pipeline to run.
            initial_input: Input data for the pipeline.
            ref_id: Business reference ID (e.g. order ID, customer ID).
            ref_type: Type of the business reference (e.g. "order", "customer").
            parent_run_id: ID of the parent run for hierarchical tracking.
            wait: If True, block until the pipeline completes.
            timeout: Maximum wait time in seconds (only when ``wait=True``).

        Returns
        -------
            The run ID of the spawned pipeline.
        """
        ...

    @abstractmethod
    async def aspawn_many(
        self,
        pipeline_name: str,
        inputs: list[dict[str, Any]],
        *,
        ref_id: str | None = None,
        ref_type: str | None = None,
        parent_run_id: str | None = None,
    ) -> list[str]:
        """Spawn multiple pipeline runs concurrently.

        Args
        ----
            pipeline_name: Name or path of the pipeline to run.
            inputs: List of input data dicts, one per run.
            ref_id: Shared business reference ID.
            ref_type: Type of the business reference.
            parent_run_id: ID of the parent run.

        Returns
        -------
            List of run IDs for the spawned pipelines.
        """
        ...

    @abstractmethod
    async def await_result(self, run_id: str, timeout: float | None = None) -> dict[str, Any]:
        """Wait for a pipeline run to complete and return its results.

        Args
        ----
            run_id: The run ID to wait for.
            timeout: Maximum wait time in seconds.

        Returns
        -------
            The pipeline results dict.

        Raises
        ------
        TimeoutError
            If the timeout is exceeded.
        """
        ...

    @abstractmethod
    async def acancel(self, run_id: str) -> None:
        """Cancel a running pipeline.

        Args
        ----
            run_id: The run ID to cancel.
        """
        ...


__all__ = ["PipelineSpawner"]
