"""ProcessRegistry lib — tracks pipeline runs like ``ps`` in Linux.

The ProcessRegistry is the first system lib.  It maintains an in-memory
index of pipeline runs and exposes CRUD + query operations as agent tools.

Population happens via :class:`ProcessRegistryObserver`, which listens
to ``PipelineStarted`` / ``PipelineCompleted`` events.

Usage in YAML::

    spec:
      libs:
        process_registry:
          class: hexdag.stdlib.lib.ProcessRegistry
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hexdag.kernel.domain.pipeline_run import pipeline_run_to_storage
from hexdag.stdlib.lib_base import HexDAGLib

if TYPE_CHECKING:
    from hexdag.kernel.domain.pipeline_run import PipelineRun, RunStatus
    from hexdag.kernel.ports.data_store import SupportsCollectionStorage

_COLLECTION = "pipeline_runs"


class ProcessRegistry(HexDAGLib):
    """In-memory registry of pipeline runs with optional persistent storage.

    Exposed tools
    -------------
    - ``aget(run_id)`` — get a single run
    - ``alist(status?, limit?)`` — list runs with optional filter
    - ``alist_by_ref(ref_id, ref_type?)`` — query by business ref
    """

    def __init__(self, storage: SupportsCollectionStorage | None = None) -> None:
        """Initialise the run store.

        Args
        ----
            storage: Optional persistent backend.  When ``None`` (default),
                all data lives only in memory.
        """
        self._storage = storage
        self._runs: dict[str, PipelineRun] = {}

    # ------------------------------------------------------------------
    # Internal mutation API (called by observer, not exposed as tools)
    # ------------------------------------------------------------------

    async def register(self, run: PipelineRun) -> None:
        """Register a new pipeline run (called by observer)."""
        self._runs[run.run_id] = run
        if self._storage is not None:
            await self._storage.asave(_COLLECTION, run.run_id, pipeline_run_to_storage(run))

    async def update_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        duration_ms: float | None = None,
        node_results: dict[str, Any] | None = None,
        error: str | None = None,
        completed_at: float | None = None,
        started_at: float | None = None,
    ) -> None:
        """Update a run's status (called by observer)."""
        if run_id not in self._runs:
            return
        run = self._runs[run_id]
        run.status = status
        if duration_ms is not None:
            run.duration_ms = duration_ms
        if node_results is not None:
            run.node_results = node_results
        if error is not None:
            run.error = error
        if completed_at is not None:
            run.completed_at = completed_at
        if started_at is not None:
            run.started_at = started_at

        if self._storage is not None:
            await self._storage.asave(_COLLECTION, run_id, pipeline_run_to_storage(run))

    # ------------------------------------------------------------------
    # Agent-callable tools (auto-exposed via HexDAGLib.get_tools)
    # ------------------------------------------------------------------

    async def aget(self, run_id: str) -> dict[str, Any] | None:
        """Get a pipeline run by ID.

        Args
        ----
            run_id: The pipeline run ID.

        Returns
        -------
            Run details dict, or None if not found.
        """
        run = self._runs.get(run_id)
        if run is not None:
            return _run_to_dict(run)
        if self._storage is not None:
            data = await self._storage.aload(_COLLECTION, run_id)
            if data is not None:
                return _storage_to_output(data)
        return None

    async def alist(self, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """List pipeline runs, optionally filtered by status.

        Args
        ----
            status: Filter by run status (created/running/completed/failed/cancelled).
            limit: Maximum number of results (default 50).

        Returns
        -------
            List of run detail dicts, newest first.
        """
        if self._storage is not None:
            filters = {"status": status} if status else None
            docs = await self._storage.aquery(_COLLECTION, filters)
            docs.sort(key=lambda d: d.get("created_at", 0), reverse=True)
            return [_storage_to_output(d) for d in docs[:limit]]

        runs = list(self._runs.values())
        if status:
            runs = [r for r in runs if r.status == status]
        runs.sort(key=lambda r: r.created_at, reverse=True)
        return [_run_to_dict(r) for r in runs[:limit]]

    async def alist_by_ref(self, ref_id: str, ref_type: str | None = None) -> list[dict[str, Any]]:
        """List pipeline runs linked to a business reference.

        Args
        ----
            ref_id: The business reference ID (e.g. order ID).
            ref_type: Optional type filter (e.g. "order", "customer").

        Returns
        -------
            List of matching runs, newest first.
        """
        if self._storage is not None:
            filters: dict[str, Any] = {"ref_id": ref_id}
            if ref_type:
                filters["ref_type"] = ref_type
            docs = await self._storage.aquery(_COLLECTION, filters)
            docs.sort(key=lambda d: d.get("created_at", 0), reverse=True)
            return [_storage_to_output(d) for d in docs]

        runs = [r for r in self._runs.values() if r.ref_id == ref_id]
        if ref_type:
            runs = [r for r in runs if r.ref_type == ref_type]
        runs.sort(key=lambda r: r.created_at, reverse=True)
        return [_run_to_dict(r) for r in runs]


def _run_to_dict(run: PipelineRun) -> dict[str, Any]:
    """Serialise a PipelineRun to a plain dict for tool output."""
    return {
        "run_id": run.run_id,
        "pipeline_name": run.pipeline_name,
        "status": run.status,
        "ref_id": run.ref_id,
        "ref_type": run.ref_type,
        "parent_run_id": run.parent_run_id,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "duration_ms": run.duration_ms,
        "error": run.error,
    }


def _storage_to_output(data: dict[str, Any]) -> dict[str, Any]:
    """Convert a storage document to tool output format."""
    return {
        "run_id": data.get("run_id"),
        "pipeline_name": data.get("pipeline_name"),
        "status": data.get("status"),
        "ref_id": data.get("ref_id"),
        "ref_type": data.get("ref_type"),
        "parent_run_id": data.get("parent_run_id"),
        "created_at": data.get("created_at"),
        "started_at": data.get("started_at"),
        "completed_at": data.get("completed_at"),
        "duration_ms": data.get("duration_ms"),
        "error": data.get("error"),
    }
