"""VFS provider for /proc/runs/ — pipeline run introspection.

Delegates to :class:`~hexdag.stdlib.lib.process_registry.ProcessRegistry`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from hexdag.kernel.domain.vfs import DirEntry, EntryType, StatResult
from hexdag.kernel.exceptions import VFSError

if TYPE_CHECKING:
    from hexdag.stdlib.lib.process_registry import ProcessRegistry


class ProcRunsProvider:
    """VFS provider for /proc/runs/ — pipeline run introspection."""

    def __init__(self, registry: ProcessRegistry) -> None:
        self._registry = registry

    async def read(self, relative_path: str) -> str:
        """Read a run's details.

        Paths
        -----
        - ``<run_id>`` → JSON of run details
        - ``<run_id>/status`` → just the status string
        """
        path = relative_path.strip("/")
        if not path:
            raise VFSError("/proc/runs/", "cannot read directory; use readdir")

        parts = path.split("/")
        run_id = parts[0]
        run = await self._registry.aget(run_id)
        if run is None:
            raise VFSError(f"/proc/runs/{path}", f"run '{run_id}' not found")

        if len(parts) == 1:
            return json.dumps(run, indent=2, default=str)

        if len(parts) == 2 and parts[1] == "status":
            return json.dumps(run.get("status", "unknown"))

        raise VFSError(f"/proc/runs/{path}", "path not found")

    async def readdir(self, relative_path: str) -> list[DirEntry]:
        """List pipeline runs.

        Paths
        -----
        - ``""`` → all runs
        """
        path = relative_path.strip("/")
        if path:
            # A specific run_id is a file, not a directory
            raise VFSError(f"/proc/runs/{path}", "not a directory")

        runs: list[dict[str, Any]] = await self._registry.alist()
        return [
            DirEntry(
                name=r["run_id"],
                entry_type=EntryType.FILE,
                path=f"/proc/runs/{r['run_id']}",
            )
            for r in runs
        ]

    async def stat(self, relative_path: str) -> StatResult:
        """Get metadata about a run."""
        path = relative_path.strip("/")

        if not path:
            runs: list[dict[str, Any]] = await self._registry.alist()
            return StatResult(
                path="/proc/runs",
                entry_type=EntryType.DIRECTORY,
                description="Active and completed pipeline runs",
                child_count=len(runs),
                capabilities=["read"],
            )

        parts = path.split("/")
        run_id = parts[0]
        run = await self._registry.aget(run_id)
        if run is None:
            raise VFSError(f"/proc/runs/{path}", f"run '{run_id}' not found")

        tags: dict[str, str] = {}
        if run.get("pipeline_name"):
            tags["pipeline_name"] = run["pipeline_name"]
        if run.get("ref_id"):
            tags["ref_id"] = run["ref_id"]

        return StatResult(
            path=f"/proc/runs/{run_id}",
            entry_type=EntryType.FILE,
            entity_type="run",
            description=f"Pipeline '{run.get('pipeline_name', 'unknown')}' run",
            status=run.get("status"),
            capabilities=["read"],
            tags=tags,
        )
