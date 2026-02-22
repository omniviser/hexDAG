"""VFS provider for /proc/scheduled/ — scheduled task introspection.

Delegates to :class:`~hexdag.stdlib.lib.scheduler.Scheduler`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from hexdag.kernel.domain.vfs import DirEntry, EntryType, StatResult
from hexdag.kernel.exceptions import VFSError

if TYPE_CHECKING:
    from hexdag.stdlib.lib.scheduler import Scheduler


class ProcScheduledProvider:
    """VFS provider for /proc/scheduled/ — scheduled task introspection."""

    def __init__(self, scheduler: Scheduler) -> None:
        self._scheduler = scheduler

    async def read(self, relative_path: str) -> str:
        """Read a scheduled task's details.

        Paths
        -----
        - ``<task_id>`` → JSON of task details
        """
        path = relative_path.strip("/")
        if not path:
            raise VFSError("/proc/scheduled/", "cannot read directory; use readdir")

        tasks: list[dict[str, Any]] = await self._scheduler.alist_scheduled()
        for task in tasks:
            if task.get("task_id") == path:
                return json.dumps(task, indent=2, default=str)

        raise VFSError(f"/proc/scheduled/{path}", f"task '{path}' not found")

    async def readdir(self, relative_path: str) -> list[DirEntry]:
        """List scheduled tasks.

        Paths
        -----
        - ``""`` → all scheduled tasks
        """
        path = relative_path.strip("/")
        if path:
            raise VFSError(f"/proc/scheduled/{path}", "not a directory")

        tasks: list[dict[str, Any]] = await self._scheduler.alist_scheduled()
        return [
            DirEntry(
                name=t["task_id"],
                entry_type=EntryType.FILE,
                path=f"/proc/scheduled/{t['task_id']}",
            )
            for t in tasks
        ]

    async def stat(self, relative_path: str) -> StatResult:
        """Get metadata about a scheduled task."""
        path = relative_path.strip("/")

        if not path:
            tasks: list[dict[str, Any]] = await self._scheduler.alist_scheduled()
            return StatResult(
                path="/proc/scheduled",
                entry_type=EntryType.DIRECTORY,
                description="Scheduled pipeline tasks",
                child_count=len(tasks),
                capabilities=["read"],
            )

        tasks = await self._scheduler.alist_scheduled()
        for task in tasks:
            if task.get("task_id") == path:
                tags: dict[str, str] = {}
                if task.get("pipeline_name"):
                    tags["pipeline_name"] = task["pipeline_name"]
                if task.get("ref_id"):
                    tags["ref_id"] = task["ref_id"]
                return StatResult(
                    path=f"/proc/scheduled/{path}",
                    entry_type=EntryType.FILE,
                    entity_type="scheduled_task",
                    description=f"Scheduled '{task.get('pipeline_name', 'unknown')}'",
                    status=task.get("status", "active"),
                    capabilities=["read"],
                    tags=tags,
                )

        raise VFSError(f"/proc/scheduled/{path}", f"task '{path}' not found")
