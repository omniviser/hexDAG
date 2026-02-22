"""VFS API — unified path-based introspection.

Provides functions that both the MCP server and hexdag-studio
REST API consume for path-based entity access.

MCP server usage::

    from hexdag.api import vfs

    @mcp.tool()
    async def read_vfs_path(path: str) -> str:
        return await vfs.read_path(vfs_instance, path)

Studio REST API::

    @router.get("/vfs/read")
    async def read_vfs(path: str) -> str:
        return await vfs.read_path(vfs_instance, path)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hexdag.drivers.vfs import LocalVFS
from hexdag.drivers.vfs.providers import (
    LibProvider,
    ProcEntitiesProvider,
    ProcRunsProvider,
    ProcScheduledProvider,
)

if TYPE_CHECKING:
    from hexdag.kernel.ports.vfs import VFS
    from hexdag.stdlib.lib.entity_state import EntityState
    from hexdag.stdlib.lib.process_registry import ProcessRegistry
    from hexdag.stdlib.lib.scheduler import Scheduler


def create_vfs(
    *,
    registry: ProcessRegistry | None = None,
    scheduler: Scheduler | None = None,
    entity_state: EntityState | None = None,
) -> VFS:
    """Create a VFS instance with all available providers mounted.

    Always mounts ``/lib/`` (component discovery). Conditionally mounts
    ``/proc/*`` providers when their dependencies are supplied.

    Parameters
    ----------
    registry : ProcessRegistry | None
        Mount ``/proc/runs/`` if provided.
    scheduler : Scheduler | None
        Mount ``/proc/scheduled/`` if provided.
    entity_state : EntityState | None
        Mount ``/proc/entities/`` if provided.

    Returns
    -------
    VFS
        A fully configured :class:`~hexdag.drivers.vfs.LocalVFS`.
    """

    vfs = LocalVFS()
    vfs.mount("/lib/", LibProvider())

    if registry is not None:
        vfs.mount("/proc/runs/", ProcRunsProvider(registry))
    if scheduler is not None:
        vfs.mount("/proc/scheduled/", ProcScheduledProvider(scheduler))
    if entity_state is not None:
        vfs.mount("/proc/entities/", ProcEntitiesProvider(entity_state))

    return vfs


async def read_path(vfs: VFS, path: str) -> str:
    """Read the content at a VFS path.

    Args
    ----
        vfs: VFS instance.
        path: Absolute VFS path (e.g. ``/lib/nodes/llm_node``).

    Returns
    -------
        JSON-serialized content string.
    """
    return await vfs.aread(path)


async def list_path(vfs: VFS, path: str) -> list[dict[str, Any]]:
    """List entries in a VFS directory.

    Args
    ----
        vfs: VFS instance.
        path: Absolute VFS path (e.g. ``/lib/nodes/``).

    Returns
    -------
        List of entry dicts with ``name``, ``entry_type``, ``path``.
    """
    entries = await vfs.alist(path)
    return [e.model_dump() for e in entries]


async def stat_path(vfs: VFS, path: str) -> dict[str, Any]:
    """Get metadata about a VFS path.

    Args
    ----
        vfs: VFS instance.
        path: Absolute VFS path (e.g. ``/proc/runs/abc123``).

    Returns
    -------
        Metadata dict (description, status, capabilities, tags, etc.).
    """
    result = await vfs.astat(path)
    return result.model_dump(exclude_none=True)
