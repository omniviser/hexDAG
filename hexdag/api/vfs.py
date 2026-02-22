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

if TYPE_CHECKING:
    from hexdag.kernel.ports.vfs import VFS


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
