"""VFSTools — agent-callable VFS operations.

Extends :class:`~hexdag.kernel.service.Service` with ``@tool``-decorated
methods.  Gives any agent the ability to introspect the running system
through path-based queries.

YAML configuration::

    spec:
      services:
        vfs:
          class: hexdag.stdlib.lib.vfs_tools.VFSTools
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hexdag.kernel.service import Service, tool

if TYPE_CHECKING:
    from hexdag.kernel.ports.vfs import VFS


class VFSTools(Service):
    """Agent-callable VFS operations.

    Gives agents the ability to introspect the running hexDAG system
    through path-based queries. Every entity — nodes, adapters, runs,
    scheduled tasks, entity states — is addressable by path.

    Exposed tools
    -------------
    - ``aread_path(path)`` — read content at a VFS path
    - ``alist_directory(path)`` — list entries in a VFS directory
    - ``astat_path(path)`` — get metadata about a VFS path
    """

    def __init__(self, vfs: VFS) -> None:
        """Initialise with the given VFS port."""
        self._vfs = vfs

    @tool
    async def aread_path(self, path: str) -> str:
        """Read a VFS path and return its content as JSON.

        Use this to inspect entities: ``/lib/nodes/llm_node``,
        ``/proc/runs/<run_id>``, etc.

        Args
        ----
            path: Absolute VFS path (e.g. ``/lib/nodes/llm_node``).

        Returns
        -------
            JSON-serialized content string.
        """
        return await self._vfs.aread(path)

    @tool
    async def alist_directory(self, path: str) -> list[dict[str, Any]]:
        """List entries in a VFS directory.

        Use this to discover what's available: ``/lib/``,
        ``/proc/runs/``, ``/lib/nodes/``, etc.

        Args
        ----
            path: Absolute VFS path (e.g. ``/lib/nodes/``).

        Returns
        -------
            List of entry dicts with ``name``, ``entry_type``, ``path``.
        """
        entries = await self._vfs.alist(path)
        return [e.model_dump() for e in entries]

    @tool
    async def astat_path(self, path: str) -> dict[str, Any]:
        """Get metadata about a VFS path.

        Returns useful context for decision-making: description,
        entity type, status, module path, capabilities, and tags.

        Args
        ----
            path: Absolute VFS path (e.g. ``/proc/runs/abc123``).

        Returns
        -------
            Metadata dict (description, status, capabilities, tags, etc.).
        """
        result = await self._vfs.astat(path)
        return result.model_dump(exclude_none=True)
