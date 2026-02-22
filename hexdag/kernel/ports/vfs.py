"""VFS port — virtual filesystem for uniform entity introspection.

The VFS provides a path-based interface to all hexDAG entities. Every
pipeline run, component, port, and configuration is addressable by path —
the core of "everything is a file."

**Namespace tree:**

.. code-block:: text

    /lib/nodes|adapters|macros|tools|tags/ Component discovery
    /proc/runs/<run_id>/status|info       ProcessRegistry
    /proc/scheduled/<task_id>/status      Scheduler
    /proc/entities/<type>/<id>/state      EntityState

**Mount system:** Providers register at path prefixes. Longest-prefix
match resolves paths. Each subsystem implements :class:`VFSProvider`.

Drivers
-------
- ``LocalVFS`` — in-process, mount-based dispatch.
- Future: Distributed VFS (etcd/consul), REST-backed VFS.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from hexdag.kernel.domain.vfs import DirEntry, StatResult


@runtime_checkable
class VFSProvider(Protocol):
    """Mount point provider — handles a subtree of the VFS namespace.

    Each provider is responsible for a single path prefix (e.g. ``/lib/``,
    ``/proc/runs/``). The :class:`VFS` implementation routes requests to
    the appropriate provider based on longest-prefix matching.
    """

    @abstractmethod
    async def read(self, relative_path: str) -> str:
        """Read the content at a relative path.

        Args
        ----
            relative_path: Path relative to this provider's mount point.

        Returns
        -------
            JSON-serialized content string.

        Raises
        ------
        VFSError
            If the path does not exist or cannot be read.
        """
        ...

    @abstractmethod
    async def readdir(self, relative_path: str) -> list[DirEntry]:
        """List entries in a directory at a relative path.

        Args
        ----
            relative_path: Path relative to this provider's mount point.
                Empty string means the root of this provider's subtree.

        Returns
        -------
            List of directory entries.

        Raises
        ------
        VFSError
            If the path does not exist or is not a directory.
        """
        ...

    @abstractmethod
    async def stat(self, relative_path: str) -> StatResult:
        """Get metadata about a path.

        Args
        ----
            relative_path: Path relative to this provider's mount point.

        Returns
        -------
            Metadata about the path, designed for agent/MCP consumption.

        Raises
        ------
        VFSError
            If the path does not exist.
        """
        ...


@runtime_checkable
class VFS(Protocol):
    """Virtual filesystem port — uniform path-based entity access.

    The VFS is the primary query interface for all hexDAG entities.
    Implementations route absolute paths to registered
    :class:`VFSProvider` instances via a mount table.
    """

    @abstractmethod
    async def aread(self, path: str) -> str:
        """Read the content at an absolute VFS path.

        Args
        ----
            path: Absolute path (e.g. ``/lib/nodes/llm_node``).

        Returns
        -------
            JSON-serialized content string.
        """
        ...

    @abstractmethod
    async def alist(self, path: str) -> list[DirEntry]:
        """List entries in a directory at an absolute VFS path.

        Args
        ----
            path: Absolute path (e.g. ``/lib/nodes/``).

        Returns
        -------
            List of directory entries.
        """
        ...

    @abstractmethod
    async def astat(self, path: str) -> StatResult:
        """Get metadata about an absolute VFS path.

        Args
        ----
            path: Absolute path (e.g. ``/proc/runs/abc123``).

        Returns
        -------
            Metadata about the path.
        """
        ...

    @abstractmethod
    def mount(self, prefix: str, provider: VFSProvider) -> None:
        """Mount a provider at a path prefix.

        Args
        ----
            prefix: Path prefix (e.g. ``/lib/``, ``/proc/runs/``).
                Must start and end with ``/``.
            provider: The provider to handle paths under this prefix.
        """
        ...

    @abstractmethod
    def mounts(self) -> dict[str, VFSProvider]:
        """Return a copy of the current mount table.

        Returns
        -------
            Mapping of prefix → provider.
        """
        ...


__all__ = ["VFS", "VFSProvider"]
