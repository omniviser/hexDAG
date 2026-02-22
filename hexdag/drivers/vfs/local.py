"""Local in-process VFS driver.

Routes absolute paths to registered :class:`VFSProvider` instances
using longest-prefix matching. This is the default VFS implementation
for single-process hexDAG deployments.

Example
-------
.. code-block:: python

    vfs = LocalVFS()
    vfs.mount("/lib/", LibProvider())
    vfs.mount("/proc/runs/", ProcRunsProvider(registry))

    content = await vfs.aread("/lib/nodes/llm_node")
    entries = await vfs.alist("/proc/runs/")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from hexdag.kernel.domain.vfs import DirEntry, EntryType, StatResult
from hexdag.kernel.exceptions import VFSError

if TYPE_CHECKING:
    from hexdag.kernel.ports.vfs import VFSProvider

logger = logging.getLogger(__name__)


class LocalVFS:
    """In-process VFS with mount-based dispatch.

    Routes paths to registered VFSProvider instances using
    longest-prefix matching.
    """

    def __init__(self) -> None:
        self._mounts: dict[str, VFSProvider] = {}

    def mount(self, prefix: str, provider: VFSProvider) -> None:
        """Mount a provider at a path prefix.

        Args
        ----
            prefix: Path prefix (e.g. ``/lib/``, ``/proc/runs/``).
                Must start and end with ``/``.
            provider: The provider to handle paths under this prefix.
        """
        if not prefix.startswith("/"):
            msg = f"Mount prefix must start with '/': {prefix!r}"
            raise ValueError(msg)
        if not prefix.endswith("/"):
            prefix = prefix + "/"
        self._mounts[prefix] = provider
        logger.debug("Mounted VFS provider at %s: %s", prefix, type(provider).__name__)

    def mounts(self) -> dict[str, VFSProvider]:
        """Return a copy of the current mount table."""
        return dict(self._mounts)

    def _resolve(self, path: str) -> tuple[VFSProvider, str]:
        """Find the provider for a path using longest-prefix matching.

        Args
        ----
            path: Absolute VFS path.

        Returns
        -------
            Tuple of (provider, relative_path).

        Raises
        ------
        VFSError
            If no provider is mounted for the given path.
        """
        if not path.startswith("/"):
            raise VFSError(path, "path must be absolute (start with '/')")

        # Normalise: ensure trailing slash for matching
        normalized = path if path.endswith("/") else path + "/"

        # Longest-prefix match
        best_prefix = ""
        best_provider: VFSProvider | None = None

        for prefix, provider in self._mounts.items():
            if normalized.startswith(prefix) and len(prefix) > len(best_prefix):
                best_prefix = prefix
                best_provider = provider

        if best_provider is None:
            available = sorted(self._mounts.keys())
            raise VFSError(path, f"no provider mounted for this path. Available: {available}")

        # Strip the prefix to get relative path, also strip trailing slash
        relative = path[len(best_prefix) :]
        return best_provider, relative

    async def aread(self, path: str) -> str:
        """Read the content at an absolute VFS path."""
        if path == "/":
            raise VFSError(path, "cannot read root directory; use alist('/') instead")
        provider, relative = self._resolve(path)
        return await provider.read(relative)

    async def alist(self, path: str) -> list[DirEntry]:
        """List entries in a directory at an absolute VFS path."""
        if path == "/":
            return self._list_root()
        provider, relative = self._resolve(path)
        return await provider.readdir(relative)

    async def astat(self, path: str) -> StatResult:
        """Get metadata about an absolute VFS path."""
        if path == "/":
            return StatResult(
                path="/",
                entry_type=EntryType.DIRECTORY,
                description="hexDAG virtual filesystem root",
                child_count=len(self._mounts),
                capabilities=["read"],
            )
        provider, relative = self._resolve(path)
        return await provider.stat(relative)

    def _list_root(self) -> list[DirEntry]:
        """List mount-point prefixes as root directory entries."""
        entries: list[DirEntry] = []
        # Collect unique top-level directories from mount prefixes
        seen: set[str] = set()
        for prefix in sorted(self._mounts):
            # Extract top-level dir: "/lib/" → "lib", "/proc/runs/" → "proc"
            parts = prefix.strip("/").split("/")
            top = parts[0]
            if top not in seen:
                seen.add(top)
                entries.append(
                    DirEntry(
                        name=top,
                        entry_type=EntryType.DIRECTORY,
                        path=f"/{top}",
                    )
                )
        return entries
