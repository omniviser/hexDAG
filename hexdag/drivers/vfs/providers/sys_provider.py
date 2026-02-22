"""VFS provider for /sys/ — system metadata.

Exposes hexDAG version, configuration, and other system-level
information.
"""

from __future__ import annotations

import json

from hexdag.kernel.domain.vfs import DirEntry, EntryType, StatResult
from hexdag.kernel.exceptions import VFSError

_SYS_ENTRIES = ("version", "config")


class SysProvider:
    """VFS provider for /sys/ — system metadata."""

    def __init__(self, config: dict[str, str] | None = None) -> None:
        self._config = config or {}

    def _get_version(self) -> str:
        """Get the hexDAG version string."""
        try:
            from importlib.metadata import version

            return version("hexdag")
        except Exception:
            return "unknown"

    async def read(self, relative_path: str) -> str:
        """Read system metadata.

        Paths
        -----
        - ``version`` → hexDAG version string
        - ``config`` → JSON of runtime config
        """
        path = relative_path.strip("/")
        if not path:
            raise VFSError("/sys/", "cannot read directory; use readdir")

        if path == "version":
            return json.dumps(self._get_version())

        if path == "config":
            return json.dumps(self._config, indent=2)

        raise VFSError(f"/sys/{path}", "path not found")

    async def readdir(self, relative_path: str) -> list[DirEntry]:
        """List system entries.

        Paths
        -----
        - ``""`` → ["version", "config"]
        """
        path = relative_path.strip("/")
        if path:
            raise VFSError(f"/sys/{path}", "not a directory")

        return [
            DirEntry(name=name, entry_type=EntryType.FILE, path=f"/sys/{name}")
            for name in _SYS_ENTRIES
        ]

    async def stat(self, relative_path: str) -> StatResult:
        """Get metadata about system entries."""
        path = relative_path.strip("/")

        if not path:
            return StatResult(
                path="/sys",
                entry_type=EntryType.DIRECTORY,
                description="System metadata and configuration",
                child_count=len(_SYS_ENTRIES),
                capabilities=["read"],
            )

        if path == "version":
            return StatResult(
                path="/sys/version",
                entry_type=EntryType.FILE,
                description=f"hexDAG version: {self._get_version()}",
                capabilities=["read"],
            )

        if path == "config":
            return StatResult(
                path="/sys/config",
                entry_type=EntryType.FILE,
                description="Runtime configuration",
                capabilities=["read"],
            )

        raise VFSError(f"/sys/{path}", "path not found")
