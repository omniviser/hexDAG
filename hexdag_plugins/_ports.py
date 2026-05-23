"""Port protocols used by hexdag_plugins.

These protocols were previously in hexdag.kernel.ports but were moved here
because they are only used by the plugins package, not by core hexDAG.
"""

from typing import Protocol, runtime_checkable

from hexdag.kernel.ports.healthcheck import HealthStatus


@runtime_checkable
class FileStorage(Protocol):
    """Port for file storage operations.

    Provides a unified interface for local and cloud file storage.
    """

    async def aupload(self, local_path: str, remote_path: str) -> dict:
        """Upload a file."""
        ...

    async def adownload(self, remote_path: str, local_path: str) -> dict:
        """Download a file."""
        ...

    async def adelete(self, remote_path: str) -> dict:
        """Delete a file."""
        ...

    async def alist(self, prefix: str = "") -> list[str]:
        """List files with optional prefix."""
        ...

    async def aexists(self, remote_path: str) -> bool:
        """Check if file exists."""
        ...

    async def aget_metadata(self, remote_path: str) -> dict:
        """Get file metadata."""
        ...

    async def ahealth_check(self) -> HealthStatus:
        """Check storage health."""
        ...
