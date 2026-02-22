"""File storage port definition."""

from typing import Protocol, runtime_checkable

from hexdag.kernel.ports.healthcheck import HealthStatus


@runtime_checkable
class FileStorage(Protocol):
    """Port for file storage operations.

    Provides a unified interface for local and cloud file storage.
    """

    async def aupload(self, local_path: str, remote_path: str) -> dict:
        """Upload a file.

        Parameters
        ----------
        local_path : str
            Path to local file
        remote_path : str
            Destination path in storage

        Returns
        -------
        dict
            Upload result with metadata
        """
        ...

    async def adownload(self, remote_path: str, local_path: str) -> dict:
        """Download a file.

        Parameters
        ----------
        remote_path : str
            Path in storage
        local_path : str
            Destination local path

        Returns
        -------
        dict
            Download result with metadata
        """
        ...

    async def adelete(self, remote_path: str) -> dict:
        """Delete a file.

        Parameters
        ----------
        remote_path : str
            Path to file in storage

        Returns
        -------
        dict
            Deletion result
        """
        ...

    async def alist(self, prefix: str = "") -> list[str]:
        """List files with optional prefix.

        Parameters
        ----------
        prefix : str
            Optional path prefix for filtering

        Returns
        -------
        list[str]
            List of file paths
        """
        ...

    async def aexists(self, remote_path: str) -> bool:
        """Check if file exists.

        Parameters
        ----------
        remote_path : str
            Path to check

        Returns
        -------
        bool
            True if file exists
        """
        ...

    async def aget_metadata(self, remote_path: str) -> dict:
        """Get file metadata.

        Parameters
        ----------
        remote_path : str
            Path to file

        Returns
        -------
        dict
            File metadata (size, modified_time, etc.)
        """
        ...

    async def ahealth_check(self) -> HealthStatus:
        """Check storage health.

        Returns
        -------
        HealthStatus
            Health status of the storage system
        """
        ...


# Backward-compat alias (deprecated: use FileStorage)
FileStoragePort = FileStorage
