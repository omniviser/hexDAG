"""Local filesystem storage adapter."""

import contextlib
import shutil
import time
from pathlib import Path
from typing import Any

from hexdag.core.configurable import AdapterConfig, ConfigurableAdapter
from hexdag.core.ports.file_storage import FileStoragePort
from hexdag.core.ports.healthcheck import HealthStatus
from hexdag.core.registry.decorators import adapter


class LocalFileStorageConfig(AdapterConfig):
    """Configuration for local file storage.

    Attributes
    ----------
    base_path : str
        Base directory for file storage (default: "./storage")
    create_if_missing : bool
        Create base directory if it doesn't exist (default: True)
    """

    base_path: str = "./storage"
    create_if_missing: bool = True


@adapter("file_storage", name="local", namespace="storage")
class LocalFileStorage(ConfigurableAdapter, FileStoragePort):
    """Local filesystem storage adapter.

    Provides file storage on the local filesystem with a base directory.

    Examples
    --------
    Basic usage::

        from hexdag.core.registry import registry

        storage = registry.get("local", namespace="storage")(
            base_path="./data"
        )

        # Upload file
        await storage.aupload("document.pdf", "docs/document.pdf")

        # List files
        files = await storage.alist(prefix="docs/")

        # Download file
        await storage.adownload("docs/document.pdf", "/tmp/document.pdf")

        # Check if exists
        exists = await storage.aexists("docs/document.pdf")

        # Delete file
        await storage.adelete("docs/document.pdf")
    """

    Config = LocalFileStorageConfig

    def __init__(self, **kwargs: Any) -> None:
        """Initialize local file storage."""
        super().__init__(**kwargs)
        self._base_path = Path(self.config.base_path)

        if self.config.create_if_missing:
            self._base_path.mkdir(parents=True, exist_ok=True)

    async def aupload(self, local_path: str, remote_path: str) -> dict:
        """Upload file to storage directory.

        Parameters
        ----------
        local_path : str
            Path to local file
        remote_path : str
            Destination path relative to base_path

        Returns
        -------
        dict
            Upload result with metadata
        """
        src = Path(local_path)
        dest = self._base_path / remote_path

        if not src.exists():
            msg = f"Source file not found: {local_path}"
            raise FileNotFoundError(msg)

        # Create destination directory
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy2(src, dest)

        return {
            "uploaded": True,
            "local_path": str(src),
            "remote_path": remote_path,
            "storage_path": str(dest),
            "size_bytes": dest.stat().st_size,
        }

    async def adownload(self, remote_path: str, local_path: str) -> dict:
        """Download file from storage directory.

        Parameters
        ----------
        remote_path : str
            Path in storage relative to base_path
        local_path : str
            Destination local path

        Returns
        -------
        dict
            Download result with metadata
        """
        src = self._base_path / remote_path
        dest = Path(local_path)

        if not src.exists():
            msg = f"File not found in storage: {remote_path}"
            raise FileNotFoundError(msg)

        # Create destination directory
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy2(src, dest)

        return {
            "downloaded": True,
            "remote_path": remote_path,
            "local_path": str(dest),
            "size_bytes": dest.stat().st_size,
        }

    async def adelete(self, remote_path: str) -> dict:
        """Delete file from storage.

        Parameters
        ----------
        remote_path : str
            Path to file relative to base_path

        Returns
        -------
        dict
            Deletion result
        """
        file_path = self._base_path / remote_path

        if not file_path.exists():
            msg = f"File not found: {remote_path}"
            raise FileNotFoundError(msg)

        # Delete file
        file_path.unlink()

        # Clean up empty parent directories
        with contextlib.suppress(FileNotFoundError):
            parent = file_path.parent
            if parent.is_dir():
                parent.rmdir()

        return {
            "deleted": True,
            "remote_path": remote_path,
        }

    async def alist(self, prefix: str = "") -> list[str]:
        """List files with optional prefix.

        Parameters
        ----------
        prefix : str
            Optional path prefix for filtering

        Returns
        -------
        list[str]
            List of file paths relative to base_path
        """
        if prefix:
            search_path = self._base_path / prefix
            pattern = "**/*"
        else:
            search_path = self._base_path
            pattern = "**/*"

        if not search_path.exists():
            return []

        files = []
        for path in search_path.glob(pattern):
            if path.is_file():
                # Get path relative to base_path
                rel_path = path.relative_to(self._base_path)
                files.append(str(rel_path))

        return sorted(files)

    async def aexists(self, remote_path: str) -> bool:
        """Check if file exists.

        Parameters
        ----------
        remote_path : str
            Path to check relative to base_path

        Returns
        -------
        bool
            True if file exists
        """
        file_path = self._base_path / remote_path
        return file_path.exists() and file_path.is_file()

    async def aget_metadata(self, remote_path: str) -> dict:
        """Get file metadata.

        Parameters
        ----------
        remote_path : str
            Path to file relative to base_path

        Returns
        -------
        dict
            File metadata (size, modified_time, created_time)
        """
        file_path = self._base_path / remote_path

        if not file_path.exists():
            msg = f"File not found: {remote_path}"
            raise FileNotFoundError(msg)

        stat = file_path.stat()

        return {
            "path": remote_path,
            "size_bytes": stat.st_size,
            "modified_time": stat.st_mtime,
            "created_time": stat.st_ctime,
            "is_file": file_path.is_file(),
            "absolute_path": str(file_path.absolute()),
        }

    async def ahealth_check(self) -> HealthStatus:
        """Check storage health.

        Returns
        -------
        HealthStatus
            Health status of the local storage
        """
        start_time = time.time()

        try:
            # Check if base path exists and is writable
            if not self._base_path.exists():
                return HealthStatus(
                    status="unhealthy",
                    adapter_name="local_file_storage",
                    port_name="file_storage",
                    latency_ms=(time.time() - start_time) * 1000,
                    details={
                        "base_path": str(self._base_path),
                        "error": "Base path does not exist",
                    },
                )

            # Test write access
            test_file = self._base_path / ".health_check"
            try:
                test_file.write_text("health_check")
                test_file.unlink()
            except Exception as e:
                return HealthStatus(
                    status="unhealthy",
                    adapter_name="local_file_storage",
                    port_name="file_storage",
                    error=e,
                    latency_ms=(time.time() - start_time) * 1000,
                    details={
                        "base_path": str(self._base_path),
                        "error": "Cannot write to base path",
                    },
                )

            # Count files
            file_count = len(await self.alist())

            return HealthStatus(
                status="healthy",
                adapter_name="local_file_storage",
                port_name="file_storage",
                latency_ms=(time.time() - start_time) * 1000,
                details={
                    "base_path": str(self._base_path.absolute()),
                    "file_count": file_count,
                    "writable": True,
                },
            )

        except Exception as e:
            return HealthStatus(
                status="unhealthy",
                adapter_name="local_file_storage",
                port_name="file_storage",
                error=e,
                latency_ms=(time.time() - start_time) * 1000,
                details={
                    "base_path": str(self._base_path),
                },
            )

    def __repr__(self) -> str:
        """String representation."""
        return f"LocalFileStorage(base_path={self._base_path})"
