"""VFS provider for /etc/ — pipeline definitions.

Scans configured pipeline directories for ``.yaml`` files and serves
their content.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hexdag.kernel.domain.vfs import DirEntry, EntryType, StatResult
from hexdag.kernel.exceptions import VFSError

if TYPE_CHECKING:
    from pathlib import Path


class EtcProvider:
    """VFS provider for /etc/ — pipeline definitions."""

    def __init__(self, pipeline_paths: list[Path] | None = None) -> None:
        self._pipeline_paths = pipeline_paths or []

    def _discover_pipelines(self) -> dict[str, Path]:
        """Scan configured paths for YAML pipeline files."""
        pipelines: dict[str, Path] = {}
        for base_path in self._pipeline_paths:
            if base_path.is_file() and base_path.suffix in (".yaml", ".yml"):
                pipelines[base_path.stem] = base_path
            elif base_path.is_dir():
                for f in sorted(base_path.iterdir()):
                    if f.suffix in (".yaml", ".yml") and f.is_file():
                        pipelines[f.stem] = f
        return pipelines

    async def read(self, relative_path: str) -> str:
        """Read pipeline definitions.

        Paths
        -----
        - ``pipelines/<name>`` → raw YAML content
        """
        path = relative_path.strip("/")
        if not path:
            raise VFSError("/etc/", "cannot read directory; use readdir")

        parts = path.split("/")
        if parts[0] != "pipelines":
            raise VFSError(f"/etc/{path}", "only /etc/pipelines/ is available")

        if len(parts) == 1:
            raise VFSError("/etc/pipelines", "cannot read directory; use readdir")

        pipeline_name = parts[1]
        pipelines = self._discover_pipelines()
        if pipeline_name not in pipelines:
            available = sorted(pipelines.keys())
            raise VFSError(
                f"/etc/{path}",
                f"pipeline '{pipeline_name}' not found. Available: {available}",
            )

        return pipelines[pipeline_name].read_text()

    async def readdir(self, relative_path: str) -> list[DirEntry]:
        """List pipeline definitions.

        Paths
        -----
        - ``""`` → ["pipelines"]
        - ``pipelines`` → list of pipeline names
        """
        path = relative_path.strip("/")

        if not path:
            return [
                DirEntry(name="pipelines", entry_type=EntryType.DIRECTORY, path="/etc/pipelines")
            ]

        if path == "pipelines":
            pipelines = self._discover_pipelines()
            return [
                DirEntry(
                    name=name,
                    entry_type=EntryType.FILE,
                    path=f"/etc/pipelines/{name}",
                )
                for name in sorted(pipelines.keys())
            ]

        raise VFSError(f"/etc/{path}", "not a directory")

    async def stat(self, relative_path: str) -> StatResult:
        """Get metadata about pipeline definitions."""
        path = relative_path.strip("/")

        if not path:
            return StatResult(
                path="/etc",
                entry_type=EntryType.DIRECTORY,
                description="Configuration and pipeline definitions",
                child_count=1,
                capabilities=["read"],
            )

        parts = path.split("/")
        if parts[0] == "pipelines" and len(parts) == 1:
            pipelines = self._discover_pipelines()
            return StatResult(
                path="/etc/pipelines",
                entry_type=EntryType.DIRECTORY,
                description="YAML pipeline definitions",
                child_count=len(pipelines),
                capabilities=["read"],
            )

        if parts[0] == "pipelines" and len(parts) == 2:
            pipeline_name = parts[1]
            pipelines = self._discover_pipelines()
            if pipeline_name not in pipelines:
                raise VFSError(f"/etc/{path}", f"pipeline '{pipeline_name}' not found")

            return StatResult(
                path=f"/etc/pipelines/{pipeline_name}",
                entry_type=EntryType.FILE,
                entity_type="pipeline",
                description=f"Pipeline definition '{pipeline_name}'",
                capabilities=["read"],
                tags={"file_path": str(pipelines[pipeline_name])},
            )

        raise VFSError(f"/etc/{path}", "path not found")
