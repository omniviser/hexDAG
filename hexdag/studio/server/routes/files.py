"""File operations API for hexdag studio.

Provides endpoints for listing, reading, and writing YAML pipeline files.
All operations work on local filesystem - no cloud storage.
"""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/files", tags=["files"])


class FileInfo(BaseModel):
    """File metadata."""

    name: str
    path: str
    is_directory: bool
    size: int | None = None
    modified: float | None = None


class FileContent(BaseModel):
    """File content with metadata."""

    path: str
    content: str
    modified: float


class SaveRequest(BaseModel):
    """Request to save file content."""

    content: str


class FileListResponse(BaseModel):
    """Response containing list of files."""

    root: str
    files: list[FileInfo]


# Global workspace root - set by studio server on startup
_workspace_root: Path | None = None


def set_workspace_root(path: Path) -> None:
    """Set the workspace root directory."""
    global _workspace_root
    _workspace_root = path


def get_workspace_root() -> Path:
    """Get the workspace root directory."""
    if _workspace_root is None:
        raise HTTPException(status_code=500, detail="Workspace root not configured")
    return _workspace_root


def _resolve_path(relative_path: str) -> Path:
    """Resolve a relative path within the workspace.

    Prevents directory traversal attacks.
    """
    root = get_workspace_root()
    resolved = (root / relative_path).resolve()

    # Security: ensure path is within workspace
    if not str(resolved).startswith(str(root.resolve())):
        raise HTTPException(status_code=403, detail="Access denied: path outside workspace")

    return resolved


def _get_file_info(path: Path, root: Path) -> FileInfo:
    """Create FileInfo from a path."""
    stat = path.stat()
    return FileInfo(
        name=path.name,
        path=str(path.relative_to(root)),
        is_directory=path.is_dir(),
        size=stat.st_size if path.is_file() else None,
        modified=stat.st_mtime,
    )


@router.get("", response_model=FileListResponse)
async def list_files(
    path: Annotated[str, Query(description="Relative path within workspace")] = "",
) -> FileListResponse:
    """List files and directories in the workspace.

    Returns YAML files and directories. Hidden files are excluded.
    """
    root = get_workspace_root()
    target = _resolve_path(path)

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {path}")

    files: list[FileInfo] = []
    for item in sorted(target.iterdir()):
        # Skip hidden files and common non-pipeline files
        if item.name.startswith("."):
            continue
        if item.name in ("__pycache__", "node_modules", ".venv", "venv"):
            continue

        # Include directories and YAML files
        if item.is_dir() or item.suffix in (".yaml", ".yml"):
            files.append(_get_file_info(item, root))

    return FileListResponse(root=str(root), files=files)


@router.get("/{file_path:path}", response_model=FileContent)
async def read_file(file_path: str) -> FileContent:
    """Read a file's content.

    Only YAML files can be read.
    """
    path = _resolve_path(file_path)

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    if path.is_dir():
        raise HTTPException(status_code=400, detail=f"Cannot read directory: {file_path}")

    if path.suffix not in (".yaml", ".yml"):
        raise HTTPException(status_code=400, detail=f"Only YAML files supported: {file_path}")

    try:
        content = path.read_text(encoding="utf-8")
        stat = path.stat()
        return FileContent(
            path=file_path,
            content=content,
            modified=stat.st_mtime,
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=f"Permission denied: {file_path}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}") from e


@router.put("/{file_path:path}", response_model=FileContent)
async def save_file(file_path: str, request: SaveRequest) -> FileContent:
    """Save content to a file.

    Creates parent directories if needed. Only YAML files can be saved.
    """
    path = _resolve_path(file_path)

    if path.suffix not in (".yaml", ".yml"):
        raise HTTPException(status_code=400, detail=f"Only YAML files supported: {file_path}")

    try:
        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write content
        path.write_text(request.content, encoding="utf-8")

        stat = path.stat()
        return FileContent(
            path=file_path,
            content=request.content,
            modified=stat.st_mtime,
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=f"Permission denied: {file_path}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {e}") from e


@router.delete("/{file_path:path}")
async def delete_file(file_path: str) -> dict[str, str]:
    """Delete a file.

    Only YAML files can be deleted. Directories cannot be deleted.
    """
    path = _resolve_path(file_path)

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    if path.is_dir():
        raise HTTPException(status_code=400, detail="Cannot delete directories")

    if path.suffix not in (".yaml", ".yml"):
        raise HTTPException(status_code=400, detail=f"Only YAML files supported: {file_path}")

    try:
        path.unlink()
        return {"status": "deleted", "path": file_path}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=f"Permission denied: {file_path}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {e}") from e
