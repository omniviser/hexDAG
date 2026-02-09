"""Export API for hexdag studio.

Generates complete standalone Python projects from pipelines.
Uses the unified hexdag.api layer for feature parity with MCP server.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from hexdag import api

router = APIRouter(prefix="/export", tags=["export"])


class ExportRequest(BaseModel):
    """Request to export a pipeline as a standalone project."""

    content: str
    project_name: str | None = None
    include_docker: bool = False
    python_version: str = "3.12"


class ExportedFile(BaseModel):
    """A single exported file."""

    path: str
    content: str


class ExportResponse(BaseModel):
    """Complete exported project."""

    success: bool
    project_name: str
    files: list[ExportedFile]
    error: str | None = None


@router.post("", response_model=ExportResponse)
async def export_pipeline(request: ExportRequest) -> ExportResponse:
    """Export a pipeline as a complete standalone Python project.

    Uses the unified hexdag.api.export module for feature parity with MCP.

    Generates:
    - pyproject.toml with dependencies
    - README.md with usage instructions
    - .env.example with required environment variables
    - main.py runner script
    - pipeline.yaml (the original pipeline)
    - .gitignore
    - Optionally: Dockerfile
    """
    # Use unified API for export
    result = api.export.export_project(
        yaml_content=request.content,
        project_name=request.project_name,
        include_docker=request.include_docker,
        python_version=request.python_version,
    )

    if not result.get("success", False):
        return ExportResponse(
            success=False,
            project_name="",
            files=[],
            error=result.get("error", "Unknown error"),
        )

    # Convert API result to response format
    files = [ExportedFile(path=f["path"], content=f["content"]) for f in result.get("files", [])]

    return ExportResponse(
        success=True,
        project_name=result.get("project_name", ""),
        files=files,
    )


@router.post("/download")
async def download_project(request: ExportRequest):
    """Export and download as a ZIP file.

    This is a Studio-specific endpoint for downloading the exported project as a ZIP.
    """
    import io
    import traceback
    import zipfile

    from fastapi.responses import StreamingResponse

    # Validate content
    if not request.content or not request.content.strip():
        raise HTTPException(status_code=400, detail="No YAML content provided")

    # Generate the export using unified API
    try:
        export_result = await export_pipeline(request)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}") from e

    if not export_result.success:
        raise HTTPException(status_code=400, detail=export_result.error or "Unknown error")

    # Create ZIP file in memory using a generator for proper cleanup
    def generate_zip():
        zip_buffer = io.BytesIO()
        try:
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for file in export_result.files:
                    zf.writestr(f"{export_result.project_name}/{file.path}", file.content)
            zip_buffer.seek(0)
            yield from iter(lambda: zip_buffer.read(8192), b"")
        finally:
            zip_buffer.close()

    try:
        return StreamingResponse(
            generate_zip(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={export_result.project_name}.zip"
            },
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"ZIP creation failed: {str(e)}") from e
