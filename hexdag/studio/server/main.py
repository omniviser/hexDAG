"""FastAPI server for hexdag studio.

Local-first visual editor for hexdag pipelines.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from hexdag.studio.server.routes import (
    execute_router,
    export_router,
    files_router,
    plugins_router,
    validate_router,
)
from hexdag.studio.server.routes.files import set_workspace_root


def create_app(workspace_path: Path) -> FastAPI:
    """Create FastAPI application for hexdag studio.

    Args:
        workspace_path: Root directory for pipeline files
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # Startup
        set_workspace_root(workspace_path)
        print(f"Workspace: {workspace_path}")
        yield
        # Shutdown
        pass

    app = FastAPI(
        title="hexdag studio",
        description="Local-first visual editor for hexdag pipelines",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3141", "http://127.0.0.1:3141", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    app.include_router(files_router, prefix="/api")
    app.include_router(validate_router, prefix="/api")
    app.include_router(execute_router, prefix="/api")
    app.include_router(export_router, prefix="/api")
    app.include_router(plugins_router, prefix="/api")

    # Health check
    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "workspace": str(workspace_path)}

    # Serve static UI files
    ui_path = Path(__file__).parent.parent / "ui" / "dist"
    if ui_path.exists():
        app.mount("/assets", StaticFiles(directory=ui_path / "assets"), name="assets")

        @app.get("/")
        async def serve_index() -> FileResponse:
            return FileResponse(ui_path / "index.html")

        @app.get("/{path:path}")
        async def serve_spa(path: str) -> FileResponse:
            # SPA fallback - serve index.html for all non-API routes
            file_path = ui_path / path
            if file_path.exists() and file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(ui_path / "index.html")

    return app


def run_server(workspace_path: Path, host: str = "127.0.0.1", port: int = 3141) -> None:
    """Run the studio server.

    Args:
        workspace_path: Root directory for pipeline files
        host: Host to bind to
        port: Port to bind to
    """
    import uvicorn

    app = create_app(workspace_path)
    uvicorn.run(app, host=host, port=port, log_level="info")
