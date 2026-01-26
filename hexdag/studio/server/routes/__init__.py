"""API routes for hexdag studio."""

from hexdag.studio.server.routes.execute import router as execute_router
from hexdag.studio.server.routes.export import router as export_router
from hexdag.studio.server.routes.files import router as files_router
from hexdag.studio.server.routes.plugins import router as plugins_router
from hexdag.studio.server.routes.validate import router as validate_router

__all__ = ["files_router", "validate_router", "execute_router", "export_router", "plugins_router"]
