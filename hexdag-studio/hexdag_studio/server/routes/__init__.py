"""API routes for hexdag-studio."""

from hexdag_studio.server.routes.environments import router as environments_router
from hexdag_studio.server.routes.execute import router as execute_router
from hexdag_studio.server.routes.export import router as export_router
from hexdag_studio.server.routes.files import router as files_router
from hexdag_studio.server.routes.plugins import router as plugins_router
from hexdag_studio.server.routes.registry import router as registry_router
from hexdag_studio.server.routes.validate import router as validate_router

__all__ = [
    "environments_router",
    "files_router",
    "validate_router",
    "execute_router",
    "export_router",
    "plugins_router",
    "registry_router",
]
