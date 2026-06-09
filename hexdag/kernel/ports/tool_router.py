"""Backward-compatibility shim — ToolRouter moved to hexdag.kernel.tool_router.

ToolRouter is a concrete class, not a port protocol. It was moved out of
``kernel/ports/`` to reflect this. All imports from this path still work.
"""

from hexdag.kernel.tool_router import (  # noqa: F401
    ToolRouter,
    ToolRouterCall,
    ToolRouterEvent,
    ToolRouterPortCall,
    tool_schema_from_callable,
)

__all__ = [
    "ToolRouter",
    "ToolRouterCall",
    "ToolRouterEvent",
    "ToolRouterPortCall",
    "tool_schema_from_callable",
]
