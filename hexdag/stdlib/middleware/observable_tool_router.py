"""Observability middleware for ToolRouter ports.

Wraps a ``ToolRouter`` and emits ``ToolRouterPortCall`` events for every
tool call, including error cases.
"""

from __future__ import annotations

from typing import Any

from hexdag.kernel.context import get_current_node_name, get_observer_manager
from hexdag.kernel.logging import get_logger
from hexdag.kernel.ports.tool_router import ToolRouterPortCall
from hexdag.kernel.utils.node_timer import Timer

logger = get_logger(__name__)


class ObservableToolRouter:
    """Observability middleware for ToolRouter ports.

    Wraps a ToolRouter and emits ``ToolRouterPortCall`` for every tool call.

    Does NOT inherit from ``ToolRouter`` to avoid triggering health-check
    paths that expect ``HealthStatus`` objects.
    """

    def __init__(self, inner: Any) -> None:
        """Initialize with the inner ToolRouter to observe."""
        self._inner = inner

    async def acall_tool(self, tool_name: str, params: dict[str, Any]) -> Any:
        """Forward acall_tool and emit a ToolRouterPortCall event."""
        node_name = get_current_node_name() or "unknown"
        timer = Timer()
        try:
            result = await self._inner.acall_tool(tool_name, params)
        except Exception as exc:
            logger.error("Tool '{}' failed in {:.2f}ms: {}", tool_name, timer.duration_ms, exc)
            if mgr := get_observer_manager():
                await mgr.notify(
                    ToolRouterPortCall(
                        port_type="tool_router",
                        method="acall_tool",
                        node_name=node_name,
                        duration_ms=timer.duration_ms,
                        tool_name=tool_name,
                        params=params,
                        result={"error": str(exc)},
                    )
                )
            raise

        if mgr := get_observer_manager():
            await mgr.notify(
                ToolRouterPortCall(
                    port_type="tool_router",
                    method="acall_tool",
                    node_name=node_name,
                    duration_ms=timer.duration_ms,
                    tool_name=tool_name,
                    params=params,
                    result=result,
                )
            )
        return result

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the inner ToolRouter."""
        return getattr(self._inner, name)
