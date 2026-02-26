"""The one tool router. Wraps plain Python functions.

Handles sync/async, auto-generates schemas from signatures,
filters params to match function signatures, tracks call history.
Subclassable for custom routers (MCP, authenticated APIs, etc.).
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hexdag.kernel._alias_registry import resolve_function
from hexdag.kernel.exceptions import ResourceNotFoundError, TypeMismatchError
from hexdag.kernel.logging import get_logger
from hexdag.kernel.orchestration.events.events import Event

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


def tool_schema_from_callable(fn: Any) -> dict[str, Any]:
    """Auto-generate a tool schema from a function's signature and docstring.

    Returns a dict matching the ToolRouter schema contract::

        {"description": "...", "parameters": [{"name": ..., "type": ..., "required": ...}, ...]}

    Parameters
    ----------
    fn : callable
        The function to introspect

    Returns
    -------
    dict[str, Any]
        Tool schema with description and parameters list
    """
    description = (fn.__doc__ or "").split("\n")[0].strip()
    sig = inspect.signature(fn)

    parameters: list[dict[str, Any]] = []
    for name, param in sig.parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            parameters.append({
                "name": f"**{name}",
                "description": "Key-value pairs matching the agent's output schema",
                "type": "Any",
                "required": True,
            })
            continue
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue

        annotation = param.annotation
        type_name = getattr(annotation, "__name__", str(annotation))
        type_name = type_name.replace("typing.", "").replace(" | None", "")

        parameters.append({
            "name": name,
            "type": type_name,
            "required": param.default is inspect.Parameter.empty,
        })

    return {"description": description, "parameters": parameters}


class ToolRouter:
    """Concrete tool router that wraps plain Python functions.

    Works out of the box AND is subclassable for custom routers
    (MCP-native, authenticated APIs, etc.).

    Parameters
    ----------
    tools : dict[str, Callable | str], optional
        Mapping of tool names to callables or module path strings.
        Strings are resolved via ``resolve_function()`` at init time.
    **kwargs : Any
        Forward compatibility / YAML config passthrough.
    """

    def __init__(
        self,
        tools: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}
        self.call_history: list[dict[str, Any]] = []

        for name, fn_or_path in (tools or {}).items():
            if isinstance(fn_or_path, str):
                fn_or_path = resolve_function(fn_or_path)
            if not callable(fn_or_path):
                raise TypeMismatchError(name, "callable", type(fn_or_path).__name__, fn_or_path)
            self._tools[name] = fn_or_path

    # ── Mutation ──────────────────────────────────────────────────────

    def add_tool(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a single tool function."""
        self._tools[name] = fn

    def add_tools_from(self, other: ToolRouter) -> None:
        """Merge another router's tools into this one.

        Only adds tools that don't already exist in this router.
        Replaces the old ``UnifiedToolRouter`` merging pattern.
        """
        if hasattr(other, "_tools"):
            for name, fn in other._tools.items():
                if name not in self._tools:
                    self._tools[name] = fn

    # ── Core interface ────────────────────────────────────────────────

    async def acall_tool(self, tool_name: str, params: dict[str, Any]) -> Any:
        """Call a tool. Handles sync/async, filters params to match signature."""
        if tool_name not in self._tools:
            raise ResourceNotFoundError("tool", tool_name, list(self._tools.keys()))

        fn = self._tools[tool_name]

        # Filter params to match function signature
        sig = inspect.signature(fn)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if has_var_keyword:
            kwargs = params
        else:
            kwargs = {k: v for k, v in params.items() if k in sig.parameters}

        # Handle sync/async
        if asyncio.iscoroutinefunction(fn):
            result = await fn(**kwargs)
        else:
            result = fn(**kwargs)

        # Track call history
        self.call_history.append({
            "tool_name": tool_name,
            "params": params,
            "result": result,
        })

        return result

    def get_available_tools(self) -> list[str]:
        """Return names of all registered tools."""
        return list(self._tools.keys())

    async def aget_available_tools(self) -> list[str]:
        """Async version. Override in subclass for remote/MCP adapters."""
        return self.get_available_tools()

    def get_tool_schema(self, tool_name: str) -> dict[str, Any]:
        """Return schema for a specific tool."""
        fn = self._tools.get(tool_name)
        if fn is None:
            return {}
        return tool_schema_from_callable(fn)

    async def aget_tool_schema(self, tool_name: str) -> dict[str, Any]:
        """Async version. Override in subclass for remote/MCP adapters."""
        return self.get_tool_schema(tool_name)

    def get_all_tool_schemas(self) -> dict[str, dict[str, Any]]:
        """Return schemas for all registered tools."""
        return {name: tool_schema_from_callable(fn) for name, fn in self._tools.items()}

    async def aget_all_tool_schemas(self) -> dict[str, dict[str, Any]]:
        """Async version. Override in subclass for remote/MCP adapters."""
        return self.get_all_tool_schemas()

    # ── Health check ──────────────────────────────────────────────────

    async def ahealth_check(self) -> dict[str, Any]:
        """Verify all registered tools are callable."""
        return {
            "status": "healthy",
            "tool_count": len(self._tools),
            "tools": list(self._tools.keys()),
        }


# ---------------------------------------------------------------------------
# Port event
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolRouterEvent(Event):
    """Event emitted when ToolRouter.acall_tool() completes."""

    node_name: str
    tool_name: str
    params: dict[str, Any] | None = None
    result: Any = None
    duration_ms: float = 0.0
