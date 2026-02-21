"""Core agent lifecycle tools as a proper ToolRouter.

This module moves ``tool_end`` and ``change_phase`` from hardcoded special
cases in ``agent_node.py`` into a first-class ``ToolRouter`` that can be
registered in ``UnifiedToolRouter`` like any other tool source.

The ``AgentToolRouter`` is the single source of truth for agent lifecycle
tools.  It is discoverable via ``get_available_tools()`` and
``get_all_tool_schemas()``, fixing the hexagonal architecture violation
where these tools were invisible to the standard tool interface.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from hexdag.core.ports.tool_router import ToolRouter, tool_schema_from_callable

if TYPE_CHECKING:
    from collections.abc import Callable

# Agent lifecycle tool names — importable constants so other modules
# can reference them without magic strings.
TOOL_END = "tool_end"
CHANGE_PHASE = "change_phase"


class AgentToolRouter(ToolRouter):
    """ToolRouter implementation for agent lifecycle tools.

    Wraps ``tool_end`` and ``change_phase`` as first-class tools that
    are discoverable, have schemas, and route through the standard
    ``ToolRouter`` interface.

    Parameters
    ----------
    on_phase_change : callable, optional
        Callback invoked when ``change_phase`` is called.  Receives
        the result dict and can mutate agent state accordingly.
        Signature: ``(result: dict) -> None``.
    """

    def __init__(
        self,
        on_phase_change: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._on_phase_change = on_phase_change

        # Register tool functions
        self._tools: dict[str, Callable[..., Any]] = {
            TOOL_END: _tool_end,
            CHANGE_PHASE: _change_phase,
        }

    # ── ToolRouter protocol ──────────────────────────────────────────

    async def acall_tool(self, tool_name: str, params: dict[str, Any]) -> Any:
        """Call an agent lifecycle tool."""
        if tool_name not in self._tools:
            from hexdag.core.exceptions import ResourceNotFoundError

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

        result = fn(**kwargs)

        # Fire callback for phase changes
        if tool_name == CHANGE_PHASE and self._on_phase_change is not None:
            self._on_phase_change(result)

        return result

    def get_available_tools(self) -> list[str]:
        """Return names of agent lifecycle tools."""
        return list(self._tools.keys())

    def get_tool_schema(self, tool_name: str) -> dict[str, Any]:
        """Return schema for an agent lifecycle tool."""
        fn = self._tools.get(tool_name)
        if fn is None:
            return {}
        return tool_schema_from_callable(fn)

    def get_all_tool_schemas(self) -> dict[str, dict[str, Any]]:
        """Return schemas for all agent lifecycle tools."""
        return {name: tool_schema_from_callable(fn) for name, fn in self._tools.items()}


# ── Tool implementations ────────────────────────────────────────────
# Extracted from hexdag/builtin/tools/builtin_tools.py so the domain
# layer owns them.


def _tool_end(**kwargs: Any) -> dict[str, Any]:
    """End agent execution with structured output."""
    return kwargs


def _change_phase(
    phase: str,
    previous_phase: str | None = None,
    reason: str | None = None,
    carried_data: dict[str, Any] | None = None,
    target_output: str | None = None,
    iteration: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Change the agent's reasoning phase with typed context."""
    context: dict[str, Any] = {}
    if previous_phase is not None:
        context["previous_phase"] = previous_phase
    if reason is not None:
        context["reason"] = reason
    if carried_data is not None:
        context["carried_data"] = carried_data
    if target_output is not None:
        context["target_output"] = target_output
    if iteration is not None:
        context["iteration"] = iteration
    if metadata is not None:
        context["metadata"] = metadata

    return {"action": "change_phase", "new_phase": phase, "context": context}
