"""Agent lifecycle tool functions.

Plain functions for ``tool_end`` and ``change_phase`` — the two built-in
tools every agent gets automatically.  These are registered into a
``ToolRouter`` at execution time by ``agent_node.py``.

No class needed — just functions + constants.
"""

from __future__ import annotations

from typing import Any

# Agent lifecycle tool names — importable constants so other modules
# can reference them without magic strings.
TOOL_END = "tool_end"
CHANGE_PHASE = "change_phase"

__all__ = ["CHANGE_PHASE", "TOOL_END", "change_phase", "tool_end"]


def tool_end(**kwargs: Any) -> dict[str, Any]:
    """End agent execution with structured output."""
    return kwargs


def change_phase(
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
