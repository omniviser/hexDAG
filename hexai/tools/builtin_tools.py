"""Built-in tools that are essential for agent operations."""

from typing import Any

from hexai.core.registry import tool


# Register tool_end
@tool(name="tool_end", namespace="core")
def tool_end(**kwargs: Any) -> dict[str, Any]:
    """End tool execution with structured output.

    This is a built-in tool that agents can use to return
    structured data matching their output schema.

    Args
    ----
        **kwargs: Any structured data to return

    Returns
    -------
        The structured data as provided
    """
    return kwargs


# Register change_phase
@tool(name="change_phase", namespace="core")
def change_phase(phase: str, **context: Any) -> dict[str, Any]:
    """Change the agent's reasoning phase.

    This tool allows agents to transition between different
    reasoning phases with optional context data.

    Args
    ----
        phase: The new phase name to transition to
        **context: Optional context data for the phase transition

    Returns
    -------
        Dictionary with phase change information
    """
    return {"action": "change_phase", "new_phase": phase, "context": context}
