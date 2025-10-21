"""Built-in tools that are essential for agent operations."""

from typing import TYPE_CHECKING, Any

from hexdag.core.registry import tool

if TYPE_CHECKING:
    from hexdag.builtin.nodes.agent_node import PhaseContext


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


# Register change_phase with typed parameters
@tool(name="change_phase", namespace="core")
def change_phase(
    phase: str,
    previous_phase: str | None = None,
    reason: str | None = None,
    carried_data: dict[str, Any] | None = None,
    target_output: str | None = None,
    iteration: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Change the agent's reasoning phase with typed context.

    This tool allows agents to transition between different
    reasoning phases with strongly-typed context data.

    Args
    ----
        phase: The new phase name to transition to
        previous_phase: The phase being transitioned from (auto-filled if not provided)
        reason: Explanation for why the phase change is occurring
        carried_data: Data to carry forward from the previous phase
        target_output: Expected output format or goal for the new phase
        iteration: Current iteration number if in a loop or retry scenario
        metadata: Additional metadata about the phase transition

    Returns
    -------
        Dictionary with phase change information including:
        - action: Always "change_phase"
        - new_phase: The phase being transitioned to
        - context: PhaseContext-typed data

    Examples
    --------
    >>> change_phase("analysis", reason="Initial data gathering complete")
    {'action': 'change_phase', 'new_phase': 'analysis', 'context': {'reason':
    'Initial data gathering complete'}}

    >>> change_phase("synthesis",
    ...              previous_phase="analysis",
    ...              carried_data={"key_findings": ["item1", "item2"]},
    ...              iteration=2)
    {'action': 'change_phase', 'new_phase': 'synthesis', 'context': {...}}
    """
    context: PhaseContext = {}

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
