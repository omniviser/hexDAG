"""ToolCallNode - Execute a single tool call as a FunctionNode.

This node wraps a tool function from the registry and executes it as a node.
No ToolRouter needed - tools are accessed directly from the registry.
Used by ToolMacro to create parallel tool execution nodes.
"""

import contextlib
import inspect
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from hexdag.core.context import get_port
from hexdag.core.domain.dag import NodeSpec
from hexdag.core.logging import get_logger
from hexdag.core.orchestration.events import ToolCalled, ToolCompleted
from hexdag.core.registry import node, registry
from hexdag.core.registry.models import NodeSubtype

from .base_node_factory import BaseNodeFactory

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


class ToolCallInput(BaseModel):
    """Input for a tool call."""

    tool_name: str
    arguments: dict[str, Any] = {}
    tool_call_id: str | None = None


class ToolCallOutput(BaseModel):
    """Output from a tool call."""

    result: Any
    tool_call_id: str | None = None
    tool_name: str
    error: str | None = None


@node(name="tool_call_node", subtype=NodeSubtype.FUNCTION, namespace="core")
class ToolCallNode(BaseNodeFactory):
    """Execute a single tool call as a FunctionNode.

    This node is a simple wrapper that:
    1. Takes a tool name and arguments
    2. Gets the tool function from the registry
    3. Injects required ports (if tool needs them)
    4. Executes it and returns the result with metadata
    5. Emits ToolCalled and ToolCompleted events

    Port Injection:
    ---------------
    If a tool declares `required_ports=["database"]`, ToolCallNode will:
    - Get the "database" port from context
    - Inject it as `database_port` parameter
    - Tool receives it: `def my_tool(arg1, database_port=None)`

    No ToolRouter needed - tools accessed directly from registry.
    Used by ToolMacro to create parallel tool execution nodes.

    Examples
    --------
    Direct usage::

        tool_node = ToolCallNode()(
            name="search_tool",
            tool_name="search",
            arguments={"query": "python async"}
        )

        result = await orchestrator.run(
            graph,
            {"tool_call_id": "1"}
        )
        # Returns: {"result": [...], "tool_call_id": "1", "tool_name": "search"}

    Tool requiring ports::

        # Tool definition
        @tool(name="db_query", required_ports=["database"])
        async def query_db(sql: str, database_port=None):
            return await database_port.aexecute_query(sql)

        # ToolCallNode automatically injects database port
        db_tool = ToolCallNode()(
            name="db_query_node",
            tool_name="db_query",
            arguments={"sql": "SELECT * FROM users"}
        )

        # When executed, database_port is injected from context

    In a macro (automatic parallel execution)::

        # ToolMacro creates multiple ToolCallNodes
        tool1 = ToolCallNode()(name="tool_1", tool_name="search", ...)
        tool2 = ToolCallNode()(name="tool_2", tool_name="calc", ...)
        # Orchestrator executes them in parallel automatically
        # Ports injected automatically for tools that need them
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize ToolCallNode."""
        super().__init__()

    def __call__(
        self,
        name: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a tool call execution node.

        Args
        ----
            name: Node name (should be unique)
            tool_name: Name of the tool to execute (from registry)
            arguments: Arguments to pass to the tool (default: {})
            tool_call_id: Optional ID for tracking (from LLM tool calls)
            deps: Dependencies (typically the LLM node that requested the tool)
            **kwargs: Additional parameters

        Returns
        -------
        NodeSpec
            Configured node specification for tool execution
        """
        arguments = arguments or {}

        async def execute_tool(input_data: dict[str, Any]) -> dict[str, Any]:
            """Execute the tool call with event emission."""

            # Get observer for event emission (optional)
            observer_manager = None
            with contextlib.suppress(Exception):
                observer_manager = get_port("observer_manager")

            # Emit ToolCalled event
            if observer_manager:
                await observer_manager.notify(
                    ToolCalled(node_name=name, tool_name=tool_name, params=arguments)
                )

            logger.debug(f"🔧 Executing tool '{tool_name}' with args: {arguments}")

            start_time = time.time()
            try:
                # Get tool function and metadata from registry
                tool_fn_obj = registry.get(tool_name)
                tool_metadata = registry.get_metadata(tool_name)

                # Type assertion for callable
                tool_fn: Callable[..., Any] = tool_fn_obj  # type: ignore[assignment]

                # Prepare tool arguments + inject required ports
                tool_kwargs = dict(arguments)  # Start with provided arguments

                if tool_metadata and tool_metadata.port_requirements:
                    # Tool needs ports - inject them as parameters
                    for port_name in tool_metadata.port_requirements:
                        try:
                            # Get port and inject as {port_name}_port parameter
                            # Example: "database" port → "database_port" parameter
                            port = get_port(port_name)
                            port_param_name = f"{port_name}_port"
                            tool_kwargs[port_param_name] = port
                            logger.debug(f"  Injected port '{port_name}' as '{port_param_name}'")
                        except Exception as e:
                            logger.warning(
                                f"  Failed to inject required port '{port_name}': {e}. "
                                f"Tool may fail if port is required."
                            )
                            # Don't add the port - tool will get None or handle missing port

                # Execute tool with arguments + injected ports (handle both sync and async)
                if inspect.iscoroutinefunction(tool_fn):
                    result = await tool_fn(**tool_kwargs)
                else:
                    result = tool_fn(**tool_kwargs)

                duration_ms = (time.time() - start_time) * 1000

                # Emit ToolCompleted event
                if observer_manager:
                    await observer_manager.notify(
                        ToolCompleted(
                            node_name=name,
                            tool_name=tool_name,
                            result=result,
                            duration_ms=duration_ms,
                        )
                    )

                logger.debug(f"✅ Tool '{tool_name}' completed in {duration_ms:.2f}ms")

                return {
                    "result": result,
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "error": None,
                }
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.warning(f"❌ Tool '{tool_name}' failed after {duration_ms:.2f}ms: {e}")
                return {
                    "result": None,
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "error": str(e),
                }

        return self.create_node_with_mapping(
            name=name,
            wrapped_fn=execute_tool,
            input_schema={},  # No specific input schema (uses context)
            output_schema=ToolCallOutput,
            deps=deps,
            **kwargs,
        )
