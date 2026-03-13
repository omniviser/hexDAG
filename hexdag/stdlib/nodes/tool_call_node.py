"""ToolCallNode - Execute a single tool call as a FunctionNode.

This node wraps a tool function and executes it as a node.
The raw return value of the tool is passed directly to downstream nodes.
"""

import contextlib
import inspect
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from hexdag.kernel.context import get_port
from hexdag.kernel.domain.dag import NodeSpec
from hexdag.kernel.logging import get_logger
from hexdag.kernel.ports.tool_router import ToolRouterEvent
from hexdag.kernel.resolver import resolve_function

from .base_node_factory import BaseNodeFactory

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


class ToolCallInput(BaseModel):
    """Input for a tool call."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    tool_call_id: str | None = None


class ToolCallNode(BaseNodeFactory, yaml_alias="tool_call_node"):
    """Execute a single tool call as a FunctionNode.

    This node is a simple wrapper that:
    1. Takes a tool name and arguments
    2. Resolves the tool function using the module path resolver
    3. Executes it and returns the raw result
    4. Emits ToolRouterEvent events

    Errors raise as exceptions (handled by the orchestrator's retry/error
    system) rather than being wrapped in an output envelope.

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
        # Returns the raw tool result directly

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

        tool1 = ToolCallNode()(name="tool_1", tool_name="search", ...)
        tool2 = ToolCallNode()(name="tool_2", tool_name="calc", ...)
        # Orchestrator executes them in parallel automatically
        # Ports injected automatically for tools that need them
    """

    # Studio UI metadata
    _hexdag_icon = "Wrench"
    _hexdag_color = "#f97316"  # orange-500

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
            tool_name: Full module path to the tool function (e.g., 'mymodule.my_tool')
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

        async def execute_tool(input_data: dict[str, Any]) -> Any:
            """Execute the tool call with event emission."""

            # Get observer for event emission (optional)
            observer_manager = None
            with contextlib.suppress(Exception):
                observer_manager = get_port("observer_manager")

            logger.debug("Executing tool '{}' with args: {}", tool_name, arguments)

            start_time = time.time()

            # Resolve tool function using module path
            tool_fn: Callable[..., Any] = resolve_function(tool_name)

            # Prepare tool arguments
            tool_kwargs = dict(arguments)

            # Execute tool (handle both sync and async)
            if inspect.iscoroutinefunction(tool_fn):
                result = await tool_fn(**tool_kwargs)
            else:
                result = tool_fn(**tool_kwargs)

            duration_ms = (time.time() - start_time) * 1000

            # Emit ToolRouterEvent
            if observer_manager:
                await observer_manager.notify(
                    ToolRouterEvent(
                        port_type="tool_router",
                        method="acall_tool",
                        node_name=name,
                        tool_name=tool_name,
                        params=arguments,
                        result=result,
                        duration_ms=duration_ms,
                    )
                )

            logger.debug("Tool '{}' completed in {:.2f}ms", tool_name, duration_ms)

            return result

        return self.create_node_with_mapping(
            name=name,
            wrapped_fn=execute_tool,
            input_schema={},  # No specific input schema (uses context)
            output_schema=None,
            deps=deps,
            **kwargs,
        )
