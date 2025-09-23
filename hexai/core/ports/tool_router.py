"""Port interface for Tool Routers."""

from typing import Any, Protocol, runtime_checkable

from hexai.core.registry.decorators import port


@port(
    name="tool_router",
    namespace="core",
    required_methods=["acall_tool", "get_available_tools"],
    optional_methods=["get_tool_schema", "get_all_tool_schemas"],
)
@runtime_checkable
class ToolRouter(Protocol):
    """Protocol for routing tool calls."""

    # Required methods
    async def acall_tool(self, tool_name: str, params: dict[str, Any]) -> Any:
        """Call a tool with parameters.

        Args
        ----
            tool_name: The name of the tool to call.
            params: Parameters to pass to the tool.

        Returns
        -------
            The result of the tool call.
        """
        ...

    def get_available_tools(self) -> list[str]:
        """Get list of available tool names.

        Returns
        -------
            List of tool names that can be called
        """
        ...

    # Optional methods for enhanced functionality
    def get_tool_schema(self, tool_name: str) -> dict[str, Any]:
        """Get schema for a specific tool.

        Args
        ----
            tool_name: Name of the tool

        Returns
        -------
            Dictionary containing tool schema (name, description, parameters)
        """
        ...

    def get_all_tool_schemas(self) -> dict[str, dict[str, Any]]:
        """Get schemas for all available tools.

        Returns
        -------
            Dictionary mapping tool names to their schemas
        """
        ...
