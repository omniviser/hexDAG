"""Port interface for Tool Routers."""

from typing import Any, Protocol


class ToolRouter(Protocol):
    """Protocol for routing tool calls."""

    async def aroute(self, tool_name: str, input_data: Any) -> Any:
        """Route a tool call asynchronously.

        Args
        ----
            tool_name: The name of the tool to route to.
            input_data: The input data for the tool.

        Returns
        -------
            The result of the tool call.
        """
        ...

    async def call_tool(self, tool_name: str, params: dict[str, Any]) -> Any:
        """Call a tool with parameters (compatibility method for agent nodes)."""
        ...

    def get_available_tools(self) -> list[str]:
        """Get list of available tool names.

        Returns
        -------
            List of tool names that can be called
        """
        ...

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
