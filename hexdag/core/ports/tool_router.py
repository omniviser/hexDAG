"""Port interface for Tool Routers."""

from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ToolRouter(Protocol):
    """Protocol for routing tool calls."""

    # Required methods
    @abstractmethod
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

    @abstractmethod
    def get_available_tools(self) -> list[str]:
        """Get list of available tool names (sync version for local adapters).

        Returns
        -------
            List of tool names that can be called
        """
        ...

    async def aget_available_tools(self) -> list[str]:
        """Get list of available tool names (async version for remote/MCP adapters).

        Default implementation delegates to sync version. Override for true async behavior.

        Returns
        -------
            List of tool names that can be called
        """
        return self.get_available_tools()

    def get_tool_schema(self, tool_name: str) -> dict[str, Any]:
        """Get schema for a specific tool (sync version).

        Args
        ----
            tool_name: Name of the tool

        Returns
        -------
            Dictionary containing tool schema (name, description, parameters)
        """
        ...

    async def aget_tool_schema(self, tool_name: str) -> dict[str, Any]:
        """Get schema for a specific tool (async version for remote/MCP adapters).

        Default implementation delegates to sync version. Override for true async behavior.

        Args
        ----
            tool_name: Name of the tool

        Returns
        -------
            Dictionary containing tool schema (name, description, parameters)
        """
        return self.get_tool_schema(tool_name)

    def get_all_tool_schemas(self) -> dict[str, dict[str, Any]]:
        """Get schemas for all available tools (sync version).

        Returns
        -------
            Dictionary mapping tool names to their schemas
        """
        ...

    async def aget_all_tool_schemas(self) -> dict[str, dict[str, Any]]:
        """Get schemas for all available tools (async version for remote/MCP adapters).

        Default implementation delegates to sync version. Override for true async behavior.

        Returns
        -------
            Dictionary mapping tool names to their schemas
        """
        return self.get_all_tool_schemas()
