"""Port interface for Tool Routers."""

import inspect
from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable


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
