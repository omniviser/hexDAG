"""Simple mock tool adapter with predefined responses for testing."""

from typing import Any

from hexdag.core.ports.tool_router import ToolRouter


class MockToolAdapter(ToolRouter):
    """Mock tool adapter that returns predefined responses.

    This is a simpler alternative to MockToolRouter, designed for
    unit tests and offline runs where you want predictable,
    predefined responses for specific tool calls.

    Example
    -------
    Example usage::

        mock_tools = {
            "search_customers": [{"id": 1, "name": "Alice"}],
            "get_product": {"id": 42, "name": "Widget", "price": 9.99}
        }
    """

    # Type annotations for attributes
    default_response: Any | None
    raise_on_unknown: bool
    mock_responses: dict[str, Any]
    call_history: list[dict[str, Any]]

    def __init__(
        self,
        mock_responses: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the mock tool adapter.

        Args
        ----
            mock_responses: Dictionary mapping tool names to their predefined responses.
            **kwargs: Configuration options (default_response, raise_on_unknown)
        """
        self.default_response = kwargs.get("default_response")
        self.raise_on_unknown = kwargs.get("raise_on_unknown", False)
        self.mock_responses = mock_responses or {}

        self.call_history: list[dict[str, Any]] = []

    async def acall_tool(self, tool_name: str, params: dict[str, Any]) -> Any:
        """Call a tool and return its predefined response.

        Args
        ----
            tool_name: Name of the tool to call
            params: Parameters passed to the tool (logged but not used)

        Returns
        -------
            The predefined response for the tool

        Raises
        ------
            ValueError: If tool not found and raise_on_unknown is True
        """
        self.call_history.append({
            "tool": tool_name,
            "params": params,
        })

        if tool_name in self.mock_responses:
            response = self.mock_responses[tool_name]
            # If response is callable, call it with params
            if callable(response):
                return response(params)
            return response

        if self.raise_on_unknown:
            available = ", ".join(self.mock_responses.keys())
            raise ValueError(f"Unknown tool: '{tool_name}'. Available tools: {available or 'none'}")

        if self.default_response is not None:
            return self.default_response

        return {
            "status": "success",
            "tool": tool_name,
            "message": f"Mock response for {tool_name}",
        }

    def get_available_tools(self) -> list[str]:
        """Get list of available tool names.

        Returns
        -------
            List of tool names that have predefined responses
        """
        return list(self.mock_responses.keys())

    def get_tool_schema(self, tool_name: str) -> dict[str, Any]:
        """Get schema for a specific tool.

        Since this is a mock adapter, it returns a basic schema.

        Args
        ----
            tool_name: Name of the tool

        Returns
        -------
            Basic schema for the tool
        """
        if tool_name in self.mock_responses:
            return {
                "name": tool_name,
                "description": f"Mock tool: {tool_name}",
                "parameters": [],  # Mock doesn't validate parameters
            }

        return {}

    def get_all_tool_schemas(self) -> dict[str, dict[str, Any]]:
        """Get schemas for all available tools.

        Returns
        -------
            Dictionary mapping tool names to their basic schemas
        """
        return {name: self.get_tool_schema(name) for name in self.mock_responses}

    # Utility methods for testing
    def set_response(self, tool_name: str, response: Any) -> None:
        """Set or update the response for a tool.

        Args
        ----
            tool_name: Name of the tool
            response: Response to return (can be callable)
        """
        self.mock_responses[tool_name] = response

    def remove_tool(self, tool_name: str) -> None:
        """Remove a tool from available tools.

        Args
        ----
            tool_name: Name of the tool to remove
        """
        self.mock_responses.pop(tool_name, None)

    def clear_history(self) -> None:
        """Clear the call history."""
        self.call_history.clear()

    def get_call_count(self, tool_name: str | None = None) -> int:
        """Get number of times a tool was called.

        Args
        ----
            tool_name: Name of specific tool, or None for all tools

        Returns
        -------
            Number of calls
        """
        if tool_name is None:
            return len(self.call_history)

        return sum(1 for call in self.call_history if call["tool"] == tool_name)

    def get_last_call(self, tool_name: str | None = None) -> dict[str, Any] | None:
        """Get the most recent call.

        Args
        ----
            tool_name: Name of specific tool, or None for any tool

        Returns
        -------
            Last call details or None if no calls
        """
        if not self.call_history:
            return None

        if tool_name is None:
            return self.call_history[-1]

        # Find last call for specific tool
        for call in reversed(self.call_history):
            if call["tool"] == tool_name:
                return call

        return None
