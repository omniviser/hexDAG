"""Mock tool router implementation for testing."""

import ast
import asyncio
import operator
from typing import Any

from hexdag.core.exceptions import ResourceNotFoundError
from hexdag.core.ports.tool_router import ToolRouter
from hexdag.core.registry import adapter


@adapter(name="mock_tool_router", implements_port="tool_router", namespace="plugin")
class MockToolRouter(ToolRouter):
    """Mock implementation of ToolRouter for testing."""

    # Type annotations for attributes
    available_tools: list[str]
    delay_seconds: float
    raise_on_unknown_tool: bool
    tools: dict[str, dict[str, Any]]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize mock tool router.

        Args
        ----
            **kwargs: Configuration options
        """
        self.available_tools = kwargs.get("available_tools", [])
        self.delay_seconds = kwargs.get("delay_seconds", 0.0)
        self.raise_on_unknown_tool = kwargs.get("raise_on_unknown_tool", False)

        # Default mock tools
        self.tools: dict[str, dict[str, Any]] = {
            "search": {
                "name": "search",
                "description": "Search for information",
                "parameters": {
                    "query": {"type": "string", "description": "Search query"},
                },
            },
            "calculate": {
                "name": "calculate",
                "description": "Perform calculations",
                "parameters": {
                    "expression": {"type": "string", "description": "Math expression"},
                },
            },
            "get_weather": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "location": {"type": "string", "description": "Location name"},
                },
            },
        }

        for tool_name in self.available_tools:
            if tool_name not in self.tools:
                self.tools[tool_name] = {
                    "name": tool_name,
                    "description": f"Mock tool: {tool_name}",
                    "parameters": {},
                }

        self.call_history: list[dict[str, Any]] = []

    async def acall_tool(self, tool_name: str, params: dict[str, Any]) -> Any:
        """Call a mock tool with parameters.

        Args
        ----
            tool_name: The name of the tool to call
            params: Parameters to pass to the tool

        Returns
        -------
            Mock result based on the tool

        Raises
        ------
        ResourceNotFoundError
            If tool not found and raise_on_unknown_tool is True
        """
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        self.call_history.append({
            "tool": tool_name,
            "params": params,
            "timestamp": asyncio.get_event_loop().time(),
        })

        if tool_name not in self.tools:
            if self.raise_on_unknown_tool:
                raise ResourceNotFoundError("tool", tool_name, list(self.tools.keys()))
            return {"error": f"Unknown tool: {tool_name}"}

        if tool_name == "search":
            query = params.get("query", "")
            return {
                "results": [
                    {"title": f"Result 1 for {query}", "url": "http://example.com/1"},
                    {"title": f"Result 2 for {query}", "url": "http://example.com/2"},
                ]
            }
        if tool_name == "calculate":
            expression = params.get("expression", "0")
            try:
                # Safe evaluation using ast for simple math expressions
                # Supports: +, -, *, /, //, %, **, and numbers

                ops: dict[type[ast.operator] | type[ast.unaryop], Any] = {
                    ast.Add: operator.add,
                    ast.Sub: operator.sub,
                    ast.Mult: operator.mul,
                    ast.Div: operator.truediv,
                    ast.FloorDiv: operator.floordiv,
                    ast.Mod: operator.mod,
                    ast.Pow: operator.pow,
                    ast.USub: operator.neg,
                    ast.UAdd: operator.pos,
                }

                def safe_eval(node: ast.AST) -> float | int:
                    """Safely evaluate AST nodes for math expressions.

                    Args
                    ----
                        node: AST node to evaluate

                    Returns
                    -------
                        float | int: The evaluated numeric result

                    Raises
                    ------
                        ValueError: If operation or values are not supported
                    """
                    if isinstance(node, ast.Constant):  # Python 3.8+
                        val = node.value
                        if not isinstance(val, (int, float)):
                            raise ValueError(f"Only numeric constants are allowed, got {type(val)}")
                        return val
                    if isinstance(node, ast.BinOp):
                        left = safe_eval(node.left)
                        right = safe_eval(node.right)
                        op_func = ops.get(type(node.op))
                        if op_func is None:
                            raise ValueError(f"Unsupported binary operation: {ast.dump(node)}")
                        result = op_func(left, right)
                        if not isinstance(result, (int, float)):
                            raise ValueError("Operation resulted in non-numeric value")
                        return result
                    if isinstance(node, ast.UnaryOp):
                        operand = safe_eval(node.operand)
                        op_func = ops.get(type(node.op))
                        if op_func is None:
                            raise ValueError(f"Unsupported unary operation: {ast.dump(node)}")
                        result = op_func(operand)
                        if not isinstance(result, (int, float)):
                            raise ValueError("Operation resulted in non-numeric value")
                        return result
                    raise ValueError(f"Unsupported operation: {ast.dump(node)}")

                tree = ast.parse(expression, mode="eval")
                result = safe_eval(tree.body)
                return {"result": result}
            except Exception as e:
                return {"error": str(e)}
        elif tool_name == "get_weather":
            location = params.get("location", "Unknown")
            return {
                "location": location,
                "temperature": 22,
                "conditions": "Partly cloudy",
                "humidity": 65,
            }
        elif tool_name == "tool_end":
            # Return the params directly for tool_end calls
            return params
        else:
            # Generic mock response for custom tools
            return {
                "tool": tool_name,
                "status": "success",
                "result": f"Mock result for {tool_name}",
            }

    def get_available_tools(self) -> list[str]:
        """Get list of available mock tool names.

        Returns
        -------
            List of tool names
        """
        return list(self.tools.keys())

    def get_tool_schema(self, tool_name: str) -> dict[str, Any]:
        """Get schema for a specific mock tool.

        Args
        ----
            tool_name: Name of the tool

        Returns
        -------
            Tool schema dictionary

        Raises
        ------
        KeyError
            If tool not found
        """
        if tool_name not in self.tools:
            raise KeyError(f"Tool not found: {tool_name}")
        return self.tools[tool_name].copy()

    def get_all_tool_schemas(self) -> dict[str, dict[str, Any]]:
        """Get schemas for all available mock tools.

        Returns
        -------
            Dictionary mapping tool names to their schemas
        """
        return {name: schema.copy() for name, schema in self.tools.items()}

    # Testing utilities
    def reset(self) -> None:
        """Reset call history for testing."""
        self.call_history.clear()

    def get_call_history(self) -> list[dict[str, Any]]:
        """Get the history of tool calls for testing."""
        return self.call_history.copy()

    def add_tool(self, name: str, description: str, parameters: dict[str, Any]) -> None:
        """Add a new mock tool for testing.

        Args
        ----
            name: Tool name
            description: Tool description
            parameters: Tool parameter schema
        """
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
