"""Function-based ToolRouter adapter that integrates with existing tool architecture."""

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from hexai.core.application.nodes.tool_utils import ToolDefinition, ToolParameter
from hexai.core.ports.tool_router import ToolRouter

__all__ = ["FunctionBasedToolRouter"]


class FunctionBasedToolRouter(ToolRouter):
    """Real ToolRouter implementation that auto-generates ToolDefinitions from functions.

    This adapter bridges the gap between the existing tool architecture and real functions:
    1. Register actual callable functions as tools
    2. Auto-generate ToolDefinitions for ToolDescriptionManager
    3. Execute real functions when called by EnhancedToolParser
    4. Integrate seamlessly with existing agent workflow

    Key Integration Points:
    - Generates ToolDefinitions that work with ToolDescriptionManager
    - Executes real functions when ToolRouter.call_tool() is invoked
    - Compatible with ToolParser's INVOKE_TOOL: parsing

    Example
    -------
        router = FunctionBasedToolRouter()

        @router.tool
        async def search_papers(query: str, limit: int = 10) -> dict:
            '''Search for research papers in medical databases.'''
            return {"papers": [...], "count": 5}

        # Auto-generates ToolDefinition for ToolDescriptionManager
        tool_defs = router.get_tool_definitions()

        # Executes real function when agent calls it
        result = await router.call_tool("search_papers", {"query": "diabetes"})
    """

    def __init__(self) -> None:
        """Initialize the router with empty tools registry."""
        self.tools: dict[str, Callable[..., Any]] = {}
        self.tool_definitions: dict[str, ToolDefinition] = {}
        self.call_history: list[dict[str, Any]] = []

        # Register built-in tools
        self._register_builtin_tools()

    def _register_builtin_tools(self) -> None:
        """Register built-in tools that are always available."""

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

        # Register TOOL_CHANGE_PHASE for phase transitions
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

        self.register_function(tool_end, "tool_end")
        self.register_function(tool_end, "end")  # Alias
        self.register_function(change_phase, "change_phase")
        self.register_function(change_phase, "phase")  # Alias

    def tool(self, func: Callable) -> Callable:
        """Register a function as a tool decorator.

        Args
        ----
            func: The function to register as a tool

        Returns
        -------
            The original function (unmodified)
        """
        self.register_function(func)
        return func

    def register_function(self, func: Callable, name: str | None = None) -> None:
        """Register a function as a tool and auto-generate ToolDefinition.

        Args
        ----
            func: The function to register
            name: Optional custom name (defaults to function name)
        """
        tool_name = name or func.__name__
        self.tools[tool_name] = func
        self.tool_definitions[tool_name] = self._generate_tool_definition(func, tool_name)

    def _generate_tool_definition(self, func: Callable, tool_name: str) -> ToolDefinition:
        """Auto-generate ToolDefinition from function signature and docstring.

        This creates ToolDefinitions that integrate with the existing ToolDescriptionManager.

        Args:
        ----
            func: Function to analyze
            tool_name: Name to use for the tool

        Returns
        -------
            ToolDefinition compatible with existing architecture
        """
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Extract parameters with type information
        parameters: list[ToolParameter] = []
        for param_name, inspect_param in sig.parameters.items():
            param_type = type_hints.get(param_name, str)

            # Convert type to string for ToolParameter
            type_str = param_type.__name__ if hasattr(param_type, "__name__") else str(param_type)

            tool_param = ToolParameter(
                name=param_name,
                description=f"Parameter {param_name} of type {type_str}",
                param_type=type_str,
                required=inspect_param.default == inspect.Parameter.empty,
                default=(
                    inspect_param.default
                    if inspect_param.default != inspect.Parameter.empty
                    else None
                ),
            )
            parameters.append(tool_param)

        # Extract description from docstring
        doc = func.__doc__ or f"Execute {tool_name} function"

        # Split docstring into simplified and detailed descriptions
        lines = doc.strip().split("\n")
        simplified_desc = lines[0] if lines else f"Execute {tool_name}"
        detailed_desc = (
            doc.strip()
            if len(doc.strip()) > 50
            else f"Execute {tool_name} with provided parameters"
        )

        # Create examples based on function signature
        if parameters:
            example_params = []
            for param in parameters:
                if param.required:
                    if param.param_type == "str":
                        example_params.append(f"{param.name}='example'")
                    elif param.param_type == "int":
                        example_params.append(f"{param.name}=10")
                    else:
                        example_params.append(f"{param.name}='{param.param_type}_value'")

            example = (
                f"{tool_name}({', '.join(example_params)})" if example_params else f"{tool_name}()"
            )
        else:
            example = f"{tool_name}()"

        return ToolDefinition(
            name=tool_name,
            simplified_description=simplified_desc,
            detailed_description=detailed_desc,
            parameters=parameters,
            examples=[example],
        )

    async def aroute(self, tool_name: str, input_data: Any) -> Any:
        """Route a tool call to the registered function (legacy interface)."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found. Available: {list(self.tools.keys())}")

        try:
            tool_func = self.tools[tool_name]

            # Handle different input formats
            if isinstance(input_data, dict):
                # Extract parameters from dict
                sig = inspect.signature(tool_func)

                # Check if function accepts **kwargs
                has_var_keyword = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                )

                if has_var_keyword:
                    # For **kwargs functions, pass all input data
                    kwargs = input_data
                else:
                    # For regular functions, match parameters
                    kwargs = {}
                    for param_name in sig.parameters:
                        if param_name in input_data:
                            kwargs[param_name] = input_data[param_name]

                if asyncio.iscoroutinefunction(tool_func):
                    result = await tool_func(**kwargs)
                else:
                    result = tool_func(**kwargs)
            else:
                # Single parameter case
                if asyncio.iscoroutinefunction(tool_func):
                    result = await tool_func(input_data)
                else:
                    result = tool_func(input_data)

        except Exception as e:
            # Re-raise the exception instead of returning error string
            raise e

        # Log the call
        self.call_history.append(
            {
                "tool_name": tool_name,
                "input_data": input_data,
                "result": result,
            }
        )

        return result

    async def call_tool(self, tool_name: str, params: dict[str, Any]) -> Any:
        """Call a tool with parameters (main interface used by agents)."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found. Available: {list(self.tools.keys())}")

        return await self.aroute(tool_name, params)

    def get_available_tools(self) -> list[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())

    def get_tool_schema(self, tool_name: str) -> dict[str, Any]:
        """Get schema for a specific tool."""
        if tool_name in self.tool_definitions:
            tool_def = self.tool_definitions[tool_name]
            return {
                "name": tool_def.name,
                "description": tool_def.simplified_description,
                "detailed_description": tool_def.detailed_description,
                "parameters": [
                    {
                        "name": p.name,
                        "type": p.param_type,
                        "required": p.required,
                        "default": p.default,
                        "description": p.description,
                    }
                    for p in tool_def.parameters
                ],
                "examples": tool_def.examples,
            }
        return {}

    def get_all_tool_schemas(self) -> dict[str, dict[str, Any]]:
        """Get schemas for all available tools."""
        return {name: self.get_tool_schema(name) for name in self.tools}

    def get_tool_definitions(self) -> list[ToolDefinition]:
        """Get ToolDefinitions for integration with ToolDescriptionManager.

        This is the key integration method - it returns ToolDefinitions that
        can be passed to agent nodes and managed by ToolDescriptionManager.

        Returns
        -------
            List of ToolDefinitions generated from registered functions
        """
        return list(self.tool_definitions.values())

    def get_call_history(self) -> list[dict[str, Any]]:
        """Get call history for debugging."""
        return self.call_history.copy()

    def reset(self) -> None:
        """Reset call history."""
        self.call_history.clear()
