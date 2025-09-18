"""Unified ToolRouter adapter supporting both direct registration and ComponentRegistry."""

import asyncio
import inspect
import logging
from collections.abc import Callable
from typing import Any, get_type_hints

from hexai.core.application.nodes.tool_utils import ToolDefinition, ToolParameter
from hexai.core.ports.tool_router import ToolRouter
from hexai.core.registry.decorators import adapter
from hexai.core.registry.models import (
    ClassComponent,
    ComponentType,
    FunctionComponent,
    InstanceComponent,
)

logger = logging.getLogger(__name__)

__all__ = ["UnifiedToolRouter"]


@adapter(
    implements_port="tool_router",
    name="unified_tool_router",
    namespace="core",
    description="Unified tool router supporting direct registration and ComponentRegistry",
)
class UnifiedToolRouter(ToolRouter):
    """Unified ToolRouter that supports multiple tool registration patterns.

    This adapter supports:
    1. Direct function registration (backward compatible)
    2. ComponentRegistry tools with port injection
    3. Auto-generation of ToolDefinitions
    4. Future MCP server integration

    Key Features:
    - Backward compatible with direct function registration
    - Integrates with ComponentRegistry for centralized tool management
    - Supports port injection for tools requiring database/API connections
    - Generates ToolDefinitions automatically from functions and classes

    Example
    -------
        # Direct registration (backward compatible)
        router = UnifiedToolRouter()

        @router.tool
        async def search_papers(query: str, limit: int = 10) -> dict:
            '''Search for research papers in medical databases.'''
            return {"papers": [...], "count": 5}

        # Auto-generates ToolDefinition for ToolDescriptionManager
        tool_defs = router.get_tool_definitions()

        # With ComponentRegistry support
        router = UnifiedToolRouter(component_registry=registry, port_registry=ports)

        # Both patterns work seamlessly
        result = await router.acall_tool("search_papers", {"query": "diabetes"})
    """

    def __init__(
        self, component_registry: Any | None = None, port_registry: Any | None = None
    ) -> None:
        """Initialize the unified router with optional ComponentRegistry support.

        Args
        ----
            component_registry: Optional ComponentRegistry for tool lookup
            port_registry: Optional PortRegistry for dependency injection
        """
        self.component_registry = component_registry
        self.port_registry = port_registry

        # Direct registration storage (backward compatibility)
        self.tools: dict[str, Callable[..., Any]] = {}
        self.tool_definitions: dict[str, ToolDefinition] = {}

        # Tool instance cache for registry tools
        self._registry_instances: dict[str, Any] = {}

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

    async def acall_tool(self, tool_name: str, params: dict[str, Any]) -> Any:
        """Call a tool with parameters."""
        # Try direct registered tools first
        if tool_name in self.tools:
            return await self._execute_direct_tool(tool_name, params)

        # Try component registry if available
        if self.component_registry:
            tool = self._get_or_create_from_registry(tool_name)
            if tool:
                return await self._execute_tool(tool, params)

        # Tool not found
        available = self.get_available_tools()
        raise ValueError(f"Tool '{tool_name}' not found. Available: {available}")

    async def _execute_direct_tool(self, tool_name: str, params: dict[str, Any]) -> Any:
        """Execute a directly registered tool."""

        tool_func = self.tools[tool_name]
        result = await self._execute_tool(tool_func, params)

        # Log the call
        self.call_history.append(
            {
                "tool_name": tool_name,
                "input_data": params,
                "result": result,
            }
        )

        return result

    async def _execute_tool(self, tool: Any, params: dict[str, Any]) -> Any:
        """Execute any tool (function, class, or instance) with parameters."""
        try:
            # Handle different tool types
            if inspect.iscoroutinefunction(tool):
                return await self._call_with_params(tool, params, is_async=True)
            elif hasattr(tool, "execute"):
                # Class with execute method
                execute_method = tool.execute
                if asyncio.iscoroutinefunction(execute_method):
                    return await self._call_with_params(execute_method, params, is_async=True)
                else:
                    return self._call_with_params(execute_method, params, is_async=False)
            elif callable(tool):
                # Regular function
                return self._call_with_params(tool, params, is_async=False)
            else:
                raise ValueError(f"Tool {tool} is not executable")

        except Exception as e:
            raise e

    def _call_with_params(self, func: Any, params: dict[str, Any], is_async: bool) -> Any:
        """Call function with appropriate parameters.

        Args
        ----
            func: Function to call
            params: Parameters dict
            is_async: Whether function is async

        Returns
        -------
            Function result (or coroutine if async)
        """
        sig = inspect.signature(func)

        # Check if function accepts **kwargs
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

        if has_var_keyword:
            kwargs = params
        else:
            # Filter to only expected parameters
            kwargs = {}
            for param_name in sig.parameters:
                if param_name in params:
                    kwargs[param_name] = params[param_name]

        if is_async:
            return func(**kwargs)  # Return coroutine, caller will await
        else:
            return func(**kwargs)

    def get_available_tools(self) -> list[str]:
        """Get list of available tool names."""
        tools = list(self.tools.keys())

        # Add registry tools if available
        if self.component_registry:
            try:
                registry_tools = self.component_registry.list_components(
                    component_type=ComponentType.TOOL
                )
                tools.extend(tool.name for tool in registry_tools)
            except Exception as e:
                logger.debug("Could not list registry tools: %s", e)

        return tools

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
        definitions = list(self.tool_definitions.values())

        # Add registry tools if available
        if self.component_registry:
            try:
                registry_tools = self.component_registry.list_components(
                    component_type=ComponentType.TOOL
                )
                for tool_info in registry_tools:
                    tool_def = self._get_tool_definition_from_metadata(tool_info.metadata)
                    definitions.append(tool_def)
            except Exception as e:
                logger.debug("Could not get tool definitions from registry: %s", e)

        return definitions

    def get_call_history(self) -> list[dict[str, Any]]:
        """Get call history for debugging."""
        return self.call_history.copy()

    def reset(self) -> None:
        """Reset call history and caches."""
        self.call_history.clear()
        self._registry_instances.clear()

    def _get_or_create_from_registry(self, tool_name: str) -> Any | None:
        """Get or create tool from component registry.

        Args
        ----
            tool_name: Tool name

        Returns
        -------
            Tool instance or None if not found
        """
        if tool_name in self._registry_instances:
            return self._registry_instances[tool_name]

        if not self.component_registry:
            return None

        try:
            metadata = self.component_registry.get_metadata(
                tool_name, component_type=ComponentType.TOOL
            )
        except Exception as e:
            logger.debug("Tool %s not found in registry: %s", tool_name, e)
            return None

        # Create instance based on component type
        component = metadata.component

        if isinstance(component, FunctionComponent):
            tool = component.value
        elif isinstance(component, ClassComponent):
            # Instantiate with port injection if needed
            tool = self._instantiate_with_ports(component.value, metadata)
        elif isinstance(component, InstanceComponent):
            tool = component.value
        else:
            return None

        self._registry_instances[tool_name] = tool
        return tool

    def _instantiate_with_ports(self, tool_class: type, metadata: Any) -> Any:
        """Instantiate tool class with port injection.

        Args
        ----
            tool_class: Class to instantiate
            metadata: Component metadata

        Returns
        -------
            Tool instance
        """
        # Check if ports are required
        if metadata.tool_metadata and metadata.tool_metadata.required_ports and self.port_registry:
            ports = {}
            for param_name, port_type in metadata.tool_metadata.required_ports.items():
                adapter = self.port_registry.get_adapter(port_type)
                ports[param_name] = adapter
            return tool_class(**ports)
        else:
            return tool_class()

    def _get_tool_definition_from_metadata(self, metadata: Any) -> ToolDefinition:
        """Get or generate ToolDefinition from component metadata.

        Args
        ----
            metadata: Component metadata

        Returns
        -------
            ToolDefinition
        """
        component = metadata.component

        if isinstance(component, FunctionComponent):
            return self._generate_tool_definition(component.value, metadata.name)
        elif isinstance(component, ClassComponent):
            # Generate from class's execute method if it has one
            tool_class = component.value
            if hasattr(tool_class, "execute"):
                return self._generate_tool_definition(tool_class.execute, metadata.name)
            else:
                # Minimal definition
                return ToolDefinition(
                    name=metadata.name,
                    simplified_description=metadata.description or f"Tool {metadata.name}",
                    detailed_description=metadata.description or f"Tool {metadata.name}",
                    parameters=[],
                    examples=[f"{metadata.name}()"],
                )
        else:
            # Instance or unknown, return minimal definition
            return ToolDefinition(
                name=metadata.name,
                simplified_description=metadata.description or f"Tool {metadata.name}",
                detailed_description=metadata.description or f"Tool {metadata.name}",
                parameters=[],
                examples=[f"{metadata.name}()"],
            )
