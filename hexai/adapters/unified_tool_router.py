"""ToolRouter adapter that uses the global registry singleton for tool management."""

import asyncio
import inspect
import logging
from typing import Any

from hexai.core.application.nodes.tool_utils import ToolDefinition, ToolParameter
from hexai.core.ports.tool_router import ToolRouter
from hexai.core.registry import registry  # Use the direct module-level singleton
from hexai.core.registry.decorators import adapter
from hexai.core.registry.models import ClassComponent, ComponentType, FunctionComponent

logger = logging.getLogger(__name__)

__all__ = ["UnifiedToolRouter"]


@adapter(implements_port="tool_router")
class UnifiedToolRouter(ToolRouter):
    """ToolRouter adapter that uses the global registry singleton.

    This adapter is a thin wrapper around the global ComponentRegistry that:
    1. Retrieves tools from the registry
    2. Executes tools with proper async/sync handling
    3. Generates ToolDefinitions from registry metadata

    Example
    -------
        # Create router - no parameters needed!
        router = UnifiedToolRouter()

        # Execute tools from registry
        result = await router.acall_tool("search_papers", {"query": "diabetes"})

        # Get tool definitions for agent nodes
        tool_defs = router.get_tool_definitions()
    """

    def __init__(self) -> None:
        """Initialize the router - uses the global registry singleton."""
        # No initialization needed - we use get_registry() when needed
        pass

    async def acall_tool(self, tool_name: str, params: dict[str, Any]) -> Any:
        """Call a tool with parameters from the registry.

        Args
        ----
            tool_name: Name of the tool to execute
            params: Parameters to pass to the tool

        Returns
        -------
            Tool execution result

        Raises
        ------
            ValueError: If tool not found in registry
        """
        try:
            # Validate tool exists and is correct type
            registry.get_metadata(tool_name, component_type=ComponentType.TOOL)
            # Get and execute tool
            tool = registry.get(tool_name)
        except Exception as e:
            available = self.get_available_tools()
            raise ValueError(f"Tool '{tool_name}' not found. Available: {available}") from e

        # Execute tool outside the try/except so tool errors aren't wrapped
        return await self._execute_tool(tool, params)

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
        """Get list of available tool names from registry."""
        try:
            registry_tools = registry.list_components(component_type=ComponentType.TOOL)
            return [tool.name for tool in registry_tools]
        except Exception as e:
            logger.debug("Could not list registry tools: %s", e)
            return []

    def get_tool_schema(self, tool_name: str) -> dict[str, Any]:
        """Get schema for a specific tool from registry.

        Args
        ----
            tool_name: Name of the tool

        Returns
        -------
            Tool schema dictionary or empty dict if not found
        """
        try:
            tool_info = registry.get_info(tool_name)
            if tool_info.component_type == ComponentType.TOOL:
                tool_def = self._get_tool_definition_from_component(tool_info)
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
        except Exception as e:
            logger.debug("Could not get tool schema from registry: %s", e)
        return {}

    def get_all_tool_schemas(self) -> dict[str, dict[str, Any]]:
        """Get schemas for all available tools from registry."""
        schemas = {}
        for tool_name in self.get_available_tools():
            schema = self.get_tool_schema(tool_name)
            if schema:
                schemas[tool_name] = schema
        return schemas

    def get_tool_definitions(self) -> list[ToolDefinition]:
        """Get ToolDefinitions from registry for integration with ToolDescriptionManager.

        Returns
        -------
            List of ToolDefinitions generated from registry tools
        """
        definitions = []
        try:
            registry_tools = registry.list_components(component_type=ComponentType.TOOL)
            for tool_info in registry_tools:
                tool_def = self._get_tool_definition_from_component(tool_info)
                definitions.append(tool_def)
        except Exception as e:
            logger.debug("Could not get tool definitions from registry: %s", e)
        return definitions

    def _get_tool_definition_from_component(self, metadata: Any) -> ToolDefinition:
        """Generate ToolDefinition from registry component.

        Args
        ----
            metadata: ComponentMetadata from registry

        Returns
        -------
            ToolDefinition for the component
        """
        component = metadata.component

        # Try to extract from function or class
        target_func = None
        if isinstance(component, FunctionComponent):
            target_func = component.value
        elif isinstance(component, ClassComponent) and hasattr(component.value, "execute"):
            target_func = component.value.execute

        if target_func:
            return self._generate_tool_definition_from_function(target_func, metadata.name)
        else:
            # Return minimal definition for other cases
            return ToolDefinition(
                name=metadata.name,
                simplified_description=metadata.description or f"Tool {metadata.name}",
                detailed_description=metadata.description or f"Tool {metadata.name}",
                parameters=[],
                examples=[f"{metadata.name}()"],
            )

    def _generate_tool_definition_from_function(self, func: Any, tool_name: str) -> ToolDefinition:
        """Generate ToolDefinition from a function's signature and docstring.

        Args
        ----
            func: Function to analyze
            tool_name: Name for the tool

        Returns
        -------
            ToolDefinition with extracted metadata
        """
        sig = inspect.signature(func)

        # Extract parameters
        parameters: list[ToolParameter] = []
        for param_name, param in sig.parameters.items():
            if param_name == "self":  # Skip self parameter for methods
                continue

            param_type = (
                str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any"
            )
            # Clean up type string
            if "<class '" in param_type:
                param_type = param_type.replace("<class '", "").replace("'>", "")

            tool_param = ToolParameter(
                name=param_name,
                description=f"Parameter {param_name}",
                param_type=param_type,
                required=param.default == inspect.Parameter.empty,
                default=param.default if param.default != inspect.Parameter.empty else None,
            )
            parameters.append(tool_param)

        # Extract description from docstring
        doc = inspect.getdoc(func) or f"Execute {tool_name}"
        lines = doc.strip().split("\n")
        simplified_desc = lines[0] if lines else f"Execute {tool_name}"

        # Build example
        example_args = []
        for p in parameters:
            if p.required:
                if "str" in p.param_type:
                    example_args.append(f"{p.name}='example'")
                elif "int" in p.param_type:
                    example_args.append(f"{p.name}=10")
                else:
                    example_args.append(f"{p.name}=...")

        example = f"{tool_name}({', '.join(example_args)})" if example_args else f"{tool_name}()"

        return ToolDefinition(
            name=tool_name,
            simplified_description=simplified_desc,
            detailed_description=doc.strip() if len(doc.strip()) > 50 else simplified_desc,
            parameters=parameters,
            examples=[example],
        )
