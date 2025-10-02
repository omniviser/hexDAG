"""ToolRouter adapter that uses the global registry singleton for tool management."""

import asyncio
import inspect
import logging
from typing import Any

from pydantic import BaseModel

from hexai.core.application.nodes.tool_utils import ToolDefinition, ToolParameter
from hexai.core.exceptions import ResourceNotFoundError
from hexai.core.ports.configurable import ConfigurableComponent
from hexai.core.ports.tool_router import ToolRouter
from hexai.core.protocols import has_execute_method
from hexai.core.registry import registry  # Use the direct module-level singleton
from hexai.core.registry.decorators import adapter
from hexai.core.registry.models import ClassComponent, ComponentType, FunctionComponent

logger = logging.getLogger(__name__)

__all__ = ["UnifiedToolRouter"]


@adapter(implements_port="tool_router")
class UnifiedToolRouter(ToolRouter, ConfigurableComponent):
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

    # Configuration schema for TOML generation
    class Config(BaseModel):
        """Configuration schema for Unified Tool Router."""

        # No configuration needed for this adapter as it uses the global registry
        pass

    @classmethod
    def get_config_class(cls) -> type[BaseModel]:
        """Return configuration schema."""
        return cls.Config

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the router - uses the global registry singleton.

        Args
        ----
            **kwargs: Configuration options (none needed for this adapter)
        """
        # Create config from kwargs using the Config schema
        config_data = {}
        for field_name in self.Config.model_fields:
            if field_name in kwargs:
                config_data[field_name] = kwargs[field_name]

        # Create and validate config (empty for this adapter)
        config = self.Config(**config_data)
        self.config = config

        # No other initialization needed - we use get_registry() when needed

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
        ResourceNotFoundError
            If tool not found in registry
        """
        try:
            # Validate tool exists and is correct type
            registry.get_metadata(tool_name, component_type=ComponentType.TOOL)
            # Get and execute tool
            tool = registry.get(tool_name)
        except Exception as e:
            available = self.get_available_tools()
            raise ResourceNotFoundError("tool", tool_name, available) from e

        # Execute tool outside the try/except so tool errors aren't wrapped
        return await self._execute_tool(tool, params)

    async def _execute_tool(self, tool: Any, params: dict[str, Any]) -> Any:
        """Execute any tool (function, class, or instance) with parameters.

        Raises
        ------
        ValueError
            If tool is not executable
        """
        try:
            # Handle different tool types
            if inspect.iscoroutinefunction(tool):
                return await self._call_with_params(tool, params, is_async=True)
            if has_execute_method(tool):
                # Class with execute method (protocol-based check)
                execute_method = tool.execute
                if asyncio.iscoroutinefunction(execute_method):
                    return await self._call_with_params(execute_method, params, is_async=True)
                return self._call_with_params(execute_method, params, is_async=False)
            if callable(tool):
                # Regular function
                return self._call_with_params(tool, params, is_async=False)
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

    def _get_tool_definition_from_component(self, component_info: Any) -> ToolDefinition:
        """Generate ToolDefinition from registry component.

        Args
        ----
            component_info: ComponentInfo from registry

        Returns
        -------
            ToolDefinition for the component
        """
        # Access metadata from ComponentInfo
        metadata = component_info.metadata
        component = metadata.component

        # Extract function from component using pattern matching (Python 3.12+)
        match component:
            case FunctionComponent(value=func):
                return self._generate_tool_definition_from_function(func, component_info.name)
            case ClassComponent(value=cls) if has_execute_method(cls):
                return self._generate_tool_definition_from_function(
                    cls.execute, component_info.name
                )
            case _:
                # Return minimal definition for other cases
                return ToolDefinition(
                    name=component_info.name,
                    simplified_description=metadata.description or f"Tool {component_info.name}",
                    detailed_description=metadata.description or f"Tool {component_info.name}",
                    parameters=[],
                    examples=[f"{component_info.name}()"],
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
