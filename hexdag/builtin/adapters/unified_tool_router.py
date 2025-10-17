"""ToolRouter adapter that uses the global registry singleton for tool management."""

import asyncio
import inspect
from typing import Any

from hexdag.builtin.nodes.tool_utils import ToolDefinition, ToolParameter
from hexdag.core.exceptions import ResourceNotFoundError
from hexdag.core.logging import get_logger
from hexdag.core.ports.tool_router import ToolRouter
from hexdag.core.protocols import has_execute_method
from hexdag.core.registry import registry  # Use the direct module-level singleton
from hexdag.core.registry.decorators import adapter
from hexdag.core.registry.models import ClassComponent, ComponentType, FunctionComponent

logger = get_logger(__name__)

__all__ = ["UnifiedToolRouter"]


@adapter(implements_port="tool_router")
class UnifiedToolRouter(ToolRouter):
    """ToolRouter adapter that supports multiple tool sources with namespacing.

    This adapter can aggregate multiple tool routers and provide unified access
    with namespace prefixes (e.g., "builtin::tool", "mcp::tool").

    When routers are provided, tools are namespaced by router ID.
    When no routers are provided, uses the global registry singleton (legacy mode).

    Example
    -------
        # Multi-router mode (new)
        router = UnifiedToolRouter(routers={
            "builtin": PythonToolRouter(),
            "mcp_sql": MCPToolRouter("sqlite"),
        })
        result = await router.acall_tool("builtin::search_papers", {"query": "..."})

        # Legacy mode (backward compatible)
        router = UnifiedToolRouter()
        result = await router.acall_tool("search_papers", {"query": "..."})
    """

    def __init__(self, routers: dict[str, ToolRouter] | None = None, **kwargs: Any) -> None:
        """Initialize the router.

        Args
        ----
            routers: Optional dict of {router_id: ToolRouter} for multi-router mode
            **kwargs: Additional configuration options
        """
        self.routers = routers or {}
        self.multi_router_mode = bool(self.routers)

        # Store first router as default for unprefixed calls
        self.default_router = next(iter(self.routers.values())) if self.routers else None

    async def acall_tool(self, tool_name: str, params: dict[str, Any]) -> Any:
        """Call a tool with parameters.

        Supports both namespaced ("router_id::tool_name") and plain tool names.

        Args
        ----
            tool_name: Name of the tool to execute (with optional "router::" prefix)
            params: Parameters to pass to the tool

        Returns
        -------
            Tool execution result
        """
        if self.multi_router_mode:
            return await self._acall_tool_multi_router(tool_name, params)
        return await self._acall_tool_legacy(tool_name, params)

    async def _acall_tool_multi_router(self, tool_name: str, params: dict[str, Any]) -> Any:
        """Call tool in multi-router mode with namespace support."""
        # Parse namespace: "builtin::tool" or "tool"
        if "::" in tool_name:
            router_id, actual_tool = tool_name.split("::", 1)
            if router_id not in self.routers:
                available_routers = list(self.routers.keys())
                raise ResourceNotFoundError(
                    "tool_router",
                    router_id,
                    available_routers,
                )
            router = self.routers[router_id]
        else:
            # No prefix: use default router
            if not self.default_router:
                raise ResourceNotFoundError(
                    "tool",
                    tool_name,
                    [],
                )
            router = self.default_router
            actual_tool = tool_name

        # Delegate to the specific router
        return await router.acall_tool(actual_tool, params)

    async def _acall_tool_legacy(self, tool_name: str, params: dict[str, Any]) -> Any:
        """Call tool in legacy mode (global registry)."""
        try:
            # Validate tool exists and is correct type
            registry.get_metadata(tool_name, component_type=ComponentType.TOOL)
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
        """Get list of available tool names.

        In multi-router mode, returns namespaced tools: ["router1::tool1", "router2::tool2"]
        In legacy mode, returns tools from registry: ["tool1", "tool2"]
        """
        if self.multi_router_mode:
            all_tools = []
            for router_id, router in self.routers.items():
                try:
                    router_tools = router.get_available_tools()
                    # Prefix each tool with router ID
                    namespaced_tools = [f"{router_id}::{tool}" for tool in router_tools]
                    all_tools.extend(namespaced_tools)
                except Exception as e:
                    logger.debug("Could not list tools from router %s: %s", router_id, e)
            return all_tools
        # Legacy mode: registry tools
        try:
            registry_tools = registry.list_components(component_type=ComponentType.TOOL)
            return [tool.name for tool in registry_tools]
        except Exception as e:
            logger.debug("Could not list registry tools: %s", e)
            return []

    def get_tool_schema(self, tool_name: str) -> dict[str, Any]:
        """Get schema for a specific tool.

        Args
        ----
            tool_name: Name of the tool (with optional "router::" prefix in multi-router mode)

        Returns
        -------
            Tool schema dictionary or empty dict if not found
        """
        if self.multi_router_mode:
            # Parse namespace
            router_id: str
            if "::" in tool_name:
                router_id, actual_tool = tool_name.split("::", 1)
                if router_id not in self.routers:
                    return {}
                router = self.routers[router_id]
            else:
                if not self.default_router:
                    return {}
                router = self.default_router
                router_id = "default"
                actual_tool = tool_name

            # Get schema from specific router
            try:
                schema = router.get_tool_schema(actual_tool)
                # Add router info
                schema["_router"] = router_id
                return schema
            except Exception as e:
                logger.debug("Could not get tool schema from router: %s", e)
                return {}
        else:
            # Legacy mode: registry
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
        """Get schemas for all available tools.

        In multi-router mode, returns schemas with namespaced keys.
        In legacy mode, returns schemas from registry.
        """
        schemas = {}
        for tool_name in self.get_available_tools():
            if schema := self.get_tool_schema(tool_name):
                schemas[tool_name] = schema
        return schemas

    async def aget_available_tools(self) -> list[str]:
        """Async version of get_available_tools."""
        if self.multi_router_mode:
            all_tools = []
            for router_id, router in self.routers.items():
                try:
                    # Use async method if available
                    if hasattr(router, "aget_available_tools"):
                        router_tools = await router.aget_available_tools()
                    else:
                        router_tools = router.get_available_tools()
                    namespaced_tools = [f"{router_id}::{tool}" for tool in router_tools]
                    all_tools.extend(namespaced_tools)
                except Exception as e:
                    logger.debug("Could not list tools from router %s: %s", router_id, e)
            return all_tools
        return self.get_available_tools()

    async def aget_tool_schema(self, tool_name: str) -> dict[str, Any]:
        """Async version of get_tool_schema."""
        if self.multi_router_mode:
            # Parse namespace
            router_id: str
            if "::" in tool_name:
                router_id, actual_tool = tool_name.split("::", 1)
                if router_id not in self.routers:
                    return {}
                router = self.routers[router_id]
            else:
                if not self.default_router:
                    return {}
                router = self.default_router
                router_id = "default"
                actual_tool = tool_name

            # Get schema from specific router (async if available)
            try:
                if hasattr(router, "aget_tool_schema"):
                    schema = await router.aget_tool_schema(actual_tool)
                else:
                    schema = router.get_tool_schema(actual_tool)
                schema["_router"] = router_id
                return schema
            except Exception as e:
                logger.debug("Could not get tool schema from router: %s", e)
                return {}
        else:
            return self.get_tool_schema(tool_name)

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

        match component:
            case FunctionComponent(value=func):
                return self._generate_tool_definition_from_function(func, component_info.name)
            case ClassComponent(value=cls) if has_execute_method(cls):
                return self._generate_tool_definition_from_function(
                    cls.execute, component_info.name
                )
            case _:
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

        doc = inspect.getdoc(func) or f"Execute {tool_name}"
        lines = doc.strip().split("\n")
        simplified_desc = lines[0] if lines else f"Execute {tool_name}"

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
