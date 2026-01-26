"""ToolRouter adapter that aggregates multiple tool routers with namespacing."""

import asyncio
import inspect
from typing import Any

from hexdag.builtin.nodes.tool_utils import ToolDefinition, ToolParameter
from hexdag.core.exceptions import ResourceNotFoundError
from hexdag.core.logging import get_logger
from hexdag.core.ports.tool_router import ToolRouter
from hexdag.core.protocols import has_execute_method

logger = get_logger(__name__)

__all__ = ["UnifiedToolRouter"]


class UnifiedToolRouter(ToolRouter):
    """ToolRouter adapter that supports multiple tool sources with namespacing.

    This adapter aggregates multiple tool routers and provides unified access
    with namespace prefixes (e.g., "builtin::tool", "mcp::tool").

    Example
    -------
        router = UnifiedToolRouter(routers={
            "builtin": PythonToolRouter(),
            "mcp_sql": MCPToolRouter("sqlite"),
        })
        result = await router.acall_tool("builtin::search_papers", {"query": "..."})
    """

    def __init__(self, routers: dict[str, ToolRouter] | None = None, **kwargs: Any) -> None:
        """Initialize the router.

        Args
        ----
            routers: Dict of {router_id: ToolRouter} for multi-router mode
            **kwargs: Additional configuration options
        """
        self.routers = routers or {}

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

        Returns namespaced tools: ["router1::tool1", "router2::tool2"]
        """
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

    def get_tool_schema(self, tool_name: str) -> dict[str, Any]:
        """Get schema for a specific tool.

        Args
        ----
            tool_name: Name of the tool (with optional "router::" prefix)

        Returns
        -------
            Tool schema dictionary or empty dict if not found
        """
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

    def get_all_tool_schemas(self) -> dict[str, dict[str, Any]]:
        """Get schemas for all available tools.

        Returns schemas with namespaced keys.
        """
        schemas = {}
        for tool_name in self.get_available_tools():
            if schema := self.get_tool_schema(tool_name):
                schemas[tool_name] = schema
        return schemas

    async def aget_available_tools(self) -> list[str]:
        """Async version of get_available_tools."""
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

    async def aget_tool_schema(self, tool_name: str) -> dict[str, Any]:
        """Async version of get_tool_schema."""
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

    def get_tool_definitions(self) -> list[ToolDefinition]:
        """Get ToolDefinitions from all routers.

        Returns
        -------
            List of ToolDefinitions generated from router tools
        """
        definitions = []
        for router_id, router in self.routers.items():
            try:
                for tool_name in router.get_available_tools():
                    schema = router.get_tool_schema(tool_name)
                    if schema:
                        tool_def = ToolDefinition(
                            name=f"{router_id}::{tool_name}",
                            simplified_description=schema.get("description", f"Tool {tool_name}"),
                            detailed_description=schema.get("description", f"Tool {tool_name}"),
                            parameters=[
                                ToolParameter(
                                    name=p.get("name", ""),
                                    description=p.get("description", ""),
                                    param_type=p.get("type", "Any"),
                                    required=p.get("required", False),
                                    default=p.get("default"),
                                )
                                for p in schema.get("parameters", [])
                            ],
                            examples=[f"{router_id}::{tool_name}()"],
                        )
                        definitions.append(tool_def)
            except Exception as e:
                logger.debug("Could not get tool definitions from router %s: %s", router_id, e)
        return definitions
