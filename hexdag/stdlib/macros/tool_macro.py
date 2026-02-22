"""ToolMacro - Expand tool calls into parallel ToolCallNodes.

This macro enables dynamic parallel tool execution:
1. Takes a list of tool_calls from LLM
2. Creates a ToolCallNode for each tool
3. Executes them in parallel (via DAG waves)
4. Merges results back into conversation

Used by ReasoningAgent and Macro Agent for dynamic tool injection.
"""

from typing import Any

from hexdag.kernel.configurable import ConfigurableMacro, MacroConfig
from hexdag.kernel.domain.dag import DirectedGraph
from hexdag.kernel.logging import get_logger
from hexdag.stdlib.nodes import FunctionNode, ToolCallNode

logger = get_logger(__name__)


class ToolMacroConfig(MacroConfig):
    """Configuration for ToolMacro.

    Attributes
    ----------
    tool_calls : list[dict]
        List of tool calls from LLM in format:
        [
            {"id": "call_1", "name": "search", "arguments": {"query": "AI"}},
            {"id": "call_2", "name": "calc", "arguments": {"expr": "2+2"}}
        ]
    agent_name : str | None
        Name of agent requesting tools (for access control)
    allowed_tools : list[str] | None
        List of allowed tool names (if None, all tools allowed)
    """

    tool_calls: list[dict[str, Any]] = []
    agent_name: str | None = None
    allowed_tools: list[str] | None = None


class ToolMacro(ConfigurableMacro):
    """Expand tool calls into parallel ToolCallNodes.

    This macro creates a subgraph for dynamic parallel tool execution:

    Graph Structure:
    ----------------
    ```
    [dependencies] → tool_call_1 ┐
                  → tool_call_2 ├─→ merger → [output]
                  → tool_call_3 ┘
    ```

    All tool nodes execute in parallel (same wave), then results merge.

    Examples
    --------
    Basic usage::

        config = ToolMacroConfig(
            tool_calls=[
                {"id": "1", "name": "search", "arguments": {"query": "AI"}},
                {"id": "2", "name": "calc", "arguments": {"expression": "2+2"}}
            ]
        )

        macro = ToolMacro(config)
        graph = macro.expand(
            instance_name="agent_tools",
            inputs={},
            dependencies=["llm_node"]
        )

        # Graph contains:
        # - agent_tools_tool_0_search (depends on llm_node)
        # - agent_tools_tool_1_calc (depends on llm_node)
        # - agent_tools_merger (depends on both tools)

    With access control::

        config = ToolMacroConfig(
            tool_calls=[...],
            agent_name="research_agent",
            allowed_tools=["search", "summarize"]  # calc not allowed
        )

        macro = ToolMacro(config)
        graph = macro.expand(...)
        # Only creates nodes for allowed tools
    """

    Config = ToolMacroConfig

    def expand(
        self,
        instance_name: str,
        inputs: dict[str, Any],
        dependencies: list[str],
    ) -> DirectedGraph:
        """Expand tool calls into parallel execution graph.

        Parameters
        ----------
        instance_name : str
            Unique name for this macro instance (used as prefix for nodes)
        inputs : dict[str, Any]
            Input values (typically empty, config has the data)
        dependencies : list[str]
            Nodes that tool calls depend on (typically the LLM node)

        Returns
        -------
        DirectedGraph
            Graph with parallel ToolCallNodes and merger

        Examples
        --------
        Called during dynamic execution::

            # Agent's LLM returns tool_calls
            tool_calls = [
                {"id": "1", "name": "search", "arguments": {"query": "AI"}},
                {"id": "2", "name": "calc", "arguments": {"expr": "2+2"}}
            ]

            # Create macro config
            config = ToolMacroConfig(tool_calls=tool_calls)
            macro = ToolMacro(config)

            # Expand into graph
            subgraph = macro.expand(
                instance_name="agent_step_1_tools",
                inputs={},
                dependencies=["agent_step_1_llm"]
            )

            # Orchestrator merges subgraph into main graph
            # and executes tools in parallel
        """
        graph = DirectedGraph()
        config: ToolMacroConfig = self.config  # type: ignore[assignment]

        # Get tool calls from config
        tool_calls = config.tool_calls

        if not tool_calls:
            logger.debug(f"No tool calls for {instance_name}, returning empty graph")
            return self._create_passthrough(instance_name, dependencies)

        # Filter by allowed tools (access control)
        allowed_tools = self._get_allowed_tools(config)
        filtered_calls = self._filter_tool_calls(tool_calls, allowed_tools, config.agent_name)

        if not filtered_calls:
            logger.warning(f"All {len(tool_calls)} tool calls filtered out for {config.agent_name}")
            return self._create_passthrough(instance_name, dependencies)

        # Create ToolCallNode for each tool call
        tool_call_factory = ToolCallNode()
        tool_nodes = []

        for i, tc in enumerate(filtered_calls):
            tool_name = tc["name"]
            tool_call_id = tc.get("id", f"{instance_name}_{i}")
            arguments = tc.get("arguments", {})

            # Create node
            node = tool_call_factory(
                name=f"{instance_name}_tool_{i}_{tool_name}",
                tool_name=tool_name,
                arguments=arguments,
                tool_call_id=tool_call_id,
                deps=dependencies,  # All tools depend on LLM output
            )

            graph += node
            tool_nodes.append(node.name)

            logger.debug(
                f"Created tool node: {node.name} for tool '{tool_name}' with args {arguments}"
            )

        # Create merger node that consolidates all tool results
        merger = self._create_merger_node(instance_name, tool_nodes, filtered_calls)
        graph += merger

        logger.info(
            f"ToolMacro '{instance_name}' expanded to {len(tool_nodes)} "
            f"parallel tool nodes + merger"
        )

        return graph

    def _get_allowed_tools(self, config: ToolMacroConfig) -> set[str] | None:
        """Get set of allowed tools (None = all allowed).

        Parameters
        ----------
        config : ToolMacroConfig
            Macro configuration

        Returns
        -------
        set[str] | None
            Set of allowed tool names, or None if all allowed
        """
        if config.allowed_tools is None:
            return None  # All tools allowed

        return set(config.allowed_tools)

    def _filter_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        allowed_tools: set[str] | None,
        agent_name: str | None,
    ) -> list[dict[str, Any]]:
        """Filter tool calls by allowed tools (access control).

        Parameters
        ----------
        tool_calls : list[dict]
            Raw tool calls from LLM
        allowed_tools : set[str] | None
            Set of allowed tool names (None = all allowed)
        agent_name : str | None
            Name of agent (for logging)

        Returns
        -------
        list[dict]
            Filtered tool calls
        """
        if allowed_tools is None:
            return tool_calls  # No filtering

        filtered = []
        for tc in tool_calls:
            tool_name = tc["name"]
            if tool_name in allowed_tools:
                filtered.append(tc)
            else:
                logger.warning(
                    f"Tool '{tool_name}' not allowed for agent '{agent_name}' "
                    f"(allowed: {allowed_tools})"
                )

        return filtered

    def _create_merger_node(
        self,
        instance_name: str,
        tool_nodes: list[str],
        tool_calls: list[dict[str, Any]],
    ) -> Any:  # Returns NodeSpec
        """Create merger node that consolidates tool results.

        The merger node:
        1. Waits for all tool nodes to complete
        2. Collects results
        3. Formats them for the agent (tool messages)

        Parameters
        ----------
        instance_name : str
            Macro instance name
        tool_nodes : list[str]
            Names of tool nodes to wait for
        tool_calls : list[dict]
            Original tool calls (for matching)

        Returns
        -------
        NodeSpec
            Merger node specification
        """
        fn_factory = FunctionNode()

        async def merge_tool_results(input_data: dict[str, Any]) -> dict[str, Any]:
            """Merge results from all tool nodes.

            This function receives the outputs of all tool nodes and
            consolidates them into a format suitable for continuing
            the conversation with the LLM.

            Parameters
            ----------
            input_data : dict
                Contains results from all dependency nodes

            Returns
            -------
            dict
                Merged results with:
                - results: List of tool results
                - has_tools: True
                - tool_messages: Formatted for LLM conversation
            """
            results = []

            # Collect results from each tool node
            for node_name in tool_nodes:
                # Get result from this tool node
                tool_result = input_data.get(node_name)

                if tool_result:
                    # ToolCallNode returns ToolCallOutput (Pydantic model)
                    if hasattr(tool_result, "model_dump"):
                        tool_result = tool_result.model_dump()

                    results.append(tool_result)

            # Format as tool messages for LLM
            tool_messages = [
                {
                    "role": "tool",
                    "tool_call_id": result.get("tool_call_id"),
                    "name": result.get("tool_name"),
                    "content": str(result.get("result") or result.get("error", "Unknown error")),
                }
                for result in results
            ]

            logger.debug(f"Merged {len(results)} tool results into {len(tool_messages)} messages")

            return {
                "results": results,
                "has_tools": True,
                "tool_messages": tool_messages,
                "tool_count": len(results),
            }

        return fn_factory(
            name=f"{instance_name}_merger",
            fn=merge_tool_results,
            deps=tool_nodes,  # Wait for all tools
        )

    def _create_passthrough(self, instance_name: str, dependencies: list[str]) -> DirectedGraph:
        """Create passthrough node when no tools to execute.

        Used when:
        - No tool calls provided
        - All tool calls filtered out by access control

        Parameters
        ----------
        instance_name : str
            Macro instance name
        dependencies : list[str]
            Nodes to depend on

        Returns
        -------
        DirectedGraph
            Graph with single passthrough node
        """
        graph = DirectedGraph()
        fn_factory = FunctionNode()

        async def passthrough(input_data: dict[str, Any]) -> dict[str, Any]:
            """Passthrough when no tools."""
            return {
                "results": [],
                "has_tools": False,
                "tool_messages": [],
                "tool_count": 0,
            }

        passthrough_node = fn_factory(
            name=f"{instance_name}_no_tools",
            fn=passthrough,
            deps=dependencies,
        )

        graph += passthrough_node
        return graph
