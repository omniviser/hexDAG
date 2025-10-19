"""ReasoningAgentMacro - Multi-step reasoning with tools as FunctionNodes.

Architecture:
- Chain of LLM nodes for reasoning steps
- Tools as FunctionNodes (parallel to LLM chain, invoked on-demand)
- Internal memory (reasoning history passed between steps)
- Multi-tool support (agent can use multiple tools)
- No external coordination (self-contained)
"""

from typing import Any

from pydantic import field_validator

from hexdag.builtin.nodes.function_node import FunctionNode
from hexdag.builtin.nodes.llm_node import LLMNode
from hexdag.core.configurable import ConfigurableMacro, MacroConfig
from hexdag.core.domain.dag import DirectedGraph
from hexdag.core.logging import get_logger
from hexdag.core.orchestration.prompt import PromptTemplate
from hexdag.core.registry import macro

logger = get_logger(__name__)


class ReasoningAgentConfig(MacroConfig):
    """Configuration for ReasoningAgentMacro.

    Attributes
    ----------
    main_prompt : str
        Primary prompt for reasoning
    max_steps : int
        Maximum reasoning iterations (default: 3)
    tools : list[str]
        Tool names available to the agent (e.g., ["search", "calculate"])
    """

    main_prompt: str
    max_steps: int = 3
    tools: list[str] = []

    @field_validator("main_prompt", mode="before")
    @classmethod
    def convert_prompt_input(cls, v: Any) -> str:
        """Convert PromptInput to string."""
        if isinstance(v, str):
            return v
        if isinstance(v, PromptTemplate):
            return v.template
        if hasattr(v, "template"):
            return str(v.template)
        return str(v)


@macro(name="reasoning_agent", namespace="core")
class ReasoningAgentMacro(ConfigurableMacro):
    """Multi-step reasoning agent with tools as FunctionNodes.

    Architecture:
    ```
    Reasoning Chain:  step_0 → step_1 → step_2 → final

    Tool Nodes (parallel, on-demand):
      - tool_search
      - tool_calculate
      - tool_analyze
    ```

    Each step:
    - Receives previous step's output (internal memory)
    - Can invoke tools (via LLM instructing which tool to call)
    - Outputs to next step

    Tools are FunctionNodes in the graph, invoked by orchestrator when needed.
    """

    Config = ReasoningAgentConfig

    def expand(
        self,
        instance_name: str,
        inputs: dict[str, Any],
        dependencies: list[str],
    ) -> DirectedGraph:
        """Expand into reasoning chain + tool nodes.

        Graph structure:
        ```
        Reasoning: step_0 → step_1 → step_2 → final
        Tools:     tool_1, tool_2, ... (parallel, no deps)
        ```

        Tools are available for LLM to invoke during execution.
        """
        graph = DirectedGraph()
        config: ReasoningAgentConfig = self.config  # type: ignore[assignment]

        llm_factory = LLMNode()
        fn_factory = FunctionNode()

        # Add tool instruction to first step if tools available
        tool_instructions = ""
        if config.tools:
            tool_list = "\n".join(f"  - {tool}" for tool in config.tools)
            tool_instructions = f"""

Available tools:
{tool_list}

You can reference these tools in your reasoning."""

        # Build reasoning chain
        prev_step: str | None = None
        for step_idx in range(config.max_steps):
            step_name = f"{instance_name}_step_{step_idx}"
            step_deps: list[str] = dependencies if step_idx == 0 else [prev_step]  # type: ignore[list-item]

            # First step uses main_prompt with tool instructions
            if step_idx == 0:
                template = config.main_prompt + tool_instructions
            else:
                template = f"""Previous reasoning:
{{{{{prev_step}}}}}

Continue reasoning. Think deeply about the next step."""

            step_node = llm_factory(
                name=step_name,
                template=template,
                deps=step_deps,
            )
            graph += step_node
            prev_step = step_name

        # Final consolidation node
        assert prev_step is not None
        final_node = llm_factory(
            name=f"{instance_name}_final",
            template=f"""All reasoning steps:
{{{{{prev_step}}}}}

Provide final conclusion.""",
            deps=[prev_step],
        )
        graph += final_node

        # Add tool nodes as FunctionNodes (parallel, no dependencies)
        for tool_name in config.tools:
            tool_node = self._create_tool_node(fn_factory, instance_name, tool_name)
            graph += tool_node

        return graph

    def _create_tool_node(
        self, fn_factory: FunctionNode, instance_name: str, tool_name: str
    ) -> Any:
        """Create a FunctionNode that wraps a registered tool.

        Args:
        ----
            fn_factory: FunctionNode factory
            instance_name: Agent instance name
            tool_name: Name of the tool to wrap

        Returns:
        -------
            NodeSpec for the tool
        """
        from hexdag.core.registry import registry

        # Create wrapper that retrieves tool from registry
        async def tool_wrapper(input_data: Any) -> Any:
            """Execute registered tool."""
            # Get tool from registry
            tool_fn = registry.get(tool_name)

            if not tool_fn:
                logger.warning(f"Tool '{tool_name}' not found in registry")
                return {"error": f"Tool '{tool_name}' not found"}

            # Execute tool with input
            kwargs = input_data if isinstance(input_data, dict) else {"input": input_data}

            # Call tool (async or sync)
            if callable(tool_fn):
                import inspect

                if inspect.iscoroutinefunction(tool_fn):
                    result = await tool_fn(**kwargs)
                else:
                    result = tool_fn(**kwargs)
            else:
                result = {"error": f"Tool '{tool_name}' is not callable"}

            return result

        # Create FunctionNode for this tool
        return fn_factory(
            name=f"{instance_name}_tool_{tool_name}",
            fn=tool_wrapper,
            deps=[],  # Tools have no dependencies - invoked on-demand
        )
