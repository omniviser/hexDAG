"""ReasoningAgentMacro - Multi-step reasoning with adaptive tool calling.

Architecture:
- **Native Tool Calling**: Uses LLM adapter's aresponse_with_tools for OpenAI/Anthropic/Gemini
- **Text-Based Fallback**: INVOKE_TOOL: parsing for adapters without native support
- **Runtime Detection**: Adapts automatically based on adapter capabilities
- **Fallback Policy Support**: Seamlessly handles adapter switching during failures

This adaptive approach ensures:
- Optimal performance with native tool calling when available
- Compatibility with any LLM via text-based fallback
- Seamless integration with hexDAG's fallback policies
- Single graph works with multiple adapter types

Example workflow:
1. LLM reasoning step → tries native tools first, falls back to text if needed
2. Tool executor → executes parsed tool calls
3. Result merger → combines reasoning with tool results
4. Next step → continues with context from previous tools
"""

from typing import Any

from pydantic import field_validator

from hexdag.builtin.nodes.function_node import FunctionNode
from hexdag.builtin.nodes.prompt_node import PromptNode
from hexdag.builtin.nodes.raw_llm_node import RawLLMNode, RawLLMOutput
from hexdag.builtin.nodes.tool_utils import (
    ToolCallFormat,
    ToolDefinition,
    ToolParser,
    ToolSchemaConverter,
)
from hexdag.builtin.prompts.tool_prompts import get_tool_prompt_for_format
from hexdag.core.configurable import ConfigurableMacro, MacroConfig
from hexdag.core.domain.dag import DirectedGraph
from hexdag.core.logging import get_logger
from hexdag.core.orchestration.prompt import PromptTemplate
from hexdag.core.ports.llm import Message, MessageList
from hexdag.core.registry import macro, registry
from hexdag.core.registry.models import NAMESPACE_SEPARATOR, ComponentType

logger = get_logger(__name__)


class ReasoningAgentConfig(MacroConfig):
    """Configuration for ReasoningAgentMacro.

    Attributes
    ----------
    main_prompt : str
        Primary prompt for reasoning
    max_steps : int
        Maximum reasoning iterations (default: 3)
    allowed_tools : list[str]
        Tool names available to the agent (e.g., ["core:search", "core:calculate"])
        Uses qualified names (namespace:name)
    tool_format : ToolCallFormat
        Tool calling format for text-based fallback: FUNCTION_CALL, JSON, or MIXED (default: MIXED)
        Only used when adapter doesn't support native tool calling
    """

    main_prompt: str
    max_steps: int = 5
    allowed_tools: list[str] = []
    tool_format: ToolCallFormat = ToolCallFormat.MIXED

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
    """Multi-step reasoning agent with adaptive tool calling.

    Automatically detects and adapts to LLM adapter capabilities:
    - **Native mode**: OpenAI/Anthropic/Gemini → clean prompts, structured tool calls
    - **Text mode**: Local/other LLMs → INVOKE_TOOL: parsing from text
    - **Runtime adaptive**: Checks hasattr(llm, 'aresponse_with_tools') at execution time

    This adaptive approach enables:
    1. **Fallback Policy Support**: When a node fails and policy switches adapters,
       the same graph works with the new adapter type
    2. **Flexibility**: Single graph handles multiple adapter types
    3. **Optimal Performance**: Uses native tools when available

    Architecture per reasoning step:
    ```
    LLM Node (adaptive) → Tool Executor → Result Merger
    ```

    The LLM node adapts at runtime:
    - Tries native tool calling first (if available)
    - Falls back to text-based parsing automatically
    - Both paths prepared, chosen based on adapter capabilities
    """

    Config = ReasoningAgentConfig

    def expand(
        self,
        instance_name: str,
        inputs: dict[str, Any],
        dependencies: list[str],
    ) -> DirectedGraph:
        """Expand into reasoning chain with adaptive tool calling strategy."""
        graph = DirectedGraph()
        config: ReasoningAgentConfig = self.config  # type: ignore[assignment]

        fn_factory = FunctionNode()

        # Build tool schemas for native calling
        tool_schemas = self._build_tool_schemas_for_native(config.allowed_tools)

        # Build tool list for text-based fallback
        tool_list_text = self._build_tool_list_for_text(config.allowed_tools)
        tool_prompt = get_tool_prompt_for_format(config.tool_format)

        # Build reasoning chain
        prev_step: str | None = None
        for step_idx in range(config.max_steps):
            step_name_prefix = f"{instance_name}_step_{step_idx}"
            step_deps: list[str] = dependencies if step_idx == 0 else [prev_step]  # type: ignore[list-item]

            # Create LLM subgraph (Prompt + RawLLM + Adapter)
            llm_subgraph = self._create_llm_subgraph(
                fn_factory,
                step_name_prefix,
                step_idx,
                prev_step,
                config,
                tool_schemas,
                tool_list_text,
                tool_prompt,
                step_deps,
            )
            graph |= llm_subgraph

            # Create tool executor node
            tool_executor = self._create_tool_executor_node(fn_factory, step_name_prefix, config)
            graph += tool_executor

            # Create result merger node
            result_merger = self._create_result_merger_node(fn_factory, step_name_prefix)
            graph += result_merger

            prev_step = f"{step_name_prefix}_result_merger"

        # Final consolidation using composable nodes
        if prev_step is None:
            raise ValueError("prev_step is None")

        # Prompt for final consolidation
        prompt_factory = PromptNode()
        final_prompt = prompt_factory(
            name=f"{instance_name}_final_prompt",
            template=f"""All reasoning steps and tool results:
{{{{{prev_step}}}}}

Provide your final conclusion based on all reasoning and evidence gathered.""",
            output_format="messages",
            deps=[prev_step],
        )
        graph += final_prompt

        # RawLLM for final response
        llm_factory = RawLLMNode()
        final_llm = llm_factory(
            name=f"{instance_name}_final",
            deps=[f"{instance_name}_final_prompt"],
        )
        graph += final_llm

        return graph

    def _create_llm_subgraph(
        self,
        fn_factory: FunctionNode,
        step_name: str,
        step_idx: int,
        prev_step: str | None,
        config: ReasoningAgentConfig,
        tool_schemas: list[dict[str, Any]],
        tool_list_text: str,
        tool_prompt: Any,
        deps: list[str],
    ) -> DirectedGraph:
        """Create prompt + LLM nodes using composable architecture.

        Returns a subgraph with:
        - PromptNode: builds prompt with tool instructions
        - RawLLMNode: calls LLM with native tools (auto-detects capability)
        - FunctionNode: parses text-based tool calls if needed
        """
        subgraph = DirectedGraph()

        # Build prompt content
        if step_idx == 0:
            base_prompt = config.main_prompt
        else:
            base_prompt = f"""Previous reasoning and tool results:
{{{{{prev_step}}}}}

Continue reasoning. Use tools if needed to gather more information."""

        # Add tool instructions for text-based fallback
        full_prompt = f"""{base_prompt}

## Available Tools
{tool_list_text}

{tool_prompt.template if hasattr(tool_prompt, "template") else str(tool_prompt)}"""

        # Node 1: PromptNode builds the prompt
        prompt_factory = PromptNode()
        prompt_node = prompt_factory(
            name=f"{step_name}_prompt",
            template=full_prompt,
            output_format="messages",
            deps=deps,
        )
        subgraph += prompt_node

        # Node 2: RawLLMNode calls LLM (with native tools if available)
        llm_factory = RawLLMNode()
        llm_node = llm_factory(
            name=f"{step_name}_raw_llm",
            tools=tool_schemas if tool_schemas else None,
            tool_choice="auto",
            deps=[f"{step_name}_prompt"],
        )
        subgraph += llm_node

        async def normalize_response(input_data: RawLLMOutput, **kwargs: Any) -> dict[str, Any]:
            """Normalize LLM response to unified format."""
            # RawLLMNode returns RawLLMOutput:
            # RawLLMOutput(text=..., tool_calls=[...] | None, finish_reason=...)

            if input_data.tool_calls:
                # Native tool calling was used
                return {
                    "content": input_data.text,
                    "tool_calls": input_data.tool_calls,
                    "strategy": "native",
                }

            # Text-based - parse tool calls from text
            parsed_calls = ToolParser.parse_tool_calls(input_data.text, format=config.tool_format)

            return {
                "content": input_data.text,
                "tool_calls": [
                    {"id": f"text_{i}", "name": tc.name, "arguments": tc.params}
                    for i, tc in enumerate(parsed_calls)
                ],
                "strategy": "text",
            }

        adapter_node = fn_factory(
            name=f"{step_name}_llm",
            fn=normalize_response,
            deps=[f"{step_name}_raw_llm"],
        )
        subgraph += adapter_node

        return subgraph

    def _create_tool_executor_node(
        self, fn_factory: FunctionNode, step_name: str, config: ReasoningAgentConfig
    ) -> Any:
        """Create node that executes tool calls (from native or parsed)."""

        async def execute_tools(
            input_data: Any, tool_router: Any = None, **kwargs: Any
        ) -> dict[str, Any]:
            """Execute tool calls and return results."""
            llm_output = input_data
            tool_calls = llm_output.get("tool_calls", [])

            if not tool_calls:
                return {
                    "llm_content": llm_output.get("content", ""),
                    "tool_results": [],
                    "has_tools": False,
                }

            # Build tool name mapping (handle both qualified and unqualified names)
            # This provides backward compatibility with namespace:name format
            tool_name_map = {}
            for allowed_tool in config.allowed_tools:
                # Store the tool as-is
                tool_name_map[allowed_tool] = allowed_tool

                # If it has a namespace, also map the short name to it
                # This allows "search" to resolve to "demo:search" for backward compatibility
                if NAMESPACE_SEPARATOR in allowed_tool:
                    _, short_name = allowed_tool.split(NAMESPACE_SEPARATOR, 1)
                    # Only add short name mapping if it doesn't conflict
                    if short_name not in tool_name_map:
                        tool_name_map[short_name] = allowed_tool

            # Execute tools
            tool_results = []
            for tc in tool_calls:
                try:
                    # Resolve tool name (handle both "search" and "demo:search")
                    tool_name = tc["name"]
                    resolved_name = tool_name_map.get(tool_name, tool_name)

                    # Execute tool
                    if tool_router:
                        result = await tool_router.acall_tool(resolved_name, tc["arguments"])
                    else:
                        # Direct registry call
                        tool = registry.get(resolved_name)
                        if callable(tool):
                            import asyncio

                            if asyncio.iscoroutinefunction(tool):
                                result = await tool(**tc["arguments"])
                            else:
                                result = tool(**tc["arguments"])
                        else:
                            result = tool

                    tool_results.append({
                        "tool_name": tc["name"],
                        "arguments": tc["arguments"],
                        "result": result,
                    })
                except Exception as e:
                    logger.error(f"Tool execution error for {tc['name']}: {e}")
                    tool_results.append({
                        "tool_name": tc["name"],
                        "arguments": tc["arguments"],
                        "error": str(e),
                    })

            return {
                "llm_content": llm_output.get("content", ""),
                "tool_results": tool_results,
                "has_tools": True,
            }

        return fn_factory(
            name=f"{step_name}_tool_executor",
            fn=execute_tools,
            deps=[f"{step_name}_llm"],
        )

    def _create_result_merger_node(self, fn_factory: FunctionNode, step_name: str) -> Any:
        """Create node that merges LLM reasoning with tool results."""

        async def merge_results(input_data: Any, **kwargs: Any) -> str:
            """Combine LLM content with tool results into readable format."""
            executor_output = input_data
            llm_content: str = executor_output.get("llm_content", "")
            tool_results = executor_output.get("tool_results", [])
            has_tools = executor_output.get("has_tools", False)

            if not has_tools:
                return llm_content

            # Format tool results
            results_text = "\n\n## Tool Execution Results:\n"
            for tr in tool_results:
                if "error" in tr:
                    results_text += f"- {tr['tool_name']}: ERROR - {tr['error']}\n"
                else:
                    results_text += f"- {tr['tool_name']}: {tr['result']}\n"

            return f"{llm_content}{results_text}"

        return fn_factory(
            name=f"{step_name}_result_merger", fn=merge_results, deps=[f"{step_name}_tool_executor"]
        )

    def _create_final_consolidation_fn(self, prev_step: str) -> Any:
        """Create function for final consolidation of all reasoning."""

        async def consolidate(input_data: Any, llm: Any, **kwargs: Any) -> str:
            """Consolidate all reasoning steps into final answer."""
            all_reasoning = kwargs.get(prev_step, "")

            messages = MessageList([
                Message(
                    role="user",
                    content=f"""All reasoning steps and tool results:
{all_reasoning}

Provide your final conclusion based on all reasoning and evidence gathered.""",
                )
            ])

            response = await llm.aresponse(messages)
            return response or ""

        return consolidate

    def _build_tool_schemas_for_native(self, allowed_tools: list[str]) -> list[dict[str, Any]]:
        """Build OpenAI-format tool schemas for native calling."""
        schemas = []
        for tool_name in allowed_tools:
            try:
                # Get tool metadata
                metadata = registry.get_metadata(tool_name, component_type=ComponentType.TOOL)

                # Build ToolDefinition
                tool_def = ToolDefinition(
                    name=tool_name,
                    simplified_description=metadata.description or f"Tool {tool_name}",
                    detailed_description=metadata.description or f"Tool {tool_name}",
                    parameters=[],
                    examples=[],
                )

                # Convert to OpenAI format
                schema = ToolSchemaConverter.to_openai_schema(tool_def)
                schemas.append(schema)
            except Exception as e:
                logger.warning(f"Could not build schema for tool {tool_name}: {e}")

        return schemas

    def _build_tool_list_for_text(self, allowed_tools: list[str]) -> str:
        """Build text-format tool list for fallback mode."""
        if not allowed_tools:
            return "No tools available"

        tool_lines = []
        for tool_name in allowed_tools:
            try:
                metadata = registry.get_metadata(tool_name)
                description = metadata.description or "No description"
                tool_lines.append(f"  - {tool_name}: {description}")
            except Exception:
                tool_lines.append(f"  - {tool_name}: Tool description unavailable")

        return "\n".join(tool_lines)
