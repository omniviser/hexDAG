"""ReActAgentNode - Multi-step reasoning agent."""

import ast
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict

from hexdag.builtin.adapters.unified_tool_router import UnifiedToolRouter
from hexdag.core.context import get_port, get_ports
from hexdag.core.domain.agent_tools import AgentToolRouter
from hexdag.core.domain.dag import NodeSpec
from hexdag.core.logging import get_logger
from hexdag.core.orchestration.prompt import PromptInput
from hexdag.core.orchestration.prompt.template import PromptTemplate
from hexdag.core.ports.tool_router import ToolRouter
from hexdag.core.protocols import to_dict
from hexdag.core.utils.node_timer import node_timer

from .base_node_factory import BaseNodeFactory
from .llm_node import LLMNode
from .tool_utils import ToolCallFormat, ToolParser

logger = get_logger(__name__)

if TYPE_CHECKING:
    from types import MappingProxyType


class PhaseContext(TypedDict):
    """Context structure for phase transitions in agents.

    Attributes
    ----------
    previous_phase : str, optional
        The phase the agent is transitioning from
    reason : str, optional
        Explanation for why the phase change is occurring
    carried_data : dict[str, Any], optional
        Data to carry forward from the previous phase
    target_output : str, optional
        Expected output format or goal for the new phase
    iteration : int, optional
        Current iteration number if in a loop or retry scenario
    metadata : dict[str, Any], optional
        Additional metadata about the phase transition
    """

    previous_phase: NotRequired[str]
    reason: NotRequired[str]
    carried_data: NotRequired[dict[str, Any]]
    target_output: NotRequired[str]
    iteration: NotRequired[int]
    metadata: NotRequired[dict[str, Any]]


class AgentState(BaseModel):
    """Pydantic model for agent state - provides type safety and validation."""

    # Original input data (preserved)
    input_data: dict[str, Any] = {}

    # Agent reasoning state
    reasoning_steps: list[str] = []
    tool_results: list[str] = []
    tools_used: list[str] = []
    current_phase: str = "main"
    phase_history: list[str] = ["main"]
    phase_contexts: dict[str, PhaseContext] = {}  # Store typed context for each phase
    step: int = 0
    response: str = ""

    # Loop iteration tracking
    loop_iteration: int = 0

    model_config = ConfigDict(extra="allow")  # Allow additional fields from input mapping


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Agent configuration for multi-step reasoning (legacy - kept for backward compatibility)."""

    max_steps: int = 20
    tool_call_style: ToolCallFormat = ToolCallFormat.MIXED


class Agent:
    """Configuration for Agent Node.

    Attributes
    ----------
    max_steps : int
        Maximum number of reasoning steps (default: 20)
    tool_call_style : ToolCallFormat
        Format for tool calls - MIXED, FUNCTION_CALL, or JSON (default: MIXED)
    """

    max_steps: int = 20
    tool_call_style: ToolCallFormat = ToolCallFormat.MIXED


class ReActAgentNode(BaseNodeFactory):
    """Multi-step reasoning agent.

    This agent:
    1. Uses loop control internally for iteration control
    2. Implements single-step reasoning logic
    3. Maintains clean agent interface for users
    4. Leverages proven loop control patterns
    5. Supports all agent features (tools, phases, events)

    Architecture:
    ```
    Agent(input) -> Loop -> SingleStep -> Loop -> SingleStep -> ... -> Output
    ```
    """

    __aliases__ = ("agent_node",)

    # Studio UI metadata
    _hexdag_icon = "Bot"
    _hexdag_color = "#ec4899"  # pink-500

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with dependencies."""
        self.llm_node = LLMNode()
        self.tool_parser = ToolParser()
        self._tool_instructions_cache: str | None = None

    def __call__(
        self,
        name: str,
        main_prompt: PromptInput,
        continuation_prompts: dict[str, PromptInput] | None = None,
        output_schema: dict[str, type] | type[BaseModel] | None = None,
        config: AgentConfig | None = None,
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a multi-step reasoning agent with internal loop control.

        Args
        ----
            name: Agent name
            main_prompt: Initial reasoning prompt
            continuation_prompts: Phase-specific prompts
            output_schema: Custom output schema for tool_end results
            config: Agent configuration
            deps: Dependencies
            **kwargs: Additional parameters

        Returns
        -------
        NodeSpec
            A configured node specification for the agent
        """
        config = config or AgentConfig()

        # Infer input schema from prompt
        input_schema = self._infer_input_schema(main_prompt)

        input_model = self.create_pydantic_model(f"{name}Input", input_schema)
        if input_model is None:
            input_model = type(f"{name}Input", (BaseModel,), {"__annotations__": {"input": str}})
        output_model = self.create_pydantic_model(f"{name}Output", output_schema) or type(
            f"{name}Output", (BaseModel,), {"__annotations__": {"output": str}}
        )

        agent_fn = self._create_agent_with_loop(
            name, main_prompt, continuation_prompts or {}, output_model, config
        )

        # Use universal input mapping method
        return self.create_node_with_mapping(
            name=name,
            wrapped_fn=agent_fn,
            input_schema=input_schema,
            output_schema=output_model,
            deps=deps,
            **kwargs,
        )

    def _infer_input_schema(self, prompt: PromptInput) -> dict[str, Any]:
        """Infer input schema from prompt template.

        Returns
        -------
        dict[str, Any]
            Inferred input schema mapping
        """
        # Use the shared implementation from BaseNodeFactory
        # AgentNode doesn't filter special params, so pass None
        return BaseNodeFactory.infer_input_schema_from_template(prompt, special_params=None)

    def _get_current_prompt(
        self,
        main_prompt: PromptInput,
        continuation_prompts: dict[str, PromptInput],
        current_phase: str,
    ) -> PromptInput:
        """Get the appropriate prompt for the current phase.

        Returns
        -------
        PromptInput
            The prompt to use for the current phase
        """
        if current_phase != "main" and current_phase in continuation_prompts:
            return continuation_prompts[current_phase]
        return main_prompt

    def _create_agent_with_loop(
        self,
        name: str,
        main_prompt: PromptInput,
        continuation_prompts: dict[str, PromptInput],
        output_model: type[BaseModel],
        config: AgentConfig,
    ) -> Callable[..., Any]:
        """Create agent function with internal loop composition for multi-step iteration.

        Returns
        -------
        Callable[..., Any]
            Agent function with internal loop control
        """

        # Clear tool instructions cache for each new agent run
        self._tool_instructions_cache = None

        async def single_step_executor(input_data: Any) -> Any:
            """Execute single reasoning step."""
            from hexdag.core.context import get_port

            ports: MappingProxyType[str, Any] | dict[Any, Any] = get_ports() or {}

            state = self._initialize_or_update_state(input_data)

            # Execute single reasoning step
            updated_state = await self._execute_single_step(
                state, name, main_prompt, continuation_prompts, config, dict(ports)
            )

            final_output = await self._check_for_final_output(
                updated_state, output_model, get_port("event_manager")
            )
            if final_output is not None:
                return final_output

            # Return AgentState directly (Pydantic-first design)
            return updated_state

        # Define success condition using loop concepts
        def success_condition(result: Any) -> bool:
            """Check if agent should stop iterating."""
            # Stop if we got the final structured output (not AgentState)
            if not isinstance(result, AgentState):
                return True

            # Stop if we reached max steps
            if result.step >= config.max_steps:
                return True

            # Stop if tool_end was detected
            return "tool_end" in result.response.lower()

        async def agent_with_internal_loop(input_data: Any) -> Any:
            """Agent executor with internal loop control."""
            node_logger = logger.bind(node=name, node_type="agent_node")

            # Log agent start
            node_logger.info(
                "Starting agent",
                max_steps=config.max_steps,
                tool_call_style=config.tool_call_style.value,
            )

            # Start with initial input
            current_result = input_data

            with node_timer() as t:
                # Run the loop until success condition is met or max iterations reached
                for step_num in range(config.max_steps):
                    # Log step start
                    node_logger.debug(
                        "Agent step starting",
                        step=step_num + 1,
                        max_steps=config.max_steps,
                    )

                    # Execute single step
                    step_result = await single_step_executor(current_result)

                    # If not AgentState, it's the final output
                    if not isinstance(step_result, AgentState):
                        node_logger.info(
                            "Agent completed with direct output",
                            total_steps=step_num + 1,
                            duration_ms=t.duration_str,
                            output_type=type(step_result).__name__,
                        )
                        return step_result

                    # Log step completion with state info
                    node_logger.debug(
                        "Agent step completed",
                        step=step_num + 1,
                        phase=step_result.current_phase,
                        tools_used_count=len(step_result.tools_used),
                    )

                    # Check success condition
                    if success_condition(step_result):
                        final_output = await self._check_for_final_output(
                            step_result,
                            output_model,
                            get_port("event_manager"),
                        )
                        if final_output is not None:
                            node_logger.info(
                                "Agent completed",
                                total_steps=step_num + 1,
                                tools_used=step_result.tools_used,
                                phases=step_result.phase_history,
                                duration_ms=t.duration_str,
                            )
                            return final_output
                        return step_result

                    # Continue with next iteration (pass AgentState directly)
                    current_result = step_result

                # If we reach here, max steps reached
                node_logger.warning(
                    "Agent reached max steps",
                    max_steps=config.max_steps,
                    duration_ms=t.duration_str,
                )
                return current_result

        return agent_with_internal_loop

    def _initialize_or_update_state(self, input_data: Any) -> AgentState:
        """Initialize new state or update existing state from loop iteration.

        Returns
        -------
        AgentState
            Initialized or updated agent state
        """
        # Case 1: Already AgentState (from previous iteration) - return as-is
        if isinstance(input_data, AgentState):
            return input_data

        # Case 2: Dict with AgentState fields (legacy/backward compatibility)
        if isinstance(input_data, dict) and "reasoning_steps" in input_data:
            state: AgentState = AgentState.model_validate(input_data)
            return state

        # Case 3: Fresh input (first iteration) - wrap in AgentState
        try:
            raw_input = to_dict(input_data)
        except TypeError:
            # Fallback for non-dict types
            raw_input = {"input": str(input_data)}

        return AgentState(input_data=raw_input)

    def _enhance_prompt_with_tools(
        self, prompt: PromptInput, tool_router: ToolRouter | None, config: AgentConfig
    ) -> PromptInput:
        """Add tool instructions to the prompt.

        Returns
        -------
        PromptInput
            Enhanced prompt with tool instructions
        """
        if not tool_router:
            return prompt

        if isinstance(prompt, str):
            prompt = PromptTemplate(prompt)

        tool_instructions = self._build_tool_instructions(tool_router, config)

        # Use the template's enhance method
        return prompt + tool_instructions

    def _build_tool_instructions(self, tool_router: ToolRouter, config: AgentConfig) -> str:
        """Build tool usage instructions based on the configured format.

        Cached across agent steps since tool schemas don't change during execution.

        Returns
        -------
        str
            Tool usage instructions text
        """
        if self._tool_instructions_cache is not None:
            return self._tool_instructions_cache

        tool_schemas = tool_router.get_all_tool_schemas()
        if not tool_schemas:
            return "\n## No tools available"

        tool_list = []
        for name, schema in tool_schemas.items():
            params = ", ".join(p["name"] for p in schema.get("parameters", []))
            tool_list.append(f"- {name}({params}): {schema.get('description', 'No description')}")

        tools_text = "\n".join(tool_list)

        # Generate format-specific usage guidelines
        usage_guidelines = self._get_format_specific_guidelines(config.tool_call_style)

        result = f"""
## Available Tools
{tools_text}

## Usage Guidelines
{usage_guidelines}
"""
        self._tool_instructions_cache = result
        return result

    def _get_format_specific_guidelines(self, format_style: ToolCallFormat) -> str:
        """Generate format-specific tool calling guidelines.

        Returns
        -------
        str
            Format-specific guidelines text
        """
        if format_style == ToolCallFormat.FUNCTION_CALL:
            return """- Call ONE tool at a time: INVOKE_TOOL: tool_name(param='value')
- For final answer and structured output: INVOKE_TOOL: tool_end(field1='value1', field2='value2')
- For phase change: INVOKE_TOOL: change_phase(phase='new_phase', reason='why changing',
carried_data={'key': 'value'})"""

        if format_style == ToolCallFormat.JSON:
            return (
                """- Call ONE tool at a time: INVOKE_TOOL: """
                """{"tool": "tool_name", "params": {"param": "value"}}\n"""
                """- For final answer and structured output: INVOKE_TOOL: """
                """{"tool": "tool_end", "params": {"field1": "value1", "field2": "value2"}}\n"""
                """- For phase change: INVOKE_TOOL: """
                """{"tool": "change_phase", "params": {"phase": "new_phase", "reason": "why",
                "carried_data": {"key": "val"}}}"""
            )

        # ToolCallFormat.MIXED
        return """- Call ONE tool at a time using either format:
  - Function style: INVOKE_TOOL: tool_name(param='value')
  - JSON style: INVOKE_TOOL: {"tool": "tool_name", "params": {"param": "value"}}
- For final answer and structured output:
  - Function: INVOKE_TOOL: tool_end(field1='value1', field2='value2')
  - JSON: INVOKE_TOOL: {"tool": "tool_end", "params": {"field1": "value1", "field2": "value2"}}
- For phase change:
  - Function: INVOKE_TOOL: change_phase(phase='new_phase', reason='why',
  carried_data={'key': 'val'})
  - JSON: INVOKE_TOOL: {"tool": "change_phase", "params": {"phase": "new_phase",
  "reason": "why", "carried_data": {"key": "val"}}}"""

    async def _get_llm_response(
        self, prompt: PromptInput, llm_input: dict[str, Any], ports: dict[str, Any], node_name: str
    ) -> str:
        """Get response from LLM.

        Returns
        -------
        str
            LLM response text
        """
        # Ensure we have a proper template (not string)
        if isinstance(prompt, str):
            prompt = PromptTemplate(prompt)

        llm_node_spec = self.llm_node.from_template(node_name, template=prompt)

        # Execute LLM with the prepared input (no ports passed - uses ExecutionContext)
        return await llm_node_spec.fn(llm_input)  # type: ignore[no-any-return]

    async def _execute_single_step(
        self,
        state: AgentState,
        name: str,
        main_prompt: PromptInput,
        continuation_prompts: dict[str, PromptInput],
        config: AgentConfig,
        ports: dict[str, Any],
    ) -> AgentState:
        """Execute a single reasoning step.

        Returns
        -------
        AgentState
            Updated agent state after step execution
        """
        event_manager = ports.get("event_manager")
        base_router = ports.get("tool_router", UnifiedToolRouter())

        # Create phase change callback that mutates state
        def _handle_phase_change(result: dict[str, Any]) -> None:
            new_phase = result.get("new_phase")
            context = result.get("context", {})
            if new_phase and new_phase in continuation_prompts:
                old_phase = state.current_phase
                if "previous_phase" not in context:
                    context["previous_phase"] = state.current_phase
                state.phase_contexts[new_phase] = context
                state.current_phase = new_phase
                state.phase_history.append(new_phase)
                logger.info(
                    "Phase transition",
                    from_phase=old_phase,
                    to_phase=new_phase,
                    reason=context.get("reason", ""),
                )
                if "carried_data" in context and isinstance(context["carried_data"], dict):
                    state.input_data.update(context["carried_data"])

        # Register agent lifecycle tools alongside user tools
        agent_tool_router = AgentToolRouter(on_phase_change=_handle_phase_change)
        tool_router = UnifiedToolRouter(
            routers={
                "agent": agent_tool_router,
                **(base_router.routers if isinstance(base_router, UnifiedToolRouter) else {}),
            }
        )
        # For unprefixed tool calls, try user routers first, agent tools as fallback
        if isinstance(base_router, UnifiedToolRouter) and base_router.default_router:
            tool_router.default_router = base_router.default_router
        else:
            tool_router.default_router = agent_tool_router

        current_step = max(state.loop_iteration, state.step) + 1
        node_step_name = f"{name}_step_{current_step}"

        current_prompt = self._get_current_prompt(
            main_prompt, continuation_prompts, state.current_phase
        )

        # Enhance prompt with tools
        enhanced_prompt = self._enhance_prompt_with_tools(current_prompt, tool_router, config)

        current_phase_context = state.phase_contexts.get(state.current_phase, {})

        # Build LLM input - only convert to dict when needed for template
        # Merge state fields with template-specific overrides
        llm_input = {
            **state.model_dump(),  # Convert only once, at template boundary
            **state.input_data,
            "reasoning_so_far": "\n".join(state.reasoning_steps) or "Starting reasoning...",
            "phase_context": current_phase_context,
            "phase_reason": current_phase_context.get("reason", ""),
            "phase_target": current_phase_context.get("target_output", ""),
        }

        response = await self._get_llm_response(enhanced_prompt, llm_input, ports, node_step_name)

        # Process tool calls — phase changes are handled by the callback
        await self._process_tools(response, state, tool_router, config, event_manager)

        state.reasoning_steps.append(f"Step {current_step}: {response}")
        state.response = response
        state.step = current_step

        return state

    def _should_terminate(self, response: str) -> bool:
        """Check if agent should terminate.

        Returns
        -------
        bool
            True if agent should terminate execution
        """
        return "tool_end" in response.lower() or "Tool_END" in response

    async def _process_tools(
        self,
        response: str,
        state: AgentState,
        tool_router: ToolRouter | None,
        config: AgentConfig,
        event_manager: Any,
    ) -> None:
        """Process tool calls from LLM response.

        Phase changes are handled automatically by the ``AgentToolRouter``
        callback registered in ``_execute_single_step``.
        """
        if not tool_router:
            return

        # Parse tool calls
        tool_calls = self.tool_parser.parse_tool_calls(response, format=config.tool_call_style)

        if tool_calls:
            logger.debug(
                "Parsed tool calls",
                tool_count=len(tool_calls),
                tools=[tc.name for tc in tool_calls],
            )

        for tool_call in tool_calls:
            try:
                logger.debug(
                    "Executing tool",
                    tool_name=tool_call.name,
                    params_preview=str(tool_call.params)[:100],
                )

                # Route through unified interface — agent lifecycle tools
                # (tool_end, change_phase) are handled by AgentToolRouter,
                # user tools by their respective routers.
                result = await tool_router.acall_tool(tool_call.name, tool_call.params)

                state.tool_results.append(f"{tool_call.name}: {result}")
                state.tools_used.append(tool_call.name)

            except Exception as e:
                error_msg = f"{tool_call.name}: Error - {e}"
                state.tool_results.append(error_msg)
                logger.warning(
                    "Tool execution failed",
                    tool_name=tool_call.name,
                    error=str(e),
                )

    def _parse_tool_end_result(self, tool_result: str) -> dict[str, Any] | None:
        """Parse tool_end result string into structured data.

        Args
        ----
        tool_result : str
            Tool result string in format "tool_end: {data}"

        Returns
        -------
        dict[str, Any] | None
            Parsed data dictionary or None if parsing fails
        """
        if not tool_result or not tool_result.startswith("tool_end:"):
            return None

        try:
            result_str = tool_result.split(":", 1)[1].strip()
            result_data = ast.literal_eval(result_str)

            if isinstance(result_data, dict):
                return result_data
            return None
        except (json.JSONDecodeError, SyntaxError, ValueError, IndexError):
            # Failed to parse - return None to skip this result
            # IndexError: split failed (malformed tool_end output)
            return None

    async def _emit_agent_metadata(self, state: AgentState, event_manager: Any) -> None:
        """Emit agent metadata trace event.

        Args
        ----
        state : AgentState
            Current agent state
        event_manager : Any
            Event manager instance
        """
        if event_manager and hasattr(event_manager, "add_trace"):
            await event_manager.add_trace(
                "agent_metadata",
                {
                    "reasoning_steps": state.reasoning_steps,
                    "tools_used": list(set(state.tools_used)),
                    "reasoning_phases": state.phase_history,
                    "total_steps": state.step,
                },
            )

    async def _check_for_final_output(
        self,
        state: AgentState,
        output_model: type[BaseModel],
        event_manager: Any,
    ) -> Any | None:
        """Check if we have a final output from tool_end calls.

        Returns
        -------
        Any | None
            Final output model instance or None if not found
        """
        # Check for tool_end calls with structured output
        for tool_result in reversed(state.tool_results):
            parsed_data = self._parse_tool_end_result(tool_result)

            if parsed_data is not None:
                try:
                    # Emit metadata before returning final result
                    await self._emit_agent_metadata(state, event_manager)

                    return output_model.model_validate(parsed_data)

                except (ValueError, TypeError) as e:
                    # Validation failed - try next tool_end result
                    logger.debug(
                        "Failed to validate tool_end result",
                        output_model=output_model.__name__,
                        error=str(e),
                    )
                    continue  # Skip this tool result and try the next one

        return None


# Backward compatibility alias
ReasoningAgentNode = ReActAgentNode
