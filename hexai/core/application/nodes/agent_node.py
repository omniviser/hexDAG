"""ReActAgentNode - Multi-step reasoning agent."""

import ast
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict

from hexai.adapters.function_tool_router import FunctionBasedToolRouter
from hexai.core.application.prompt import PromptInput
from hexai.core.application.prompt.template import PromptTemplate
from hexai.core.domain.dag import NodeSpec
from hexai.core.ports.tool_router import ToolRouter
from hexai.core.registry import node
from hexai.core.registry.models import NodeSubtype

from .base_node_factory import BaseNodeFactory
from .llm_node import LLMNode
from .tool_utils import ToolCallFormat, ToolParser


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
    step: int = 0
    response: str = ""

    # Loop iteration tracking
    loop_iteration: int = 0

    model_config = ConfigDict(extra="allow")  # Allow additional fields from input mapping


@dataclass
class AgentConfig:
    """Agent configuration for multi-step reasoning."""

    max_steps: int = 20
    tool_call_style: ToolCallFormat = ToolCallFormat.MIXED


@node(name="agent_node", subtype=NodeSubtype.AGENT, namespace="core")
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

    def __init__(self) -> None:
        """Initialize with dependencies."""
        super().__init__()
        self.llm_node = LLMNode()
        self.tool_parser = ToolParser()

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

        # Create models
        input_model = self.create_pydantic_model(f"{name}Input", input_schema)
        if input_model is None:
            input_model = type(f"{name}Input", (BaseModel,), {"__annotations__": {"input": str}})
        output_model = self.create_pydantic_model(f"{name}Output", output_schema) or type(
            f"{name}Output", (BaseModel,), {"__annotations__": {"output": str}}
        )

        # Create the agent function with internal loop composition
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
        if isinstance(prompt, str):
            prompt = PromptTemplate(prompt)

        if hasattr(prompt, "input_vars"):
            user_vars = set(prompt.input_vars)
            return dict.fromkeys(user_vars, str) if user_vars else {"input": str}

        return {"input": str}

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

        async def single_step_executor(input_data: Any, **ports: Any) -> Any:
            """Execute single reasoning step - designed for internal loop orchestration."""
            # Initialize or update state from previous iteration
            state = self._initialize_or_update_state(input_data)

            # Execute single reasoning step
            updated_state = await self._execute_single_step(
                state, name, main_prompt, continuation_prompts, config, ports
            )

            # Check if we got a final output (tool_end was called)
            final_output = await self._check_for_final_output(
                updated_state, output_model, ports.get("event_manager")
            )
            if final_output is not None:
                return final_output

            # Return state as dict for loop compatibility
            return updated_state.model_dump()

        # Define success condition using loop concepts
        def success_condition(result: Any) -> bool:
            """Check if agent should stop iterating."""
            # Stop if we got the final structured output (not a dict)
            if not isinstance(result, dict):
                return True

            # Stop if we reached max steps
            step = result.get("step", 0)
            if step >= config.max_steps:
                return True

            # Stop if tool_end was detected
            response = result.get("response", "")
            return "tool_end" in response.lower()

        async def agent_with_internal_loop(input_data: Any, **ports: Any) -> Any:
            """Agent executor that uses loop concepts for iteration control."""
            # Start with initial input
            current_result = input_data

            # Run the loop until success condition is met or max iterations reached
            for _ in range(config.max_steps):
                # Execute single step
                step_result = await single_step_executor(current_result, **ports)

                # Check if we got final output (structured model)
                if not isinstance(step_result, dict):
                    return step_result

                # Check success condition
                if success_condition(step_result):
                    final_output = await self._check_for_final_output(
                        self._initialize_or_update_state(step_result),
                        output_model,
                        ports.get("event_manager"),
                    )
                    if final_output is not None:
                        return final_output
                    return step_result

                # Continue with next iteration
                current_result = step_result

            # If we reach here, return the last result
            return current_result

        return agent_with_internal_loop

    def _initialize_or_update_state(self, input_data: Any) -> AgentState:
        """Initialize new state or update existing state from loop iteration.

        Returns
        -------
        AgentState
            Initialized or updated agent state
        """
        # Case 1: Continuing from previous iteration (loop passes AgentState dict)
        if isinstance(input_data, dict) and "reasoning_steps" in input_data:
            return AgentState.model_validate(input_data)

        # Case 2: Fresh input (first iteration)
        # Handle both dict and Pydantic model inputs
        if isinstance(input_data, BaseModel):
            # Pydantic model - convert to dict
            raw_input = input_data.model_dump()
        elif isinstance(input_data, dict):
            # Already a dict
            raw_input = input_data
        else:
            # Fallback for other types
            raw_input = {"input": str(input_data)}

        # Create fresh AgentState
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

        # Convert to template if needed
        if isinstance(prompt, str):
            prompt = PromptTemplate(prompt)

        # Get tool instructions
        tool_instructions = self._build_tool_instructions(tool_router, config)

        # Use the template's enhance method
        return prompt + tool_instructions

    def _build_tool_instructions(self, tool_router: ToolRouter, config: AgentConfig) -> str:
        """Build tool usage instructions based on the configured format.

        Returns
        -------
        str
            Tool usage instructions text
        """
        tool_schemas = tool_router.get_all_tool_schemas()
        if not tool_schemas:
            return "\n## No tools available"

        # Build simple tool list
        tool_list = []
        for name, schema in tool_schemas.items():
            params = ", ".join(p["name"] for p in schema.get("parameters", []))
            tool_list.append(f"- {name}({params}): {schema.get('description', 'No description')}")

        tools_text = "\n".join(tool_list)

        # Generate format-specific usage guidelines
        usage_guidelines = self._get_format_specific_guidelines(config.tool_call_style)

        return f"""
## Available Tools
{tools_text}

## Usage Guidelines
{usage_guidelines}
"""

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
- For phase change: INVOKE_TOOL: change_phase(phase='new_phase')"""

        if format_style == ToolCallFormat.JSON:
            return (
                """- Call ONE tool at a time: INVOKE_TOOL: """
                """{"tool": "tool_name", "params": {"param": "value"}}\n"""
                """- For final answer and structured output: INVOKE_TOOL: """
                """{"tool": "tool_end", "params": {"field1": "value1", "field2": "value2"}}\n"""
                """- For phase change: INVOKE_TOOL: """
                """{"tool": "change_phase", "params": {"phase": "new_phase"}}"""
            )

        # ToolCallFormat.MIXED
        return """- Call ONE tool at a time using either format:
  - Function style: INVOKE_TOOL: tool_name(param='value')
  - JSON style: INVOKE_TOOL: {"tool": "tool_name", "params": {"param": "value"}}
- For final answer and structured output:
  - Function: INVOKE_TOOL: tool_end(field1='value1', field2='value2')
  - JSON: INVOKE_TOOL: {"tool": "tool_end", "params": {"field1": "value1", "field2": "value2"}}
- For phase change:
  - Function: INVOKE_TOOL: change_phase(phase='new_phase')
  - JSON: INVOKE_TOOL: {"tool": "change_phase", "params": {"phase": "new_phase"}}"""

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

        # Create LLM node for this step
        llm_node_spec = self.llm_node.from_template(node_name, template=prompt)

        # Execute LLM with the prepared input
        return await llm_node_spec.fn(llm_input, **ports)  # type: ignore[no-any-return]

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
        tool_router = ports.get("tool_router", FunctionBasedToolRouter())

        # Get current step info
        current_step = max(state.loop_iteration, state.step) + 1
        node_step_name = f"{name}_step_{current_step}"

        current_prompt = self._get_current_prompt(
            main_prompt, continuation_prompts, state.current_phase
        )

        # Enhance prompt with tools
        enhanced_prompt = self._enhance_prompt_with_tools(current_prompt, tool_router, config)

        # Get LLM response - convert state to dict for template rendering
        state_dict = state.model_dump()
        llm_input = {
            **state_dict,
            **state_dict.get("input_data", {}),
            "reasoning_so_far": "\n".join(state.reasoning_steps) or "Starting reasoning...",
        }

        # Get LLM response
        response = await self._get_llm_response(enhanced_prompt, llm_input, ports, node_step_name)

        # Process tools and phase changes
        await self._process_tools_and_phases(
            response, state, tool_router, continuation_prompts, config, event_manager
        )

        # Update state with this step's results
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

    async def _process_tools_and_phases(
        self,
        response: str,
        state: AgentState,
        tool_router: ToolRouter | None,
        continuation_prompts: dict[str, PromptInput],
        config: AgentConfig,
        event_manager: Any,
    ) -> None:
        """Process tool calls and phase changes."""
        if not tool_router:
            return

        # Parse tool calls
        tool_calls = self.tool_parser.parse_tool_calls(response, format=config.tool_call_style)

        for tool_call in tool_calls:
            try:
                # Execute tool
                result = await tool_router.acall_tool(tool_call.name, tool_call.params)

                # Store result
                state.tool_results.append(f"{tool_call.name}: {result}")
                state.tools_used.append(tool_call.name)

                # Handle special tools
                if tool_call.name in ["change_phase", "phase"] and isinstance(result, dict):
                    new_phase = result.get("new_phase")
                    if new_phase and new_phase in continuation_prompts:
                        state.current_phase = new_phase
                        state.phase_history.append(new_phase)

            except Exception as e:
                error_msg = f"{tool_call.name}: Error - {e}"
                state.tool_results.append(error_msg)

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
        except (json.JSONDecodeError, SyntaxError, ValueError):
            # Failed to parse - return None to skip this result
            pass
        except Exception as e:
            # Log unexpected errors but continue
            import logging

            logger = logging.getLogger(__name__)
            logger.debug("Unexpected error parsing tool_end result: %s", e)

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

                    # Create and return the final output
                    return output_model.model_validate(parsed_data)

                except Exception as e:
                    # Log validation errors but continue processing
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.debug("Failed to validate tool_end result: %e", e)
                    continue  # Skip this tool result and try the next one

        return None


# Backward compatibility alias
ReasoningAgentNode = ReActAgentNode
