"""Built-in node factories for the hexai framework."""

import logging
from typing import Any

from hexai.core.application.nodes.tool_utils import ToolDefinition

from ..prompt import PromptTemplate
from .agent_node import AgentConfig, ReActAgentNode
from .function_node import FunctionNode
from .llm_node import LLMNode
from .loop_node import ConditionalNode, LoopNode
from .node_factory import NodeFactory

logger = logging.getLogger("hexai.app.application.nodes.builtin_nodes")

# Create singleton instances for node factories
_function_node = FunctionNode()
_llm_node = LLMNode()
# Use ReActAgentNode as the agent implementation
_agent_node = ReActAgentNode()
_loop_node = LoopNode()
_conditional_node = ConditionalNode()


def function_node_factory(
    node_id: str,
    fn: Any = None,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Create a function node using the FunctionNode factory."""
    logger.debug("🏭 Creating function node: %s", node_id)
    # Logic moved to FunctionNode - factory is now just a thin wrapper
    result = _function_node(
        name=node_id, fn=fn, input_schema=input_schema, output_schema=output_schema, **kwargs
    )
    logger.debug("✅ Function node created: %s", node_id)
    return result


def llm_node_factory(
    node_id: str,
    prompt_template: str | PromptTemplate = "Process: {{input}}",
    output_schema: dict[str, Any] | None = None,
    parse_as_json: bool = True,
    **kwargs: Any,
) -> Any:
    """Create an LLM node using the LLMNode factory.

    Note: LLM instance is now provided through ports at runtime.
    """
    logger.debug("🏭 Creating LLM node: %s", node_id)
    # Logic moved to LLMNode - factory is now just a thin wrapper
    result = _llm_node(
        name=node_id,
        template=prompt_template,
        output_schema=output_schema,
        response_format="json" if parse_as_json else ("structured" if output_schema else "text"),
        **kwargs,
    )
    logger.debug("✅ LLM node created: %s", node_id)
    return result


def agent_node_factory(
    node_id: str,
    initial_prompt_template: str | PromptTemplate | None = None,
    continuation_prompt_template: str | PromptTemplate | None = None,
    max_steps: int = 5,
    output_schema: dict[str, Any] | None = None,
    available_tools: list[str] | None = None,
    input_schema: dict[str, Any] | None = None,
    examples: list[dict[str, Any]] | None = None,
    example_separator: str = "\n\n",
    **kwargs: Any,
) -> Any:
    """Create a reasoning agent node using the ReActAgentNode implementation.

    This agent is built from basic nodes (LLM, Function, Loop) rather than
    being a monolithic implementation.

    Supports prompt templates as:
    - String: Converted to PromptTemplate
    - Dict with system_message/human_message: Converted to ChatPromptTemplate
    - PromptTemplate/ChatPromptTemplate: Used directly

    Note: LLM and ToolRouter instances are now provided through ports at runtime.
    """
    logger.debug("🏭 Creating agent node: %s", node_id)
    # Logic moved to ReActAgentNode - factory is now just a thin wrapper

    # Convert string tool names to ToolDefinition objects if needed
    tool_definitions = None
    if available_tools:
        tool_definitions = [
            ToolDefinition(
                name=tool_name,
                simplified_description=f"Tool: {tool_name}",
                detailed_description=f"Execute {tool_name} tool with provided parameters",
            )
            for tool_name in available_tools
        ]

    # Use default prompt if none provided
    main_prompt = (
        initial_prompt_template
        or "You are a helpful assistant. Analyze the given input and provide a thoughtful response."
    )

    # Convert continuation_prompt_template to dict format if provided
    continuation_prompts = None
    if continuation_prompt_template:
        if isinstance(continuation_prompt_template, dict):
            # Convert dict[str, str] to dict[str, PromptTemplate]
            continuation_prompts = {}
            for key, value in continuation_prompt_template.items():
                if isinstance(value, str):
                    continuation_prompts[key] = PromptTemplate(value)
                else:
                    continuation_prompts[key] = value
        else:
            continuation_prompts = {"continue": continuation_prompt_template}

    result = _agent_node(
        name=node_id,
        main_prompt=main_prompt,
        continuation_prompts=continuation_prompts,
        output_schema=output_schema,
        available_tools=tool_definitions,
        config=AgentConfig(max_steps=max_steps),
        **kwargs,
    )
    logger.debug("✅ Agent node created: %s", node_id)
    return result


def loop_node_factory(
    node_id: str,
    max_iterations: int = 3,
    success_condition: Any = None,
    iteration_key: str = "loop_iteration",
    **kwargs: Any,
) -> Any:
    """Create a loop control node using the LoopNode factory."""
    logger.debug("🏭 Creating loop node: %s", node_id)
    # Logic moved to LoopNode - factory is now just a thin wrapper
    result = _loop_node(
        name=node_id,
        max_iterations=max_iterations,
        success_condition=success_condition,
        iteration_key=iteration_key,
        **kwargs,
    )
    logger.debug("✅ Loop node created: %s", node_id)
    return result


def conditional_node_factory(
    node_id: str,
    condition_key: str = "should_continue",
    true_action: str = "continue",
    false_action: str = "proceed",
    **kwargs: Any,
) -> Any:
    """Create a conditional routing node using the ConditionalNode factory."""
    logger.debug("🏭 Creating conditional node: %s", node_id)
    # Logic moved to ConditionalNode - factory is now just a thin wrapper
    result = _conditional_node(
        name=node_id,
        condition_key=condition_key,
        true_action=true_action,
        false_action=false_action,
        **kwargs,
    )
    logger.debug("✅ Conditional node created: %s", node_id)
    return result


# Register builtin node types when module is imported
NodeFactory.register_node_type(
    "function",
    function_node_factory,
    "Creates a node from a Python function with optional input/output validation",
)

NodeFactory.register_node_type(
    "llm",
    llm_node_factory,
    "Creates a node that sends prompts to an LLM and parses responses",
)

NodeFactory.register_node_type(
    "agent",
    agent_node_factory,
    "Creates a reasoning agent that can use tools and make multi-step decisions",
)

NodeFactory.register_node_type(
    "loop",
    loop_node_factory,
    "Creates a loop control node for iterative execution with success criteria",
)

NodeFactory.register_node_type(
    "conditional",
    conditional_node_factory,
    "Creates a conditional routing node for branching logic based on input conditions",
)
