"""RawLLMNode - Pure LLM API call without prompting or parsing.

This is the new atomic LLM node that does ONLY API calls.
Use PromptNode before this and ParserNode after for full functionality.
Or use the LLM macro for automatic composition with retry logic.
"""

from typing import Any

from pydantic import BaseModel

from hexdag.core.context import get_port
from hexdag.core.domain.dag import NodeSpec
from hexdag.core.logging import get_logger
from hexdag.core.ports.llm import Message

from .base_node_factory import BaseNodeFactory

logger = get_logger(__name__)


class RawLLMInput(BaseModel):
    """Input model for RawLLMNode."""

    messages: list[dict[str, str]] | None = None
    """Messages in OpenAI format: [{"role": "user", "content": "..."}]"""

    text: str | None = None
    """Plain text prompt (will be converted to user message)"""

    model_config = {"extra": "allow"}


class RawLLMOutput(BaseModel):
    """Output model for RawLLMNode."""

    text: str
    """Raw LLM response text"""

    tool_calls: list[dict[str, Any]] | None = None
    """Tool calls from native function calling (if supported)"""

    finish_reason: str | None = None
    """Finish reason from LLM response"""


class RawLLMNode(BaseNodeFactory):
    """Pure LLM API call node - no prompting, no parsing, just raw API interaction.

    This node is the atomic building block for LLM-based workflows.
    It ONLY calls the LLM port and returns the raw response.

    For full functionality, compose with PromptNode and ParserNode:
        PromptNode → RawLLMNode → ParserNode

    Or use the LLM macro for automatic composition with retry logic.

    Architecture
    ------------
    Old (monolithic)::

        LLMNode: [Prompt Building + API Call + Parsing]

    New (composable)::

        PromptNode (build) → RawLLMNode (call) → ParserNode (parse)

    Examples
    --------
    Direct usage with messages::

        raw_llm = RawLLMNode()
        spec = raw_llm(name="llm_call")

        # Execute with messages
        result = await spec.fn({
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "What is 2+2?"}
            ]
        })
        # Returns: {"text": "2+2 equals 4"}

    Direct usage with plain text::

        result = await spec.fn({"text": "What is 2+2?"})
        # Automatically converted to user message

    Composed with PromptNode::

        from hexdag.core.domain.dag import DirectedGraph
        from hexdag.builtin.nodes import PromptNode, RawLLMNode

        graph = DirectedGraph()

        prompt_spec = PromptNode()(
            name="prompt",
            template="Answer: {{question}}",
            output_format="messages"
        )

        llm_spec = RawLLMNode()(name="llm")

        graph.add_node(prompt_spec, depends_on=[])
        graph.add_node(llm_spec, depends_on=["prompt"])

        # Execute: prompt builds messages, llm calls API
        result = await orchestrator.aexecute(graph, {"question": "What is AI?"})
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize RawLLMNode."""
        super().__init__()

    def __call__(
        self,
        name: str,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a raw LLM API call node.

        Args
        ----
            name: Node name
            tools: Optional tools for native function calling
            tool_choice: Tool choice strategy ("auto", "none", or specific tool)
            deps: Dependencies
            **kwargs: Additional parameters

        Returns
        -------
        NodeSpec
            Configured node specification for raw LLM calls
        """
        # Create the LLM calling function
        llm_fn = self._create_llm_caller(
            node_name=name,
            tools=tools,
            tool_choice=tool_choice,
        )

        # Input accepts either messages or text
        input_schema = {"messages": list[dict[str, str]] | None, "text": str | None}

        # Output is always raw text
        output_model = RawLLMOutput

        # Use universal input mapping method
        return self.create_node_with_mapping(
            name=name,
            wrapped_fn=llm_fn,
            input_schema=input_schema,
            output_schema=output_model,
            deps=deps,
            **kwargs,
        )

    def _create_llm_caller(
        self,
        node_name: str,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> Any:
        """Create the LLM calling function.

        Args
        ----
            node_name: Name of the node (for logging)
            tools: Optional tools for native function calling
            tool_choice: Tool choice strategy

        Returns
        -------
        Callable
            Async function that calls LLM
        """

        async def call_llm(input_data: Any) -> dict[str, Any]:
            """Call LLM with messages or text, optionally with tools."""
            # Extract input
            if isinstance(input_data, dict):
                messages = input_data.get("messages")
                text = input_data.get("text")
            elif isinstance(input_data, BaseModel):
                messages = getattr(input_data, "messages", None)
                text = getattr(input_data, "text", None)
            else:
                # Fallback: treat as text
                messages = None
                text = str(input_data)

            # Convert text to messages if needed
            if messages is None and text:
                messages = [{"role": "user", "content": text}]

            if not messages:
                raise ValueError("RawLLMNode requires either 'messages' or 'text' input")

            # Get LLM port from execution context
            llm_port = get_port("llm")
            if not llm_port:
                raise RuntimeError("LLM port not available in execution context")

            # Try native tool calling if tools are provided and adapter actually implements it
            has_tool_support = False
            if tools and hasattr(llm_port, "aresponse_with_tools"):
                for cls in llm_port.__class__.__mro__:
                    if (
                        cls.__name__ not in ["LLM", "Protocol", "object"]
                        and "aresponse_with_tools" in cls.__dict__
                    ):
                        has_tool_support = True
                        break

            if tools and has_tool_support:
                logger.debug(f"Using native tool calling with {len(tools)} tools")

                # Convert dicts to Message objects for port wrapper compatibility
                message_objects = [Message(**msg) for msg in messages]

                # Call with tools
                response = await llm_port.aresponse_with_tools(
                    messages=message_objects,
                    tools=tools,
                    tool_choice=tool_choice,
                )

                # Return structured response
                return {
                    "text": response.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                        for tc in (response.tool_calls or [])
                    ]
                    if response.tool_calls
                    else None,
                    "finish_reason": response.finish_reason,
                }

            # Fallback to standard aresponse
            logger.debug(f"Using standard LLM call with {len(messages)} messages")

            # Convert dicts to Message objects for port wrapper compatibility
            message_objects = [Message(**msg) for msg in messages]

            # Call LLM with messages as positional argument
            response = await llm_port.aresponse(message_objects)

            # Return raw text response
            return {"text": response or "", "tool_calls": None, "finish_reason": None}

        return call_llm
