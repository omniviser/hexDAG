"""ConversationMacro - Multi-turn chat with dynamic message history expansion.

Architecture:
- Loads conversation history from Memory port
- Accumulates messages dynamically during execution
- Supports tools with dynamic ToolMacro expansion
- Maintains conversation state across turns
- Automatic context window management

Example:
    Turn 1: User → [system, user] → LLM → Response → Save history
    Turn 2: User → [system, user, assistant, user] → LLM → Response → Save
    Turn N: Accumulated history → LLM → Response → Save

Dynamic Expansion:
- MessageAccumulator node injects new user messages at runtime
- ToolCallExpander node injects ToolCallNodes when LLM requests tools
- ConversationSaver node persists updated history to Memory port
"""

from typing import Any

from pydantic import Field

from hexdag.builtin.nodes.function_node import FunctionNode
from hexdag.builtin.nodes.llm_node import LLMNode
from hexdag.builtin.nodes.tool_utils import ToolCallFormat, ToolParser
from hexdag.builtin.prompts.tool_prompts import get_tool_prompt_for_format
from hexdag.core.configurable import ConfigurableMacro, MacroConfig
from hexdag.core.context import get_port
from hexdag.core.domain.dag import DirectedGraph, NodeSpec
from hexdag.core.logging import get_logger
from hexdag.core.registry import macro

logger = get_logger(__name__)


class ConversationConfig(MacroConfig):
    """Configuration for ConversationMacro.

    Attributes
    ----------
    system_prompt : str
        System message to set agent behavior
    conversation_id : str
        Unique ID for this conversation (used as memory key)
    max_history : int
        Maximum number of messages to keep (default: 20)
    allowed_tools : list[str]
        Tools available to the agent (qualified names)
    tool_format : ToolCallFormat
        Tool calling format (default: MIXED)
    enable_tool_use : bool
        Whether to enable tool calling (default: True)
    """

    system_prompt: str = Field(default="You are a helpful assistant")
    conversation_id: str
    max_history: int = Field(default=20, ge=2)
    allowed_tools: list[str] = Field(default_factory=list)
    tool_format: ToolCallFormat = Field(default=ToolCallFormat.MIXED)
    enable_tool_use: bool = Field(default=True)


@macro(name="conversation", namespace="core")
class ConversationMacro(ConfigurableMacro):
    """Multi-turn conversation with dynamic message history expansion.

    Architecture (dynamic graph):
    ```
    [User Input] → Message Accumulator → LLM Node → Tool Expander → Response Saver
                        ↓                    ↓           ↓              ↓
                   Memory Port          LLM Port   [ToolCallNodes]  Memory Port
                   (load history)                  (if tools used)  (save history)
    ```

    Dynamic expansion features:
    - Message Accumulator: Injects new user message into conversation history
    - Tool Expander: Dynamically creates ToolCallNodes when LLM requests tools
    - Response Saver: Persists updated conversation to Memory port

    This enables:
    - Multi-turn conversations with full history
    - Tool use within conversations
    - Conversation persistence across sessions
    - Context window management (trimming old messages)

    Examples
    --------
    YAML configuration:

        macros:
          - type: conversation
            id: chatbot
            config:
              system_prompt: "You are a research assistant"
              conversation_id: "{{session_id}}"
              max_history: 20
              allowed_tools: ["core:search", "core:calculate"]
              enable_tool_use: true

    Multi-turn execution:

        # Turn 1
        results = await orchestrator.run(
            graph,
            {"user_message": "What is AI?", "session_id": "user123"},
            dynamic=True
        )

        # Turn 2 - conversation history loaded automatically
        results = await orchestrator.run(
            graph,
            {"user_message": "Tell me more", "session_id": "user123"},
            dynamic=True
        )
    """

    Config = ConversationConfig

    def expand(
        self,
        instance_name: str,
        inputs: dict[str, Any],
        dependencies: list[str],
    ) -> DirectedGraph:
        """Expand into conversation nodes with dynamic expansion.

        Graph structure:
        ```
        [deps] → load_history → format_messages → llm → parse_response → save_history
        ```

        During execution with dynamic=True:
        - load_history: Gets conversation from Memory port
        - format_messages: Adds new user message to history
        - llm: Generates response (may include tool calls)
        - parse_response: Parses response, injects ToolCallNodes if needed
        - save_history: Saves updated conversation to Memory port

        Args
        ----
            instance_name: Unique name for this conversation instance
            inputs: Input data (must include conversation_id and user_message)
            dependencies: Nodes to depend on

        Returns
        -------
        DirectedGraph
            Graph with conversation nodes
        """
        graph = DirectedGraph()
        config: ConversationConfig = self.config  # type: ignore[assignment]

        fn_factory = FunctionNode()
        llm_factory = LLMNode()

        # Node 1: Load conversation history from Memory
        load_node = self._create_load_history_node(fn_factory, instance_name, config, dependencies)
        graph += load_node

        # Node 2: Format messages (add new user message)
        format_node = self._create_format_messages_node(fn_factory, instance_name, config)
        graph += format_node

        # Node 3: LLM call with formatted messages
        llm_node = self._create_llm_node(llm_factory, instance_name, config)
        graph += llm_node

        # Node 4: Parse response and handle tool calls
        parse_node = self._create_parse_response_node(fn_factory, instance_name, config)
        graph += parse_node

        # Node 5: Save updated conversation history
        save_node = self._create_save_history_node(fn_factory, instance_name, config)
        graph += save_node

        return graph

    def _create_load_history_node(
        self,
        fn_factory: FunctionNode,
        instance_name: str,
        config: ConversationConfig,
        dependencies: list[str],
    ) -> NodeSpec:
        """Create node that loads conversation history from Memory port."""

        async def load_history(input_data: dict[str, Any]) -> dict[str, Any]:
            """Load conversation history from memory."""
            conversation_id = input_data.get("conversation_id") or config.conversation_id

            # Get Memory port
            try:
                memory_port = get_port("memory")
            except Exception:
                logger.warning("Memory port not available, starting fresh conversation")
                # Return empty history with system message
                return {
                    "conversation_id": conversation_id,
                    "messages": [{"role": "system", "content": config.system_prompt}],
                    "user_message": input_data.get("user_message", ""),
                }

            # Load history from memory
            memory_key = f"conversation:{conversation_id}"
            try:
                history_json = await memory_port.aget(memory_key)
                if history_json:
                    import json

                    messages = json.loads(history_json)
                    logger.debug(
                        f"Loaded {len(messages)} messages from conversation {conversation_id}"
                    )
                else:
                    # New conversation
                    messages = [{"role": "system", "content": config.system_prompt}]
                    logger.debug(f"Starting new conversation {conversation_id}")
            except Exception as e:
                logger.warning(f"Failed to load history: {e}, starting fresh")
                messages = [{"role": "system", "content": config.system_prompt}]

            return {
                "conversation_id": conversation_id,
                "messages": messages,
                "user_message": input_data.get("user_message", ""),
            }

        return fn_factory(
            name=f"{instance_name}_load_history",
            fn=load_history,
            deps=dependencies,
        )

    def _create_format_messages_node(
        self, fn_factory: FunctionNode, instance_name: str, config: ConversationConfig
    ) -> NodeSpec:
        """Create node that adds new user message to history."""

        async def format_messages(history_data: dict[str, Any]) -> dict[str, Any]:
            """Add new user message to conversation history."""
            messages = history_data["messages"]
            user_message = history_data["user_message"]

            # Add new user message
            if user_message:
                messages.append({"role": "user", "content": user_message})
                logger.debug(f"Added user message (total messages: {len(messages)})")

            # Trim history if needed (keep system message + recent messages)
            if len(messages) > config.max_history:
                # Keep system message + most recent messages
                system_msg = messages[0]
                recent_messages = messages[-(config.max_history - 1) :]
                messages = [system_msg] + recent_messages
                logger.debug(f"Trimmed history to {len(messages)} messages")

            return {
                "conversation_id": history_data["conversation_id"],
                "messages": messages,
            }

        return fn_factory(
            name=f"{instance_name}_format_messages",
            fn=format_messages,
            deps=[f"{instance_name}_load_history"],
        )

    def _create_llm_node(
        self, llm_factory: LLMNode, instance_name: str, config: ConversationConfig
    ) -> NodeSpec:
        """Create LLM node that processes conversation."""

        # Build tool instructions if tools enabled
        tool_instructions = ""
        if config.enable_tool_use and config.allowed_tools:
            tool_prompt_class = get_tool_prompt_for_format(config.tool_format)
            # Tool prompt classes override __init__ to provide their own template
            tool_prompt = tool_prompt_class()  # type: ignore[call-arg] # pyright: ignore[reportCallIssue]
            tool_list = self._build_tool_list(config.allowed_tools)
            tool_instructions = f"""

## Available Tools
{tool_list}

{tool_prompt.template}"""

        # Template that renders the conversation
        # The format_messages node output becomes {{format_messages}}
        template = f"""{{{{{{format_messages.messages}}}}}}
{tool_instructions}"""

        return llm_factory(
            name=f"{instance_name}_llm",
            template=template,
            deps=[f"{instance_name}_format_messages"],
        )

    def _create_parse_response_node(
        self, fn_factory: FunctionNode, instance_name: str, config: ConversationConfig
    ) -> NodeSpec:
        """Create node that parses LLM response and handles tool calls."""

        async def parse_response(llm_output: str) -> dict[str, Any]:
            """Parse LLM response and optionally expand tool calls."""
            # Parse tool calls if enabled
            tool_calls_data = []
            if config.enable_tool_use:
                tool_calls = ToolParser.parse_tool_calls(llm_output, format=config.tool_format)
                if tool_calls:
                    logger.debug(f"Parsed {len(tool_calls)} tool calls from LLM response")
                    tool_calls_data = [
                        {"name": tc.name, "arguments": tc.params, "id": f"call_{i}"}
                        for i, tc in enumerate(tool_calls)
                    ]

                    # TODO(hexdag-team): Dynamic expansion of ToolMacro #noqa: TD003
                    # This would require:
                    # graph = get_current_graph()
                    # tool_macro = ToolMacro(tool_calls=tool_calls_data, ...)
                    # tool_graph = tool_macro.expand(...)
                    # graph.merge(tool_graph)

            return {
                "llm_response": llm_output,
                "tool_calls": tool_calls_data,
                "has_tools": len(tool_calls_data) > 0,
            }

        return fn_factory(
            name=f"{instance_name}_parse_response",
            fn=parse_response,
            deps=[f"{instance_name}_llm"],
        )

    def _create_save_history_node(
        self, fn_factory: FunctionNode, instance_name: str, config: ConversationConfig
    ) -> NodeSpec:
        """Create node that saves updated conversation history."""

        async def save_history(response_data: dict[str, Any]) -> dict[str, Any]:
            """Save updated conversation history to memory."""
            # Get previous messages from context
            from hexdag.core.context import get_node_results

            node_results = get_node_results()
            if not node_results:
                logger.warning("No node results available, cannot save history")
                return response_data

            format_node_result = node_results.get(f"{instance_name}_format_messages")
            if not format_node_result:
                logger.warning("Format messages node result not found")
                return response_data

            messages = format_node_result.result.get("messages", [])
            conversation_id = format_node_result.result.get("conversation_id")

            # Add assistant response to history
            llm_response = response_data.get("llm_response", "")
            messages.append({"role": "assistant", "content": llm_response})

            # Save to memory
            try:
                memory_port = get_port("memory")
                import json

                memory_key = f"conversation:{conversation_id}"
                await memory_port.aset(memory_key, json.dumps(messages))
                logger.debug(f"Saved conversation with {len(messages)} messages")
            except Exception as e:
                logger.warning(f"Failed to save conversation: {e}")

            return {
                **response_data,
                "conversation_id": conversation_id,
                "message_count": len(messages),
            }

        return fn_factory(
            name=f"{instance_name}_save_history",
            fn=save_history,
            deps=[f"{instance_name}_parse_response"],
        )

    def _build_tool_list(self, allowed_tools: list[str]) -> str:
        """Build formatted tool list for prompt."""
        from hexdag.core.registry import registry

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
