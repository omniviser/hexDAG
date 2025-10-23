"""ConversationMacro - Multi-turn chat with dynamic message history expansion.

Architecture:
- Loads conversation history from Memory port
- Accumulates messages dynamically during execution
- Supports tools with dynamic ToolMacro expansion
- Maintains conversation state across turns
- Automatic context window management

Requirements:
- Memory port must be configured in the pipeline for persistence
- Without memory port, conversations start fresh each time

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

from hexdag.builtin.macros.reasoning_agent import ReasoningAgentMacro
from hexdag.builtin.nodes.function_node import FunctionNode
from hexdag.builtin.nodes.tool_utils import ToolCallFormat
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
    memory_adapter : str | None
        Optional memory adapter to use (default: uses pipeline's memory port)
        If None, will use the memory port configured at pipeline level
    """

    system_prompt: str = Field(default="You are a helpful assistant")
    conversation_id: str
    max_history: int = Field(default=20, ge=2)
    allowed_tools: list[str] = Field(default_factory=list)
    tool_format: ToolCallFormat = Field(default=ToolCallFormat.MIXED)
    enable_tool_use: bool = Field(default=True)
    memory_adapter: str | None = Field(
        default=None,
        description="Optional memory adapter override (e.g., 'plugin:redis_memory')",
    )


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
    YAML configuration with memory port:

        apiVersion: v1
        kind: Pipeline
        metadata:
          name: chatbot_pipeline
        spec:
          ports:
            memory:
              adapter: in_memory_memory
              config:
                max_size: 1000
            llm:
              adapter: plugin:openai
              config:
                model: gpt-4
          nodes:
            - kind: macro_invocation
              metadata:
                name: chatbot
              spec:
                macro: core:conversation
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
        """Expand into conversation nodes using ReasoningAgent for core logic.

        Graph structure:
        ```
        [deps] → load_history → format_prompt → reasoning_agent → save_history
        ```

        During execution:
        - load_history: Gets conversation from Memory port
        - format_prompt: Formats conversation into prompt for reasoning agent
        - reasoning_agent: Uses ReasoningAgent for multi-step reasoning with tools
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

        # Node 1: Load conversation history from Memory
        load_node = self._create_load_history_node(fn_factory, instance_name, config, dependencies)
        graph += load_node

        # Node 2: Format conversation into prompt for reasoning agent
        format_node = self._create_format_prompt_node(fn_factory, instance_name, config)
        graph += format_node

        # Node 3: Use ReasoningAgent for core reasoning and tool execution
        # Create ReasoningAgent with config from ConversationConfig
        reasoning_macro = ReasoningAgentMacro(
            main_prompt="{{conversation_prompt}}",  # Will be filled from format_prompt node
            max_steps=3,  # Allow multi-step reasoning
            allowed_tools=config.allowed_tools if config.enable_tool_use else [],
            tool_format=config.tool_format,
        )

        # Expand reasoning agent with dependency on format_prompt
        reasoning_graph = reasoning_macro.expand(
            f"{instance_name}_reasoning", inputs, [f"{instance_name}_format_prompt"]
        )
        graph |= reasoning_graph

        # Node 4: Save updated conversation history
        save_node = self._create_save_history_node(
            fn_factory,
            instance_name,
            config,
            f"{instance_name}_reasoning_final",  # Depend on reasoning agent's final output
        )
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

            # Get Memory port (use configured adapter or default)
            try:
                if config.memory_adapter:
                    # Use specific adapter if configured
                    from hexdag.core.registry import registry

                    memory_port = registry.get(config.memory_adapter)
                    # Verify it has the required methods
                    assert hasattr(memory_port, "aget") and hasattr(memory_port, "aset")
                else:
                    # Use default memory port from pipeline
                    memory_port = get_port("memory")
            except Exception as e:
                logger.warning(f"Memory port not available ({e}), starting fresh conversation")
                # Return empty history with system message
                return {
                    "conversation_id": conversation_id,
                    "messages": [{"role": "system", "content": config.system_prompt}],
                    "user_message": input_data.get("user_message", ""),
                }

            # Load history from memory
            memory_key = f"conversation:{conversation_id}"
            try:
                history_json = await memory_port.aget(memory_key)  # pyright: ignore[reportAttributeAccessIssue]
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

    def _create_format_prompt_node(
        self, fn_factory: FunctionNode, instance_name: str, config: ConversationConfig
    ) -> NodeSpec:
        """Create node that formats conversation history into a prompt for reasoning agent."""

        async def format_prompt(history_data: dict[str, Any]) -> dict[str, Any]:
            """Format conversation history into prompt for reasoning agent."""
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

            # Format messages into a prompt for reasoning agent
            # Include system prompt and conversation context
            conversation_context = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}" for msg in messages
            ])

            prompt = f"""{config.system_prompt}

## Conversation History:
{conversation_context}

Please provide a thoughtful response to continue this conversation."""

            return {
                "conversation_id": history_data["conversation_id"],
                "messages": messages,
                "conversation_prompt": prompt,
            }

        return fn_factory(
            name=f"{instance_name}_format_prompt",
            fn=format_prompt,
            deps=[f"{instance_name}_load_history"],
        )

    def _create_save_history_node(
        self,
        fn_factory: FunctionNode,
        instance_name: str,
        config: ConversationConfig,
        reasoning_node: str,
    ) -> NodeSpec:
        """Create node that saves updated conversation history."""

        async def save_history(reasoning_response: Any) -> dict[str, Any]:
            """Save updated conversation history to memory."""
            # Get previous messages from context
            from hexdag.core.context import get_node_results

            node_results = get_node_results()
            if not node_results:
                logger.warning("No node results available, cannot save history")
                return {"response": str(reasoning_response)}

            format_node_result = node_results.get(f"{instance_name}_format_prompt")
            if not format_node_result:
                logger.warning("Format prompt node result not found")
                return {"response": str(reasoning_response)}

            messages = format_node_result.result.get("messages", [])
            conversation_id = format_node_result.result.get("conversation_id")

            # Add assistant response to history
            # ReasoningAgent returns a string response
            assistant_response = str(reasoning_response) if reasoning_response else ""
            messages.append({"role": "assistant", "content": assistant_response})

            # Save to memory
            try:
                if config.memory_adapter:
                    # Use specific adapter if configured
                    from hexdag.core.registry import registry

                    memory_port = registry.get(config.memory_adapter)
                    # Verify it has the required method
                    assert hasattr(memory_port, "aset")
                else:
                    # Use default memory port from pipeline
                    memory_port = get_port("memory")

                import json

                memory_key = f"conversation:{conversation_id}"
                await memory_port.aset(memory_key, json.dumps(messages))  # pyright: ignore[reportAttributeAccessIssue]
                logger.debug(f"Saved conversation with {len(messages)} messages")
            except Exception as e:
                logger.warning(f"Failed to save conversation: {e}")

            return {
                "response": assistant_response,
                "conversation_id": conversation_id,
                "message_count": len(messages),
            }

        return fn_factory(
            name=f"{instance_name}_save_history",
            fn=save_history,
            deps=[reasoning_node],
        )
