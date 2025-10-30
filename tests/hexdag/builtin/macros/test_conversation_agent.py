"""Tests for ConversationMacro with memory port support."""

from hexdag.builtin.macros.conversation_agent import ConversationMacro
from hexdag.builtin.nodes.tool_utils import ToolCallFormat
from hexdag.core.domain.dag import DirectedGraph


class TestConversationMacro:
    """Test suite for ConversationMacro functionality."""

    def test_conversation_macro_basic_expansion(self):
        """Test basic macro expansion creates expected nodes."""
        macro = ConversationMacro(
            system_prompt="You are a helpful assistant",
            conversation_id="test_123",
            max_history=10,
            enable_tool_use=False,
        )

        graph = macro.expand(
            instance_name="chatbot", inputs={"user_message": "Hello"}, dependencies=[]
        )

        assert isinstance(graph, DirectedGraph)
        assert len(graph.nodes) > 0

        # Check key nodes exist
        assert "chatbot_load_history" in graph.nodes
        assert "chatbot_format_prompt" in graph.nodes
        assert "chatbot_save_history" in graph.nodes

        # Should have reasoning agent nodes
        reasoning_nodes = [n for n in graph.nodes if "reasoning" in n]
        assert len(reasoning_nodes) > 0

    def test_conversation_macro_with_tools(self):
        """Test macro expansion with tool support."""
        macro = ConversationMacro(
            system_prompt="You are an assistant with tools",
            conversation_id="test_456",
            max_history=5,
            allowed_tools=["core:search", "core:calculate"],
            enable_tool_use=True,
            tool_format=ToolCallFormat.MIXED,
        )

        graph = macro.expand(
            instance_name="agent", inputs={"user_message": "Search for AI"}, dependencies=["init"]
        )

        # Should have more nodes due to tool support
        assert len(graph.nodes) > 10

        # Load history should depend on initial dependencies
        load_history = graph.nodes["agent_load_history"]
        assert "init" in load_history.deps

        # Format prompt should depend on load_history
        format_prompt = graph.nodes["agent_format_prompt"]
        assert "agent_load_history" in format_prompt.deps

        # Save history should be at the end
        save_history = graph.nodes["agent_save_history"]
        assert "agent_reasoning_final" in save_history.deps

    def test_conversation_macro_with_memory_adapter(self):
        """Test macro with custom memory adapter configuration."""
        macro = ConversationMacro(
            system_prompt="Memory test assistant",
            conversation_id="test_789",
            max_history=3,
            memory_adapter="plugin:redis_memory",  # Custom adapter
            enable_tool_use=False,
        )

        graph = macro.expand(instance_name="mem_test", inputs={}, dependencies=[])

        # Should still create all nodes regardless of adapter choice
        assert "mem_test_load_history" in graph.nodes
        assert "mem_test_save_history" in graph.nodes

        # The adapter will be resolved at runtime through registry

    def test_conversation_macro_dependencies(self):
        """Test that node dependencies are correctly set up."""
        macro = ConversationMacro(
            system_prompt="Dependency test",
            conversation_id="dep_test",
            enable_tool_use=False,
        )

        graph = macro.expand(instance_name="dep", inputs={}, dependencies=["external_dep"])

        # Check dependency chain
        load_history = graph.nodes["dep_load_history"]
        assert "external_dep" in load_history.deps

        format_prompt = graph.nodes["dep_format_prompt"]
        assert "dep_load_history" in format_prompt.deps

        # Reasoning nodes should depend on format_prompt
        reasoning_nodes = [n for n in graph.nodes if "reasoning" in n and "prompt" in n]
        for node_name in reasoning_nodes:
            node = graph.nodes[node_name]
            # At least one should depend on format_prompt or previous reasoning steps
            deps_str = " ".join(node.deps)
            assert "dep_format_prompt" in deps_str or "reasoning" in deps_str

    def test_conversation_config_validation(self):
        """Test that ConversationConfig validates inputs correctly."""
        # Should work with minimal config
        macro = ConversationMacro(
            conversation_id="minimal_test",
        )
        assert macro.config.system_prompt == "You are a helpful assistant"
        assert macro.config.max_history == 20
        assert macro.config.enable_tool_use is True

        # Should work with full config
        macro = ConversationMacro(
            system_prompt="Custom assistant",
            conversation_id="full_test",
            max_history=50,
            allowed_tools=["tool1", "tool2"],
            tool_format=ToolCallFormat.JSON,
            enable_tool_use=True,
            memory_adapter="custom:memory",
        )
        assert macro.config.system_prompt == "Custom assistant"
        assert macro.config.max_history == 50
        assert len(macro.config.allowed_tools) == 2
        assert macro.config.memory_adapter == "custom:memory"

    def test_conversation_macro_yaml_compatibility(self):
        """Test that macro config works with YAML-style input."""
        # Simulate YAML-parsed config
        yaml_config = {
            "system_prompt": "YAML assistant",
            "conversation_id": "yaml_123",
            "max_history": 15,
            "allowed_tools": ["core:tool1"],
            "enable_tool_use": True,
        }

        macro = ConversationMacro(**yaml_config)
        graph = macro.expand(
            instance_name="yaml_test", inputs={"user_message": "Test"}, dependencies=[]
        )

        assert isinstance(graph, DirectedGraph)
        assert len(graph.nodes) > 0

    def test_reasoning_agent_integration(self):
        """Test that ConversationMacro properly integrates ReasoningAgentMacro."""
        macro = ConversationMacro(
            conversation_id="reasoning_test",
            allowed_tools=["tool1", "tool2"],
            enable_tool_use=True,
        )

        graph = macro.expand(instance_name="reason", inputs={}, dependencies=[])

        # Should have reasoning agent nodes (multiple steps)
        step_nodes = [n for n in graph.nodes if "_step_" in n]
        assert len(step_nodes) > 0

        # Should have final reasoning node
        assert "reason_reasoning_final" in graph.nodes

        # Save history should depend on final reasoning
        save_history = graph.nodes["reason_save_history"]
        assert "reason_reasoning_final" in save_history.deps
