"""Tests for Node Factory functionality.

Tests cover:
- Node type registration
- Node creation through factory
- Registry management functions
- Builtin node types registration
- Parameter validation
"""

import pytest
from fastapi_app.src.hexai.app.application.nodes.node_factory import NodeFactory
from fastapi_app.src.hexai.app.domain.dag import NodeSpec


class TestNodeFactory:
    """Test cases for unified node factory functionality."""

    def setup_method(self):
        """Clear registry before each test."""
        NodeFactory.clear_registry()

    def teardown_method(self):
        """Clear registry after each test to avoid side effects."""
        NodeFactory.clear_registry()

    def test_register_node_type(self):
        """Test that node type registration works correctly."""

        def create_test_node(name: str, value: int = 42) -> NodeSpec:
            def test_fn():
                return value

            return NodeSpec(name, test_fn)

        NodeFactory.register_node_type("test_type", create_test_node, "Test node description")

        # Check that the function was registered
        assert NodeFactory.node_type_exists("test_type")
        assert NodeFactory.get_registry_size() == 1

    def test_register_duplicate_type_raises_error(self):
        """Test that registering duplicate type names raises an error."""

        def first_factory(name: str) -> NodeSpec:
            return NodeSpec(name, lambda: "first")

        def second_factory(name: str) -> NodeSpec:
            return NodeSpec(name, lambda: "second")

        NodeFactory.register_node_type("duplicate", first_factory)

        # Attempting to register the same type again should raise an error
        with pytest.raises(ValueError, match="Node type 'duplicate' is already registered"):
            NodeFactory.register_node_type("duplicate", second_factory)

    def test_create_node_success(self):
        """Test successful node creation using create_node."""

        def create_math_node(name: str, operation: str, operand: int = 1) -> NodeSpec:
            def math_fn(x: int) -> int:
                if operation == "add":
                    return x + operand
                elif operation == "multiply":
                    return x * operand
                return x

            return NodeSpec(name, math_fn)

        NodeFactory.register_node_type("math", create_math_node)

        # Create a node using the factory
        node = NodeFactory.create_node("math", "add_five", operation="add", operand=5)

        assert isinstance(node, NodeSpec)
        assert node.name == "add_five"
        # Test the wrapped function
        result = node.fn(10)
        assert result == 15

    def test_create_node_with_no_params(self):
        """Test create_node with no additional parameters."""

        def create_simple_node(name: str) -> NodeSpec:
            def simple_fn():
                return "simple_result"

            return NodeSpec(name, simple_fn)

        NodeFactory.register_node_type("simple", create_simple_node)

        node = NodeFactory.create_node("simple", "test_node")
        assert node.name == "test_node"
        assert node.fn() == "simple_result"

    def test_create_node_unregistered_type_raises_error(self):
        """Test that create_node raises ValueError for unregistered types."""
        with pytest.raises(ValueError, match="Unknown node type: 'nonexistent'"):
            NodeFactory.create_node("nonexistent", "test_id")

    def test_create_node_factory_error_raises_type_error(self):
        """Test that factory function errors are properly wrapped."""

        def failing_factory(name: str, required_param: str) -> NodeSpec:
            # This will fail if required_param is not provided
            return NodeSpec(name, lambda: required_param.upper())

        NodeFactory.register_node_type("failing", failing_factory)

        with pytest.raises(TypeError, match="Failed to create node 'test_id' of type 'failing'"):
            # Missing required_param should cause factory to fail
            NodeFactory.create_node("failing", "test_id")

    def test_list_registered_types(self):
        """Test listing all registered types."""

        def factory_a(name: str) -> NodeSpec:
            return NodeSpec(name, lambda: "a")

        def factory_b(name: str) -> NodeSpec:
            return NodeSpec(name, lambda: "b")

        NodeFactory.register_node_type("type_a", factory_a)
        NodeFactory.register_node_type("type_b", factory_b)

        types = NodeFactory.list_registered_types()
        assert set(types) == {"type_a", "type_b"}
        assert len(types) == 2

    def test_clear_registry(self):
        """Test clearing the registry."""

        def temp_factory(name: str) -> NodeSpec:
            return NodeSpec(name, lambda: "temp")

        NodeFactory.register_node_type("temp", temp_factory)
        assert NodeFactory.get_registry_size() == 1

        NodeFactory.clear_registry()

        assert NodeFactory.get_registry_size() == 0
        assert NodeFactory.list_registered_types() == []

    def test_get_registry_size(self):
        """Test getting registry size."""
        assert NodeFactory.get_registry_size() == 0

        def factory1(name: str) -> NodeSpec:
            return NodeSpec(name, lambda: 1)

        def factory2(name: str) -> NodeSpec:
            return NodeSpec(name, lambda: 2)

        NodeFactory.register_node_type("size_test_1", factory1)
        assert NodeFactory.get_registry_size() == 1

        NodeFactory.register_node_type("size_test_2", factory2)
        assert NodeFactory.get_registry_size() == 2

    def test_get_node_info(self):
        """Test getting node type information."""

        def test_factory(name: str) -> NodeSpec:
            return NodeSpec(name, lambda: "test")

        NodeFactory.register_node_type("info_test", test_factory, "Test description")

        info = NodeFactory.get_node_info("info_test")
        assert info["type"] == "info_test"
        assert info["description"] == "Test description"
        assert info["factory_function"] == "test_factory"

    def test_get_node_info_unregistered_raises_error(self):
        """Test that get_node_info raises error for unregistered types."""
        with pytest.raises(ValueError, match="Unknown node type: 'nonexistent'"):
            NodeFactory.get_node_info("nonexistent")

    def test_unregister_node_type(self):
        """Test unregistering a node type."""

        def test_factory(name: str) -> NodeSpec:
            return NodeSpec(name, lambda: "test")

        NodeFactory.register_node_type("to_unregister", test_factory)
        assert NodeFactory.node_type_exists("to_unregister")

        NodeFactory.unregister_node_type("to_unregister")
        assert not NodeFactory.node_type_exists("to_unregister")

    def test_unregister_nonexistent_type_raises_error(self):
        """Test that unregistering nonexistent type raises error."""
        with pytest.raises(ValueError, match="Node type 'nonexistent' is not registered"):
            NodeFactory.unregister_node_type("nonexistent")

    def test_register_custom_node(self):
        """Test registering a custom node through factory."""

        def custom_factory(node_id: str, **params) -> NodeSpec:
            def custom_fn():
                return f"custom_{node_id}"

            return NodeSpec(node_id, custom_fn)

        NodeFactory.register_node_type("custom", custom_factory, "Custom node type")

        node = NodeFactory.create_node("custom", "my_custom")
        assert node.name == "my_custom"
        assert node.fn() == "custom_my_custom"


class TestBuiltinNodes:
    """Test cases for builtin node types."""

    def setup_method(self):
        """Setup for builtin node tests."""
        NodeFactory.clear_registry()
        # Manually register builtin nodes since import only happens once
        from fastapi_app.src.hexai.app.application.nodes.builtin_nodes import (
            agent_node_factory,
            function_node_factory,
            llm_node_factory,
        )

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

    def teardown_method(self):
        """Cleanup after builtin node tests."""
        NodeFactory.clear_registry()

    def test_builtin_nodes_are_registered(self):
        """Test that all builtin nodes are automatically registered."""
        registered_types = NodeFactory.list_registered_types()

        # Check that all expected builtin types are registered
        expected_types = {"function", "llm", "agent"}
        assert expected_types.issubset(set(registered_types))

    def test_builtin_nodes_have_descriptions(self):
        """Test that builtin nodes have descriptions."""
        function_info = NodeFactory.get_node_info("function")
        llm_info = NodeFactory.get_node_info("llm")
        agent_info = NodeFactory.get_node_info("agent")

        assert function_info["description"]  # Should not be empty
        assert llm_info["description"]  # Should not be empty
        assert agent_info["description"]  # Should not be empty

    def test_function_node_creation(self):
        """Test creating a function node through the factory."""

        def add_numbers(a: int, b: int) -> int:
            return a + b

        node = NodeFactory.create_node("function", "adder", fn=add_numbers)

        assert isinstance(node, NodeSpec)
        assert node.name == "adder"

    def test_llm_node_creation(self):
        """Create an LLM node through the factory."""
        node = NodeFactory.create_node("llm", "analyzer", prompt_template="Analyze: {{input}}")

        assert isinstance(node, NodeSpec)
        assert node.name == "analyzer"

    def test_llm_node_creation_with_string_template(self):
        """Create an LLM node with string template."""
        node = NodeFactory.create_node("llm", "processor", prompt_template="Process: {{data}}")

        assert isinstance(node, NodeSpec)
        assert node.name == "processor"

    def test_agent_node_creation(self):
        """Create a reasoning agent node through the factory."""
        node = NodeFactory.create_node("agent", "reasoner", max_steps=3)

        assert isinstance(node, NodeSpec)
        assert node.name == "reasoner"

    def test_agent_node_creation_with_tool_router(self):
        """Create a reasoning agent node with tool router through the factory."""
        node = NodeFactory.create_node(
            "agent", "tool_reasoner", max_steps=3, available_tools=["search"]
        )

        assert isinstance(node, NodeSpec)
        assert node.name == "tool_reasoner"
