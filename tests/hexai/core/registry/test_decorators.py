"""Tests for the simplified decorators module."""

import pytest

from hexai.core.registry import registry
from hexai.core.registry.decorators import (
    _snake_case,
    adapter,
    agent_node,
    component,
    function_node,
    llm_node,
    memory,
    node,
    observer,
    policy,
    tool,
)
from hexai.core.registry.exceptions import ComponentAlreadyRegisteredError
from hexai.core.registry.types import ComponentType  # Internal for tests
from hexai.core.registry.types import Namespace, NodeSubtype


class TestSnakeCase:
    """Test snake_case conversion."""

    def test_simple_camel_case(self):
        """Test simple CamelCase conversion."""
        assert _snake_case("CamelCase") == "camel_case"
        assert _snake_case("SimpleTest") == "simple_test"

    def test_consecutive_capitals(self):
        """Test handling of consecutive capital letters."""
        assert _snake_case("HTTPServer") == "http_server"
        assert _snake_case("XMLParser") == "xml_parser"

    def test_mixed_case(self):
        """Test mixed case handling."""
        assert _snake_case("getHTTPResponseCode") == "get_http_response_code"
        assert _snake_case("HTTPSConnection") == "https_connection"

    def test_already_snake_case(self):
        """Test strings already in snake_case."""
        assert _snake_case("already_snake_case") == "already_snake_case"
        assert _snake_case("snake_case") == "snake_case"

    def test_single_word(self):
        """Test single word conversion."""
        assert _snake_case("Word") == "word"
        assert _snake_case("word") == "word"

    def test_leading_underscore(self):
        """Test handling of leading underscores."""
        assert _snake_case("_Leading") == "leading"
        assert _snake_case("__DoubleLeading") == "double_leading"


class TestComponentDecorator:
    """Test the main component decorator with direct registration."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Create a fresh registry for each test."""
        # Clear the registry before each test
        registry._components.clear()
        registry._protected_components.clear()
        yield
        # Clean up after
        registry._components.clear()
        registry._protected_components.clear()

    def test_basic_decoration(self):
        """Test basic component decoration and registration."""

        @component(ComponentType.NODE, namespace="test")
        class TestComponent:
            """Test component."""

            pass

        # Component should be registered immediately
        metadata = registry.get_metadata("test_component", namespace="test")
        assert metadata.name == "test_component"
        assert metadata.component_type == ComponentType.NODE
        assert metadata.namespace == "test"
        assert metadata.description == "Test component."

    def test_custom_name(self):
        """Test decoration with custom name."""

        @component(ComponentType.NODE, name="custom", namespace="test")
        class TestComponent:
            pass

        # Should use custom name
        metadata = registry.get_metadata("custom", namespace="test")
        assert metadata.name == "custom"

    def test_description_from_docstring(self):
        """Test description extraction from docstring."""

        @component(ComponentType.NODE, namespace="test")
        class DocumentedComponent:
            """Well-documented component.

            It has multiple lines.
            """

            pass

        metadata = registry.get_metadata("documented_component", namespace="test")
        assert "Well-documented component" in metadata.description

    def test_explicit_description(self):
        """Test explicit description parameter."""

        @component(ComponentType.NODE, namespace="test", description="Explicit description")
        class TestComponent:
            """Docstring description."""

            pass

        metadata = registry.get_metadata("test_component", namespace="test")
        assert metadata.description == "Explicit description"

    def test_subtype_parameter(self):
        """Test subtype parameter for nodes."""

        @component(ComponentType.NODE, namespace="test", subtype=NodeSubtype.FUNCTION)
        class FunctionComponent:
            pass

        metadata = registry.get_metadata("function_component", namespace="test")
        assert metadata.subtype == NodeSubtype.FUNCTION

    def test_core_namespace_privilege(self):
        """Test that core namespace gets privileged access."""

        @component(ComponentType.NODE, namespace=Namespace.CORE)
        class CoreComponent:
            pass

        # Core component should be marked as protected
        assert "core_component" in registry._protected_components
        metadata = registry.get_metadata("core_component", namespace=Namespace.CORE)
        assert metadata.namespace == Namespace.CORE


class TestTypeSpecificDecorators:
    """Test type-specific decorator shortcuts."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Create a fresh registry for each test."""
        registry._components.clear()
        registry._protected_components.clear()
        yield
        registry._components.clear()
        registry._protected_components.clear()

    def test_node_decorator(self):
        """Test @node decorator."""

        @node(namespace="test")
        class TestNode:
            pass

        metadata = registry.get_metadata("test_node", namespace="test")
        assert metadata.component_type == ComponentType.NODE

    def test_tool_decorator(self):
        """Test @tool decorator."""

        @tool(namespace="test")
        class TestTool:
            pass

        metadata = registry.get_metadata("test_tool", namespace="test")
        assert metadata.component_type == ComponentType.TOOL

    def test_adapter_decorator(self):
        """Test @adapter decorator."""

        @adapter(namespace="test")
        class TestAdapter:
            pass

        metadata = registry.get_metadata("test_adapter", namespace="test")
        assert metadata.component_type == ComponentType.ADAPTER

    def test_policy_decorator(self):
        """Test @policy decorator."""

        @policy(namespace="test")
        class TestPolicy:
            pass

        metadata = registry.get_metadata("test_policy", namespace="test")
        assert metadata.component_type == ComponentType.POLICY

    def test_memory_decorator(self):
        """Test @memory decorator."""

        @memory(namespace="test")
        class TestMemory:
            pass

        metadata = registry.get_metadata("test_memory", namespace="test")
        assert metadata.component_type == ComponentType.MEMORY

    def test_observer_decorator(self):
        """Test @observer decorator."""

        @observer(namespace="test")
        class TestObserver:
            pass

        metadata = registry.get_metadata("test_observer", namespace="test")
        assert metadata.component_type == ComponentType.OBSERVER

    def test_function_node_decorator(self):
        """Test @function_node decorator."""

        @function_node(namespace="test")
        class TestFunctionNode:
            pass

        metadata = registry.get_metadata("test_function_node", namespace="test")
        assert metadata.component_type == ComponentType.NODE
        assert metadata.subtype == NodeSubtype.FUNCTION

    def test_llm_node_decorator(self):
        """Test @llm_node decorator."""

        @llm_node(namespace="test")
        class TestLLMNode:
            pass

        metadata = registry.get_metadata("test_llm_node", namespace="test")
        assert metadata.component_type == ComponentType.NODE
        assert metadata.subtype == NodeSubtype.LLM

    def test_agent_node_decorator(self):
        """Test @agent_node decorator."""

        @agent_node(namespace="test")
        class TestAgentNode:
            pass

        metadata = registry.get_metadata("test_agent_node", namespace="test")
        assert metadata.component_type == ComponentType.NODE
        assert metadata.subtype == NodeSubtype.AGENT


class TestDecoratorIntegration:
    """Test decorator integration with registry."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Create a fresh registry for each test."""
        registry._components.clear()
        registry._protected_components.clear()
        yield
        registry._components.clear()
        registry._protected_components.clear()

    def test_multiple_components_same_namespace(self):
        """Test registering multiple components in same namespace."""

        @node(namespace="test")
        class Node1:
            pass

        @node(namespace="test")
        class Node2:
            pass

        @tool(namespace="test")
        class Tool1:
            pass

        # All should be registered
        assert registry.get("node1", namespace="test") is not None
        assert registry.get("node2", namespace="test") is not None
        assert registry.get("tool1", namespace="test") is not None

    def test_component_replacement(self):
        """Test that decorators don't replace by default."""

        @node(namespace="test")
        class OriginalNode:
            pass

        # This should raise an error since replace=False by default
        with pytest.raises(ComponentAlreadyRegisteredError):

            @node(namespace="test", name="original_node")
            class ReplacementNode:
                pass

    def test_get_component_instance(self):
        """Test getting component instance from registry."""

        @node(namespace="test")
        class SimpleNode:
            def __init__(self, value=42):
                self.value = value

        # Get instance with default args
        instance = registry.get("simple_node", namespace="test")
        assert isinstance(instance, SimpleNode)
        assert instance.value == 42

        # Get instance with custom args
        instance = registry.get("simple_node", namespace="test", value=100)
        assert instance.value == 100


class TestFunctionBehavior:
    """Test that functions are NOT called when retrieved from registry."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clean registry before each test."""
        registry._components.clear()
        registry._protected_components.clear()
        yield
        registry._components.clear()
        registry._protected_components.clear()

    def test_function_not_called_on_get(self):
        """Test that registry.get() returns function without calling it."""
        call_count = 0

        @tool(namespace="test")
        def counting_tool():
            """Tool that counts how many times it's called."""
            nonlocal call_count
            call_count += 1
            return f"called {call_count} times"

        # Get the function from registry
        func = registry.get("counting_tool", namespace="test")

        # Function should NOT have been called yet
        assert call_count == 0, "Function was called during get()!"
        assert callable(func), "Should return a callable"
        assert func is counting_tool, "Should return the original function"

        # Now call it manually
        result = func()
        assert call_count == 1
        assert result == "called 1 times"

    def test_function_with_side_effects_not_triggered(self):
        """Test that functions with side effects aren't triggered on get()."""
        side_effects = []

        @tool(namespace="test")
        def side_effect_tool(value):
            """Tool with side effects."""
            side_effects.append(f"executed with {value}")
            return value * 2

        # Get the function - should NOT trigger side effects
        func = registry.get("side_effect_tool", namespace="test")
        assert len(side_effects) == 0, "Side effects triggered during get()!"

        # Now call it manually
        result = func(5)
        assert result == 10
        assert side_effects == ["executed with 5"]

    def test_kwargs_ignored_for_functions(self):
        """Test that kwargs passed to get() are ignored for functions."""
        call_args = []

        @tool(namespace="test")
        def arg_tracking_tool(x=1, y=2):
            """Tool that tracks its arguments."""
            call_args.append((x, y))
            return x + y

        # Get with kwargs - they should be ignored
        func = registry.get("arg_tracking_tool", namespace="test", x=100, y=200)

        # Function should not have been called
        assert len(call_args) == 0

        # When we call it, it uses its own defaults, not the get() kwargs
        result = func()
        assert result == 3  # 1 + 2 (defaults)
        assert call_args == [(1, 2)]

    def test_function_returned_same_instance(self):
        """Test that same function instance is returned each time."""

        @tool(namespace="test")
        def singleton_tool():
            return "data"

        # Get multiple times
        func1 = registry.get("singleton_tool", namespace="test")
        func2 = registry.get("singleton_tool", namespace="test")
        func3 = registry.get("singleton_tool", namespace="test", unused_kwarg="ignored")

        # Should all be the same function object
        assert func1 is func2 is func3 is singleton_tool

    def test_exception_not_raised_on_get(self):
        """Ensure function exceptions aren't raised during get()."""

        @tool(namespace="test")
        def failing_tool():
            """Tool that always raises an exception."""
            raise ValueError("This tool always fails!")

        # Getting the function should NOT raise the exception
        func = registry.get("failing_tool", namespace="test")
        assert callable(func)

        # But calling it should raise
        with pytest.raises(ValueError, match="This tool always fails"):
            func()

    def test_generator_function_not_started(self):
        """Test that generator functions aren't started on get()."""

        @tool(namespace="test")
        def generator_tool():
            """Yield values from generator."""
            yield 1
            yield 2
            yield 3

        # Get the generator function
        func = registry.get("generator_tool", namespace="test")

        # It should be the function, not a generator instance
        assert callable(func)
        assert func is generator_tool

        # Now we can create generator instances
        gen1 = func()
        gen2 = func()

        # They should be different generator instances
        assert gen1 is not gen2
        assert list(gen1) == [1, 2, 3]
        assert list(gen2) == [1, 2, 3]


class TestStringUsage:
    """Test string-based decorator usage for user-friendliness."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clean registry before each test."""
        registry._components.clear()
        registry._protected_components.clear()
        yield
        registry._components.clear()
        registry._protected_components.clear()

    def test_string_component_type(self):
        """Test that component types can be strings."""

        @component("node", namespace="user")
        class StringNode:
            pass

        @component("tool", namespace="user")
        class StringTool:
            pass

        @component("adapter", namespace="user")
        class StringAdapter:
            pass

        # All should be registered with correct types
        assert registry.get_metadata("string_node", "user").component_type == ComponentType.NODE
        assert registry.get_metadata("string_tool", "user").component_type == ComponentType.TOOL
        assert (
            registry.get_metadata("string_adapter", "user").component_type == ComponentType.ADAPTER
        )

    def test_string_namespace(self):
        """Test that namespaces can be strings."""

        @node(namespace="my_plugin")
        class PluginNode:
            pass

        # Should be registered in plugin namespace
        metadata = registry.get_metadata("plugin_node", "my_plugin")
        assert metadata.namespace == "my_plugin"

    def test_string_subtype(self):
        """Test that subtypes can be strings."""

        @component("node", namespace="user", subtype="function")
        class FuncNode:
            pass

        @component("node", namespace="user", subtype="llm")
        class LLMNode:
            pass

        # Should have correct subtypes
        assert registry.get_metadata("func_node", "user").subtype == "function"
        assert registry.get_metadata("llm_node", "user").subtype == "llm"

    def test_mixed_string_and_enum(self):
        """Test mixing strings and enums works."""

        @component(ComponentType.NODE, namespace="user")  # Enum type, string namespace
        class MixedNode1:
            pass

        @component("node", namespace="user")  # String type, string namespace
        class MixedNode2:
            pass

        # Both should work
        assert registry.get_metadata("mixed_node1", "user").component_type == ComponentType.NODE
        assert registry.get_metadata("mixed_node2", "user").component_type == ComponentType.NODE

    def test_default_string_namespace(self):
        """Test that default namespace is 'user' string."""

        @node()  # No namespace specified, should default to "user"
        class DefaultNode:
            pass

        # Should be in user namespace
        metadata = registry.get_metadata("default_node", "user")
        assert metadata.namespace == "user"
        # Should NOT be protected (user components aren't protected)
        assert "default_node" not in registry._protected_components
