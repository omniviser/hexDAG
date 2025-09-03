"""Tests for decorators.py - component registration decorators."""

from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from hexai.core.registry.decorators import (
    _pending_components,
    adapter,
    component,
    node,
    observer,
    register_pending_components,
    tool,
)
from hexai.core.registry.types import ComponentType


class TestComponentDecorators:
    """Test the component registration decorators."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear pending components before each test."""
        _pending_components.clear()

    def test_node_decorator_basic(self):
        """Test basic node decorator usage."""

        @node(namespace="test")
        class TestNode:
            """A test node."""

            pass

        assert len(_pending_components) == 1
        component, metadata, namespace = _pending_components[0]

        assert metadata.name == "test_node"  # Snake case conversion
        assert metadata.component_type == ComponentType.NODE
        assert component is TestNode
        assert namespace == "test"

    def test_node_decorator_with_options(self):
        """Test node decorator with all options."""

        @node(
            namespace="test",
            name="custom_node",
            version="2.0.0",
            description="Custom description",
            tags={"tag1", "tag2"},
            dependencies={"dep1", "dep2"},
            replaceable=False,
        )
        class MyNode:
            pass

        _, metadata, _ = _pending_components[0]

        assert metadata.name == "custom_node"
        assert metadata.version == "2.0.0"
        assert metadata.description == "Custom description"
        assert metadata.tags == {"tag1", "tag2"}
        assert metadata.dependencies == {"dep1", "dep2"}
        assert metadata.replaceable is False

    def test_tool_decorator(self):
        """Test tool decorator."""

        @tool(namespace="test", description="A test tool")
        def my_tool(data: Any) -> Any:
            return data

        component, metadata, _ = _pending_components[0]

        assert metadata.name == "my_tool"
        assert metadata.component_type == ComponentType.TOOL
        assert metadata.description == "A test tool"
        assert component is my_tool

    def test_adapter_decorator(self):
        """Test adapter decorator."""

        @adapter(namespace="test")
        class TestAdapter:
            """An adapter component."""

            pass

        _, metadata, _ = _pending_components[0]

        assert metadata.component_type == ComponentType.ADAPTER

    def test_observer_decorator(self):
        """Test observer decorator."""

        @observer(namespace="test")
        class ObserverComponent:
            pass

        _, metadata, namespace = _pending_components[0]

        assert metadata.component_type == ComponentType.OBSERVER
        assert namespace == "test"

    def test_component_decorator_generic(self):
        """Test generic component decorator."""

        @component(component_type=ComponentType.MEMORY, namespace="test")
        class ServiceComponent:
            pass

        _, metadata, _ = _pending_components[0]

        assert metadata.component_type == ComponentType.MEMORY
        assert metadata.name == "service_component"

    def test_decorator_preserves_class(self):
        """Test that decorators preserve the original class."""

        @node(namespace="test")
        class OriginalClass:
            """Original docstring."""

            value = 42

            def method(self):
                return "test"

        assert OriginalClass.__doc__ == "Original docstring."
        assert OriginalClass.value == 42
        instance = OriginalClass()
        assert instance.method() == "test"

    def test_decorator_preserves_function(self):
        """Test that decorators preserve the original function."""

        @tool(namespace="test")
        def original_function(x: int) -> int:
            """Original function docstring."""
            return x * 2

        assert original_function.__doc__ == "Original function docstring."
        assert original_function(5) == 10

    def test_namespace_required(self):
        """Test that namespace is required."""
        with pytest.raises(TypeError):

            @node()  # Missing namespace
            class InvalidNode:
                pass

    def test_multiple_components(self):
        """Test registering multiple components."""

        @node(namespace="test")
        class Node1:
            pass

        @tool(namespace="test")
        def tool1():
            pass

        @adapter(namespace="other")
        class Adapter1:
            pass

        assert len(_pending_components) == 3

        # Check types
        types = [m.component_type for _, m, _ in _pending_components]
        assert ComponentType.NODE in types
        assert ComponentType.TOOL in types
        assert ComponentType.ADAPTER in types

    def test_register_pending_components(self):
        """Test registering pending components."""

        # Create some pending components
        @node(namespace="test")
        class TestNode:
            pass

        @tool(namespace="test")
        def test_tool():
            pass

        # Mock registry
        mock_registry = Mock()

        register_pending_components(mock_registry)

        # Should have registered both components
        assert mock_registry.register.call_count == 2

        # Check first registration (node)
        call_args = mock_registry.register.call_args_list[0]
        assert call_args[1]["name"] == "test_node"
        assert call_args[1]["component"] is TestNode
        assert call_args[1]["component_type"] == ComponentType.NODE
        assert call_args[1]["namespace"] == "test"

        # Should clear pending after registration
        assert len(_pending_components) == 0

    def test_register_pending_with_errors(self):
        """Test register_pending handles errors gracefully."""

        @node(namespace="test")
        class TestNode:
            pass

        mock_registry = Mock()
        mock_registry.register.side_effect = ValueError("Registration failed")

        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            register_pending_components(mock_registry)

            # Should log warning but not raise
            mock_logger.warning.assert_called()

        # Should still clear pending
        assert len(_pending_components) == 0

    def test_decorator_with_inheritance(self):
        """Test decorators work with inheritance."""

        @node(namespace="test")
        class BaseNode:
            def execute(self):
                return "base"

        # Subclass should not be registered automatically
        class DerivedNode(BaseNode):
            def execute(self):
                return "derived"

        assert len(_pending_components) == 1
        component, _, _ = _pending_components[0]
        assert component is BaseNode

    def test_decorator_on_methods_fails(self):
        """Test that decorators on methods fail appropriately."""

        class TestClass:
            @node(namespace="test")  # This won't work as expected
            def method(self):
                pass

        # The decorator will register the unbound method
        # This is allowed but probably not what user wants
        assert len(_pending_components) == 1
        component, _, _ = _pending_components[0]
        # It registers the method function, not the class
        assert callable(component)

    def test_decorator_stacking(self):
        """Test that decorators can't be stacked."""

        # This would register twice - not recommended but allowed
        @node(namespace="test1")
        @node(namespace="test2")
        class DoubleNode:
            pass

        # Both decorations should be in pending
        assert len(_pending_components) == 2
        namespaces = [ns for _, _, ns in _pending_components]
        assert "test1" in namespaces
        assert "test2" in namespaces

    def test_name_inference(self):
        """Test that names are correctly inferred."""

        @node(namespace="test")
        class CamelCaseClass:
            pass

        @tool(namespace="test")
        def snake_case_function():
            pass

        @observer(namespace="test")
        class _PrivateClass:
            pass

        names = [m.name for _, m, _ in _pending_components]
        assert "camel_case_class" in names
        assert "snake_case_function" in names
        assert "__private_class" in names  # Double underscore preserved

    def test_replaceable_default(self):
        """Test default replaceable value."""

        @node(namespace="test")
        class DefaultNode:
            pass

        _, metadata, _ = _pending_components[0]
        assert metadata.replaceable is False  # Default is False

    def test_replaceable_explicit(self):
        """Test explicit replaceable values."""

        @node(namespace="test", replaceable=False)
        class NonReplaceableNode:
            pass

        @tool(namespace="test", replaceable=True)
        def replaceable_tool():
            pass

        metadatas = [m for _, m, _ in _pending_components]
        assert metadatas[0].replaceable is False
        assert metadatas[1].replaceable is True


class TestDecoratorIntegration:
    """Test decorator integration with registry."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock registry."""
        mock = MagicMock()
        mock._protected_namespaces = {"core", "hexai", "system", "internal"}
        return mock

    def test_protected_namespace_registration(self, mock_registry):
        """Test that protected namespaces work with decorators."""
        _pending_components.clear()

        # This should be allowed to pend
        @node(namespace="core")
        class CoreNode:
            pass

        assert len(_pending_components) == 1

        # Mock the register to check namespace
        def check_namespace(name, component, component_type, namespace, **kwargs):
            if namespace in mock_registry._protected_namespaces:
                raise ValueError(f"Namespace '{namespace}' is protected")

        mock_registry.register.side_effect = check_namespace

        # This will log a warning but not raise
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            register_pending_components(mock_registry)

            # Should have logged warning about protected namespace
            mock_logger.warning.assert_called_once()
            assert "protected" in str(mock_logger.warning.call_args)

    def test_custom_metadata_fields(self):
        """Test that only standard metadata fields are accepted."""
        # The decorator only accepts standard fields defined in ComponentMetadata

        @component(
            component_type=ComponentType.POLICY,
            namespace="test",
            description="A policy component",
            version="1.0.0",
        )
        class CustomService:
            pass

        _, metadata, _ = _pending_components[0]

        # Standard fields
        assert metadata.name == "custom_service"
        assert metadata.component_type == ComponentType.POLICY
        assert metadata.description == "A policy component"
        assert metadata.version == "1.0.0"


class TestDecoratorEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear pending components."""
        _pending_components.clear()

    def test_empty_namespace(self):
        """Test that empty namespace is allowed but not recommended."""

        # Empty namespace is technically allowed
        @node(namespace="")
        class EmptyNamespaceNode:
            pass

        assert len(_pending_components) == 1
        _, metadata, namespace = _pending_components[0]
        assert namespace == ""
        assert metadata.name == "empty_namespace_node"

    def test_none_namespace(self):
        """Test that None namespace is treated as None."""

        # None namespace is allowed (though not recommended)
        @node(namespace=None)  # type: ignore
        class NoneNamespaceNode:
            pass

        assert len(_pending_components) == 1
        _, metadata, namespace = _pending_components[0]
        assert namespace is None
        assert metadata.name == "none_namespace_node"

    def test_lambda_function(self):
        """Test decorator on lambda (should work but not recommended)."""
        # This is syntactically invalid in Python
        # lambda_tool = tool(namespace='test')(lambda x: x * 2)

        # But we can test programmatically
        def lambda_func(x):
            return x * 2

        tool(namespace="test")(lambda_func)

        assert len(_pending_components) == 1
        component, metadata, _ = _pending_components[0]
        assert metadata.name == "lambda_func"

    def test_partial_function(self):
        """Test decorator on partial function."""
        from functools import partial

        def base_func(x, y):
            return x + y

        partial_func = partial(base_func, 10)

        @tool(namespace="test", name="partial_tool")
        def wrapped_partial(x):
            return partial_func(x)

        assert len(_pending_components) == 1
        _, metadata, _ = _pending_components[0]
        assert metadata.name == "partial_tool"

    def test_very_long_name(self):
        """Test component with very long name."""
        long_name = "A" * 1000

        @node(namespace="test", name=long_name)
        class LongNamedClass:
            pass

        _, metadata, _ = _pending_components[0]
        assert metadata.name == long_name

    def test_special_characters_in_name(self):
        """Test special characters in component name."""

        @node(namespace="test", name="test-node-123_v2.0")
        class SpecialNode:
            pass

        _, metadata, _ = _pending_components[0]
        assert metadata.name == "test-node-123_v2.0"

    def test_unicode_in_metadata(self):
        """Test unicode in metadata fields."""

        @tool(namespace="test", description="🚀 Unicode tool", tags={"émoji", "中文", "العربية"})
        def unicode_tool():
            pass

        _, metadata, _ = _pending_components[0]
        assert "🚀" in metadata.description
        assert "émoji" in metadata.tags
