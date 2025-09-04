"""Tests for the simplified component registry."""

import warnings

import pytest

from hexai.core.registry import adapter, component, node, registry, tool
from hexai.core.registry.exceptions import (
    ComponentAlreadyRegisteredError,
    InvalidComponentError,
    NamespacePermissionError,
)
from hexai.core.registry.types import ComponentType  # Internal for tests
from hexai.core.registry.types import Namespace


class TestComponentRegistry:
    """Test the main ComponentRegistry class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clean registry before each test."""
        # Clear registry before each test
        registry._components.clear()
        registry._protected_components.clear()
        registry._ready = False

        yield

        # Clean up after test
        registry._components.clear()
        registry._protected_components.clear()
        registry._ready = False

    def test_register_and_get(self):
        """Test basic register and get functionality."""

        class TestComponent:
            def __init__(self, value=42):
                self.value = value

        registry.register(
            name="test_comp",
            component=TestComponent,
            component_type=ComponentType.NODE,
            namespace="test",
        )

        # Get component
        instance = registry.get("test_comp", namespace="test")
        assert isinstance(instance, TestComponent)
        assert instance.value == 42

        # Get with custom args
        instance = registry.get("test_comp", namespace="test", value=100)
        assert instance.value == 100

    def test_namespace_searching(self):
        """Test that get() searches namespaces correctly."""

        class CoreNode:
            pass

        class UserNode:
            pass

        # Register in different namespaces
        registry.register(
            "my_node", CoreNode, ComponentType.NODE, namespace=Namespace.CORE, privileged=True
        )
        registry.register("my_node", UserNode, ComponentType.NODE, namespace=Namespace.USER)

        # Without namespace, should get core first
        instance = registry.get("my_node")
        assert isinstance(instance, CoreNode)

        # With explicit namespace
        instance = registry.get("my_node", namespace=Namespace.USER)
        assert isinstance(instance, UserNode)

    def test_list_components(self):
        """Test listing components with filters."""

        class Node1:
            pass

        class Tool1:
            pass

        registry.register("node1", Node1, ComponentType.NODE, namespace="test")
        registry.register("tool1", Tool1, ComponentType.TOOL, namespace="test")
        registry.register("node2", Node1, ComponentType.NODE, namespace="other")

        # List all
        all_components = registry.list_components()
        assert len(all_components) == 3

        # Filter by type
        nodes = registry.list_components(component_type=ComponentType.NODE)
        assert len(nodes) == 2

        # Filter by namespace
        test_components = registry.list_components(namespace="test")
        assert len(test_components) == 2

    def test_component_replacement(self):
        """Test component replacement behavior."""

        class Original:
            pass

        class Replacement:
            pass

        registry.register("comp", Original, ComponentType.NODE, namespace="test")

        # Should fail without replace=True
        with pytest.raises(ComponentAlreadyRegisteredError):
            registry.register("comp", Replacement, ComponentType.NODE, namespace="test")

        # Should work with replace=True
        registry.register("comp", Replacement, ComponentType.NODE, namespace="test", replace=True)
        instance = registry.get("comp", namespace="test")
        assert isinstance(instance, Replacement)

    def test_core_protection(self):
        """Test that core namespace is protected."""

        class MyNode:
            pass

        # Should fail without privilege
        with pytest.raises(NamespacePermissionError):
            registry.register("my_node", MyNode, ComponentType.NODE, namespace=Namespace.CORE)

        # Should work with privilege
        registry.register(
            "my_node", MyNode, ComponentType.NODE, namespace=Namespace.CORE, privileged=True
        )
        assert "my_node" in registry._protected_components

    def test_ready_state(self):
        """Test registry ready state management."""
        assert registry.is_ready() is False
        registry.set_ready(True)
        assert registry.is_ready() is True
        registry.set_ready(False)
        assert registry.is_ready() is False


class TestDecorators:
    """Test decorator functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clean registry before each test."""
        registry._components.clear()
        registry._protected_components.clear()

        yield

        registry._components.clear()
        registry._protected_components.clear()

    def test_component_decorator(self):
        """Test the general component decorator."""

        @component(ComponentType.NODE, namespace="test")
        class TestNode:
            """A test node."""

            pass

        # Should be registered immediately
        metadata = registry.get_metadata("test_node", namespace="test")
        assert metadata.name == "test_node"
        assert metadata.component_type == ComponentType.NODE
        assert metadata.description == "A test node."

    def test_type_specific_decorators(self):
        """Test type-specific decorators."""

        @node(namespace="test")
        class TestNode:
            pass

        @tool(namespace="test")
        class TestTool:
            pass

        @adapter(namespace="test")
        class TestAdapter:
            pass

        # All should be registered with correct types
        assert registry.get_metadata("test_node", "test").component_type == ComponentType.NODE
        assert registry.get_metadata("test_tool", "test").component_type == ComponentType.TOOL
        assert registry.get_metadata("test_adapter", "test").component_type == ComponentType.ADAPTER

    def test_decorator_with_custom_name(self):
        """Test decorator with custom name."""

        @node(name="custom_name", namespace="test")
        class SomeClass:
            pass

        # Should use custom name
        metadata = registry.get_metadata("custom_name", namespace="test")
        assert metadata.name == "custom_name"

    def test_core_namespace_decorator(self):
        """Test decorator with core namespace."""

        @node(namespace=Namespace.CORE)
        class CoreNode:
            pass

        # Should be registered and protected
        assert "core_node" in registry._protected_components
        metadata = registry.get_metadata("core_node", namespace=Namespace.CORE)
        assert metadata.namespace == Namespace.CORE

    def test_valid_names(self):
        """Test that valid namespace and component names are accepted."""

        class TestComponent:
            pass

        # Valid alphanumeric namespace and component names
        registry.register("test123", TestComponent, "node", namespace="plugin123")
        instance = registry.get("test123", namespace="plugin123")
        assert isinstance(instance, TestComponent)

        # Underscores are allowed in both namespace and component names
        registry.register("test_component", TestComponent, "node", namespace="my_plugin")
        instance = registry.get("test_component", namespace="my_plugin")
        assert isinstance(instance, TestComponent)

        # Mixed case is allowed
        registry.register("MyComponent", TestComponent, "node", namespace="MyPlugin")
        instance = registry.get("MyComponent", namespace="MyPlugin")
        assert isinstance(instance, TestComponent)

    def test_invalid_namespace_names(self):
        """Test that invalid namespace names are rejected."""

        class TestComponent:
            pass

        # Namespace with hyphen should fail
        with pytest.raises(InvalidComponentError) as exc_info:
            registry.register("comp", TestComponent, "node", namespace="my-plugin")
        assert "must only contain letters, numbers, and underscores" in str(exc_info.value)

        # Namespace with space should fail
        with pytest.raises(InvalidComponentError) as exc_info:
            registry.register("comp", TestComponent, "node", namespace="my plugin")
        assert "must only contain letters, numbers, and underscores" in str(exc_info.value)

        # Namespace with dot should fail
        with pytest.raises(InvalidComponentError) as exc_info:
            registry.register("comp", TestComponent, "node", namespace="my.plugin")
        assert "must only contain letters, numbers, and underscores" in str(exc_info.value)

        # Namespace with special characters should fail
        with pytest.raises(InvalidComponentError) as exc_info:
            registry.register("comp", TestComponent, "node", namespace="my@plugin")
        assert "must only contain letters, numbers, and underscores" in str(exc_info.value)

    def test_invalid_component_names(self):
        """Test that invalid component names are rejected."""

        class TestComponent:
            pass

        # Component name with hyphen should fail
        with pytest.raises(InvalidComponentError) as exc_info:
            registry.register("my-component", TestComponent, "node", namespace="test")
        assert "must only contain letters, numbers, and underscores" in str(exc_info.value)

        # Component name with space should fail
        with pytest.raises(InvalidComponentError) as exc_info:
            registry.register("my component", TestComponent, "node", namespace="test")
        assert "must only contain letters, numbers, and underscores" in str(exc_info.value)

        # Component name with dot should fail
        with pytest.raises(InvalidComponentError) as exc_info:
            registry.register("my.component", TestComponent, "node", namespace="test")
        assert "must only contain letters, numbers, and underscores" in str(exc_info.value)

        # Component name with special character should fail
        with pytest.raises(InvalidComponentError) as exc_info:
            registry.register("my@component", TestComponent, "node", namespace="test")
        assert "must only contain letters, numbers, and underscores" in str(exc_info.value)

        # Empty component name should fail
        with pytest.raises(InvalidComponentError) as exc_info:
            registry.register("", TestComponent, "node", namespace="test")
        assert "must be a non-empty string" in str(exc_info.value)


class TestPluginShadowing:
    """Test plugin component shadowing behavior."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clean registry and set up core component."""
        registry._components.clear()
        registry._protected_components.clear()

        # Register a core component
        class CoreNode:
            pass

        registry.register(
            "processor", CoreNode, ComponentType.NODE, namespace=Namespace.CORE, privileged=True
        )

        yield

        registry._components.clear()
        registry._protected_components.clear()

    def test_plugin_shadow_warning(self):
        """Test that plugins shadowing core components generate warning."""

        class PluginNode:
            pass

        # Should generate warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            registry.register("processor", PluginNode, ComponentType.NODE, namespace="my_plugin")
            assert len(w) == 1
            assert "shadows HEXDAG CORE component" in str(w[0].message)

    def test_shadowed_core_remains_accessible(self):
        """Test that shadowed core components remain accessible."""

        class PluginNode:
            def __init__(self):
                self.is_plugin = True

        # Register plugin version
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            registry.register("processor", PluginNode, ComponentType.NODE, namespace="my_plugin")

        # Core version should still be accessible
        core_instance = registry.get("processor", namespace=Namespace.CORE)
        assert not hasattr(core_instance, "is_plugin")

        # Plugin version accessible with explicit namespace
        plugin_instance = registry.get("processor", namespace="my_plugin")
        assert plugin_instance.is_plugin is True

        # Default should still get core
        default_instance = registry.get("processor")
        assert not hasattr(default_instance, "is_plugin")
