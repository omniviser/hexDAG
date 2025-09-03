"""Tests for registry.py - the main ComponentRegistry class."""

import threading
from unittest.mock import Mock

import pytest

from hexai.core.registry.metadata import ComponentMetadata
from hexai.core.registry.registry import ComponentRegistry
from hexai.core.registry.types import ComponentType


class TestComponentRegistrySingleton:
    """Test singleton pattern implementation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear singleton before each test."""
        ComponentRegistry._instance = None

    def test_singleton_instance(self):
        """Test that registry is a singleton."""
        registry1 = ComponentRegistry()
        registry2 = ComponentRegistry()
        assert registry1 is registry2

    def test_thread_safe_singleton(self):
        """Test thread-safe singleton creation."""
        instances = []

        def create_instance():
            instances.append(ComponentRegistry())

        threads = [threading.Thread(target=create_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(inst is instances[0] for inst in instances)


class TestNamespaceManagement:
    """Test namespace registration and protection."""

    @pytest.fixture
    def registry(self):
        """Provide clean registry."""
        ComponentRegistry._instance = None
        return ComponentRegistry()

    def test_protected_namespaces(self, registry):
        """Test that protected namespaces are defined."""
        assert "core" in registry._protected_namespaces
        assert "hexai" in registry._protected_namespaces
        assert "system" in registry._protected_namespaces
        assert "internal" in registry._protected_namespaces

    def test_cannot_register_protected_namespace(self, registry):
        """Test that protected namespaces cannot be registered."""
        with pytest.raises(ValueError, match="is protected"):
            registry.register_namespace("core")

    def test_register_custom_namespace(self, registry):
        """Test registering custom namespace."""
        registry.register_namespace("my_plugin")
        assert "my_plugin" in registry._namespaces

        # Idempotent
        registry.register_namespace("my_plugin")
        assert "my_plugin" in registry._namespaces


class TestComponentRegistration:
    """Test component registration and retrieval."""

    @pytest.fixture
    def registry(self):
        """Provide clean registry."""
        ComponentRegistry._instance = None
        return ComponentRegistry()

    def test_register_component(self, registry):
        """Test basic component registration."""
        component = Mock()
        registry.register(
            name="test_node",
            component=component,
            component_type=ComponentType.NODE,
            namespace="test",
        )

        retrieved = registry.get("test_node", ComponentType.NODE, "test")
        assert retrieved is component

    def test_component_replacement(self, registry):
        """Test component replacement rules."""
        # Register non-replaceable
        registry.register(
            name="protected",
            component=Mock(),
            component_type=ComponentType.NODE,
            namespace="test",
            replaceable=False,
        )

        # Cannot replace
        with pytest.raises(ValueError, match="cannot be replaced"):
            registry.register(
                name="protected",
                component=Mock(),
                component_type=ComponentType.NODE,
                namespace="test",
                replace=True,
            )

        # Register replaceable
        registry.register(
            name="flexible",
            component=Mock(),
            component_type=ComponentType.NODE,
            namespace="test",
            replaceable=True,
        )

        # Can replace
        new_component = Mock()
        registry.register(
            name="flexible",
            component=new_component,
            component_type=ComponentType.NODE,
            namespace="test",
            replace=True,
        )

        retrieved = registry.get("flexible", ComponentType.NODE, "test")
        assert retrieved is new_component


class TestDependencyResolution:
    """Test dependency tracking and resolution."""

    @pytest.fixture
    def registry(self):
        """Provide clean registry."""
        ComponentRegistry._instance = None
        return ComponentRegistry()

    def test_resolve_dependencies(self, registry):
        """Test dependency resolution."""
        # Create chain: A -> B -> C
        registry.register("C", Mock(), ComponentType.NODE, "test")
        registry.register(
            "B",
            Mock(),
            ComponentType.NODE,
            "test",
            metadata=ComponentMetadata(
                name="B", component_type=ComponentType.NODE, dependencies={"test:C"}
            ),
        )
        registry.register(
            "A",
            Mock(),
            ComponentType.NODE,
            "test",
            metadata=ComponentMetadata(
                name="A", component_type=ComponentType.NODE, dependencies={"test:B"}
            ),
        )

        order = registry.resolve_dependencies("test:A")
        assert order == ["test:C", "test:B", "test:A"]

    def test_circular_dependency_detection(self, registry):
        """Test circular dependency detection."""
        # Create cycle: A -> B -> A
        registry.register(
            "A",
            Mock(),
            ComponentType.NODE,
            "test",
            metadata=ComponentMetadata(
                name="A", component_type=ComponentType.NODE, dependencies={"test:B"}
            ),
        )
        registry.register(
            "B",
            Mock(),
            ComponentType.NODE,
            "test",
            metadata=ComponentMetadata(
                name="B", component_type=ComponentType.NODE, dependencies={"test:A"}
            ),
        )

        with pytest.raises(ValueError, match="Circular dependency"):
            registry.resolve_dependencies("test:A")


class TestLazyLoading:
    """Test lazy loading functionality."""

    @pytest.fixture
    def registry(self):
        """Provide registry with mock plugins."""
        ComponentRegistry._instance = None
        reg = ComponentRegistry()

        # Mock available plugins
        reg._available_plugins = {"plugin1": Mock(), "plugin2": Mock()}
        reg._loaded_plugins = set()

        return reg

    def test_plugin_not_loaded_initially(self, registry):
        """Test plugins are not loaded on discovery."""
        assert "plugin1" in registry._available_plugins
        assert "plugin1" not in registry._loaded_plugins
        registry._available_plugins["plugin1"].assert_not_called()

    def test_lazy_load_on_access(self, registry):
        """Test plugin loads when accessing its components."""
        registry.register_namespace("plugin1")
        registry._components[ComponentType.NODE]["plugin1"]["test"] = (
            ComponentMetadata(name="test", component_type=ComponentType.NODE),
            Mock(),
        )

        # Access component
        registry.get("test", ComponentType.NODE, "plugin1")

        # Plugin should be loaded
        registry._available_plugins["plugin1"].assert_called_once()
        assert "plugin1" in registry._loaded_plugins

    def test_list_available_plugins(self, registry):
        """Test listing plugin availability."""
        plugins = registry.list_available_plugins()
        assert plugins == {"plugin1": False, "plugin2": False}  # Not loaded  # Not loaded

        registry._loaded_plugins.add("plugin1")
        plugins = registry.list_available_plugins()
        assert plugins == {"plugin1": True, "plugin2": False}  # Loaded  # Not loaded


class TestFiltering:
    """Test component filtering and search."""

    @pytest.fixture
    def registry(self):
        """Provide registry with test components."""
        ComponentRegistry._instance = None
        reg = ComponentRegistry()

        # Register test components
        reg.register(
            "llm_node",
            Mock(),
            ComponentType.NODE,
            "test",
            metadata=ComponentMetadata(
                name="llm_node",
                component_type=ComponentType.NODE,
                tags={"llm", "ai"},
            ),
        )
        reg.register(
            "ml_tool",
            Mock(),
            ComponentType.TOOL,
            "test",
            metadata=ComponentMetadata(
                name="ml_tool",
                component_type=ComponentType.TOOL,
                tags={"ml", "ai"},
            ),
        )

        return reg

    def test_find_by_tags(self, registry):
        """Test finding by tags."""
        results = registry.find(tags__contains="ai")
        assert len(results) == 2
        assert "test:llm_node" in results
        assert "test:ml_tool" in results

    def test_find_by_author(self, registry):
        """Test finding by author."""
        results = registry.find(author="hexdag")
        assert len(results) >= 2  # Should find our test components


class TestHooks:
    """Test registration hooks."""

    @pytest.fixture
    def registry(self):
        """Provide clean registry."""
        ComponentRegistry._instance = None
        return ComponentRegistry()

    def test_pre_registration_hook(self, registry):
        """Test pre-registration hook."""
        called = []

        def pre_hook(name, component, metadata, namespace):
            called.append(name)
            if name == "invalid":
                raise ValueError("Invalid")

        registry.add_hook(pre_hook, "pre")

        # Valid registration
        registry.register("valid", Mock(), ComponentType.NODE, "test")
        assert "valid" in called

        # Invalid registration
        with pytest.raises(ValueError, match="Invalid"):
            registry.register("invalid", Mock(), ComponentType.NODE, "test")

    def test_post_registration_hook(self, registry):
        """Test post-registration hook."""
        registered = []

        def post_hook(name, component, metadata, namespace):
            registered.append(f"{namespace}:{name}")

        registry.add_hook(post_hook, "post")

        registry.register("component", Mock(), ComponentType.NODE, "test")
        assert "test:component" in registered
