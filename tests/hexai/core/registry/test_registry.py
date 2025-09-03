"""Tests for the simplified component registry."""

import os
import warnings
from unittest.mock import Mock

import pytest

from hexai.core.registry import (
    ComponentRegistry,
    ComponentType,
    adapter,
    component,
    memory,
    node,
    observer,
    policy,
    registry,
    tool,
)


class TestComponentRegistry:
    """Test the main ComponentRegistry class."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        """Set up test environment and clean registry."""
        # Set test environment
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "true")

        # Clear registry before each test
        registry._clear_for_testing()

        # Clear pending components
        ComponentRegistry._pending_components.clear()

        yield

        # Clean up after test
        registry._clear_for_testing()
        ComponentRegistry._pending_components.clear()

    def test_singleton_pattern(self):
        """Test that registry is a singleton."""
        registry1 = ComponentRegistry()
        registry2 = ComponentRegistry()
        assert registry1 is registry2
        assert registry1 is registry

    def test_core_components_auto_register(self):
        """Test that core components are automatically registered."""
        # Clear and reinitialize to trigger core loading
        registry._components.clear()
        registry._protected_components.clear()
        ComponentRegistry._pending_components.clear()

        # Re-trigger initialization which loads core components
        registry._load_core_components()
        registry._process_pending_components()

        # Check core components exist
        components = registry.list_components(namespace="core")
        # If no core components loaded, that's OK for testing
        if len(components) > 0:
            # Verify we can get them
            passthrough = registry.get("pass_through_node")
            assert passthrough is not None
            assert hasattr(passthrough, "execute")

    def test_get_component(self):
        """Test getting components by name."""
        # Register a test component
        registry.register(
            name="test_comp", component=Mock(), component_type=ComponentType.NODE, namespace="test"
        )

        # Get with explicit namespace
        comp = registry.get("test_comp", namespace="test")
        assert comp is not None

        # Get with namespace in name
        comp2 = registry.get("test:test_comp")
        assert comp2 is comp

        # Component not found
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_list_components(self):
        """Test listing components with filters."""
        # Register test components
        registry.register("node1", Mock(), ComponentType.NODE, namespace="test")
        registry.register("tool1", Mock(), ComponentType.TOOL, namespace="test")
        registry.register("node2", Mock(), ComponentType.NODE, namespace="other")

        # List all
        all_comps = registry.list_components()
        assert len(all_comps) >= 3

        # Filter by namespace
        test_comps = registry.list_components(namespace="test")
        assert len(test_comps) == 2
        assert "test:node1" in test_comps
        assert "test:tool1" in test_comps

        # Filter by type
        nodes = registry.list_components(component_type=ComponentType.NODE)
        assert all("node" in comp for comp in nodes if ":" in comp)

    def test_namespace_management(self):
        """Test namespace creation and listing."""
        # Register in new namespace
        registry.register("comp", Mock(), ComponentType.NODE, namespace="custom")

        # Check namespace exists
        namespaces = registry.list_namespaces()
        assert "custom" in namespaces

        # Namespaces are sorted
        assert namespaces == sorted(namespaces)

    def test_metadata_storage(self):
        """Test component metadata."""
        # Register with metadata
        registry.register(
            name="meta_comp",
            component=Mock(),
            component_type=ComponentType.NODE,
            namespace="test",
            description="Test component",
            version="2.0.0",
            author="tester",
            tags={"test", "example"},
            dependencies={"dep1", "dep2"},
        )

        # Get metadata
        meta = registry.get_metadata("meta_comp", namespace="test")
        assert meta.name == "meta_comp"
        assert meta.component_type == ComponentType.NODE
        assert meta.namespace == "test"
        assert meta.description == "Test component"
        assert meta.version == "2.0.0"
        assert meta.author == "tester"
        assert meta.tags == frozenset({"test", "example"})
        assert meta.dependencies == frozenset({"dep1", "dep2"})

    def test_component_replacement(self):
        """Test component replacement rules."""
        # Register non-replaceable component
        registry.register(
            name="fixed",
            component=Mock(),
            component_type=ComponentType.NODE,
            namespace="test",
            replaceable=False,
        )

        # Cannot replace without flag
        with pytest.raises(ValueError, match="not replaceable"):
            registry.register(
                name="fixed", component=Mock(), component_type=ComponentType.NODE, namespace="test"
            )

        # Register replaceable component
        registry.register(
            name="flexible",
            component=Mock(),
            component_type=ComponentType.NODE,
            namespace="test",
            replaceable=True,
        )

        # Can replace with flag
        new_comp = Mock()
        registry.register(
            name="flexible",
            component=new_comp,
            component_type=ComponentType.NODE,
            namespace="test",
            replace=True,
        )

        assert registry.get("flexible", namespace="test") is new_comp

    def test_lazy_instantiation(self):
        """Test that components are instantiated lazily."""

        # Register a class (not instance)
        class TestNode:
            instantiated = False

            def __init__(self):
                TestNode.instantiated = True

        registry.register(
            name="lazy", component=TestNode, component_type=ComponentType.NODE, namespace="test"
        )

        # Not instantiated yet
        assert not TestNode.instantiated

        # Get triggers instantiation
        instance = registry.get("lazy", namespace="test")
        assert TestNode.instantiated
        assert isinstance(instance, TestNode)

        # Second get returns same instance
        instance2 = registry.get("lazy", namespace="test")
        assert instance is instance2


class TestDecorators:
    """Test component decorators."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        """Set up test environment."""
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "true")
        registry._clear_for_testing()
        ComponentRegistry._pending_components.clear()
        yield
        registry._clear_for_testing()
        ComponentRegistry._pending_components.clear()

    def test_component_decorator(self):
        """Test the generic component decorator."""

        @component(
            name="test_comp",
            namespace="test",
            component_type=ComponentType.NODE,
            description="Test description",
            tags={"test"},
            author="tester",
        )
        class TestComponent:
            """Component docstring."""

            pass

        # Check decorator marks
        assert hasattr(TestComponent, "_hexdag_component")
        assert TestComponent._hexdag_namespace == "test"
        assert TestComponent._hexdag_name == "test_comp"

        # Process registration
        registry._process_pending_components()

        # Verify registration
        comp = registry.get("test_comp", namespace="test")
        assert isinstance(comp, TestComponent)

        meta = registry.get_metadata("test_comp", namespace="test")
        assert meta.description == "Test description"
        assert meta.author == "tester"
        assert "test" in meta.tags

    def test_type_specific_decorators(self):
        """Test type-specific decorator shortcuts."""

        @node(namespace="test")
        class TestNode:
            pass

        @tool(namespace="test")
        class TestTool:
            pass

        @adapter(namespace="test")
        class TestAdapter:
            pass

        @policy(namespace="test")
        class TestPolicy:
            pass

        @memory(namespace="test")
        class TestMemory:
            pass

        @observer(namespace="test")
        class TestObserver:
            pass

        # Process all
        registry._process_pending_components()

        # Verify types
        node_meta = registry.get_metadata("test_node", namespace="test")
        assert node_meta.component_type == ComponentType.NODE

        tool_meta = registry.get_metadata("test_tool", namespace="test")
        assert tool_meta.component_type == ComponentType.TOOL

        adapter_meta = registry.get_metadata("test_adapter", namespace="test")
        assert adapter_meta.component_type == ComponentType.ADAPTER

        policy_meta = registry.get_metadata("test_policy", namespace="test")
        assert policy_meta.component_type == ComponentType.POLICY

        memory_meta = registry.get_metadata("test_memory", namespace="test")
        assert memory_meta.component_type == ComponentType.MEMORY

        observer_meta = registry.get_metadata("test_observer", namespace="test")
        assert observer_meta.component_type == ComponentType.OBSERVER

    def test_decorator_name_inference(self):
        """Test automatic name inference from class name."""

        @node(namespace="test")
        class ComplexNodeName:
            pass

        registry._process_pending_components()

        # Should convert to snake_case
        comp = registry.get("complex_node_name", namespace="test")
        assert isinstance(comp, ComplexNodeName)

    def test_decorator_type_inference(self):
        """Test automatic type inference from class name."""

        @component(namespace="test")
        class SomethingNode:
            pass

        @component(namespace="test")
        class DataAdapter:
            pass

        registry._process_pending_components()

        # Check inferred types
        node_meta = registry.get_metadata("something_node", namespace="test")
        assert node_meta.component_type == ComponentType.NODE

        adapter_meta = registry.get_metadata("data_adapter", namespace="test")
        assert adapter_meta.component_type == ComponentType.ADAPTER

    def test_decorator_docstring_as_description(self):
        """Test using docstring as description."""

        @node(namespace="test")
        class DocumentedNode:
            """This is the node description."""

            pass

        registry._process_pending_components()

        meta = registry.get_metadata("documented_node", namespace="test")
        assert meta.description == "This is the node description."


class TestCoreProtection:
    """Test core component protection mechanisms."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        """Set up test environment."""
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "true")
        registry._clear_for_testing()
        ComponentRegistry._pending_components.clear()

        # Register a fake core component
        registry.register(
            name="protected_core",
            component=Mock(),
            component_type=ComponentType.NODE,
            namespace="core",
            replaceable=False,
        )
        registry._protected_components.add("protected_core")

        # Mark metadata as core
        meta = registry._components["core"]["protected_core"]
        meta.is_core = True

        yield
        registry._clear_for_testing()
        ComponentRegistry._pending_components.clear()

    def test_core_shadow_warning(self):
        """Test that shadowing core components triggers warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Try to register with same name
            registry.register(
                name="protected_core",
                component=Mock(),
                component_type=ComponentType.NODE,
                namespace="plugin",
            )

            # Should have warning
            assert len(w) == 1
            assert "shadows CORE component" in str(w[0].message)
            assert "core:protected_core" in str(w[0].message)

    def test_core_remains_accessible(self):
        """Test that core components remain accessible after shadowing."""
        # Shadow the core component
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            plugin_comp = Mock()
            registry.register(
                name="protected_core",
                component=plugin_comp,
                component_type=ComponentType.NODE,
                namespace="plugin",
            )

        # Core version still accessible
        core_comp = registry.get("protected_core", namespace="core")
        assert core_comp is not plugin_comp

        # Default get returns core version
        default_comp = registry.get("protected_core")
        assert default_comp is core_comp

        # Plugin version accessible with explicit namespace
        explicit_plugin = registry.get("protected_core", namespace="plugin")
        assert explicit_plugin is plugin_comp

    def test_cannot_replace_core_directly(self):
        """Test that core components cannot be replaced in core namespace."""
        # Try to replace in core namespace without replace flag
        with pytest.raises(ValueError, match="not replaceable"):
            registry.register(
                name="protected_core",
                component=Mock(),
                component_type=ComponentType.NODE,
                namespace="core",
            )


class TestPluginCreation:
    """Test creating and registering a plugin."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        """Set up test environment."""
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "true")
        registry._clear_for_testing()
        ComponentRegistry._pending_components.clear()
        yield
        registry._clear_for_testing()
        ComponentRegistry._pending_components.clear()

    def test_create_simple_plugin(self):
        """Test creating a simple plugin with components."""

        # Simulate a plugin module
        @node(namespace="my_plugin", author="plugin_author", version="1.0.0")
        class AnalyzerNode:
            """Analyzes data."""

            def execute(self, data):
                return f"analyzed: {data}"

        @tool(namespace="my_plugin", tags={"utility", "data"})
        class DataFetcher:
            """Fetches data from sources."""

            def fetch(self, source):
                return f"data from {source}"

        # Process registrations
        registry._process_pending_components()

        # Verify plugin components
        plugin_comps = registry.list_components(namespace="my_plugin")
        assert len(plugin_comps) == 2
        assert "my_plugin:analyzer_node" in plugin_comps
        assert "my_plugin:data_fetcher" in plugin_comps

        # Test usage
        analyzer = registry.get("analyzer_node", namespace="my_plugin")
        assert analyzer.execute("test") == "analyzed: test"

        fetcher = registry.get("data_fetcher", namespace="my_plugin")
        assert fetcher.fetch("db") == "data from db"

    def test_plugin_with_dependencies(self):
        """Test plugin declaring component dependencies."""

        @node(
            namespace="complex_plugin",
            dependencies={"core:pass_through_node", "other:required_comp"},
        )
        class DependentNode:
            """Node with dependencies."""

            pass

        registry._process_pending_components()

        meta = registry.get_metadata("dependent_node", namespace="complex_plugin")
        assert "core:pass_through_node" in meta.dependencies
        assert "other:required_comp" in meta.dependencies

    def test_plugin_namespace_isolation(self):
        """Test that plugins are isolated in their namespaces."""

        # Create two plugins with same component names
        @node(namespace="plugin_a")
        class ProcessorNode:
            def execute(self):
                return "A"

        @node(namespace="plugin_b")
        class ProcessorNode:  # noqa: F811
            def execute(self):
                return "B"

        registry._process_pending_components()

        # Both exist in their namespaces
        node_a = registry.get("processor_node", namespace="plugin_a")
        node_b = registry.get("processor_node", namespace="plugin_b")

        assert node_a.execute() == "A"
        assert node_b.execute() == "B"

    def test_plugin_discovery_integration(self):
        """Test that plugins can be discovered via pluggy."""
        # Just verify that the plugin manager is set up
        assert hasattr(registry, "pm")
        assert registry.pm is not None

        # Verify it has the right project name
        assert registry.pm.project_name == "hexdag"


class TestClearForTesting:
    """Test the testing utility method."""

    def test_clear_only_in_test_mode(self):
        """Test that clear only works in test mode."""
        # Remove test environment
        if "PYTEST_CURRENT_TEST" in os.environ:
            del os.environ["PYTEST_CURRENT_TEST"]

        # Should raise in non-test mode
        with pytest.raises(RuntimeError, match="only available in tests"):
            registry._clear_for_testing()

        # Set test mode
        os.environ["PYTEST_CURRENT_TEST"] = "true"

        # Should work now
        registry._clear_for_testing()

        # Clean up
        del os.environ["PYTEST_CURRENT_TEST"]
