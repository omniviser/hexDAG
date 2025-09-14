"""Tests for the simplified component registry."""

import warnings

import pytest

from hexai.core.registry import adapter, component, node, port, registry, tool
from hexai.core.registry.registry import ComponentRegistry
from hexai.core.registry.exceptions import (
    ComponentAlreadyRegisteredError,
    ComponentNotFoundError,
    InvalidComponentError,
    NamespacePermissionError,
)
from hexai.core.registry.models import ComponentType  # Internal for tests


class TestComponentRegistry:
    """Test the main ComponentRegistry class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clean registry before each test."""
        # Clear registry before each test
        registry._components.clear()
        registry._protected_components.clear()
        registry._ready = False
        registry._manifest = None
        registry._dev_mode = False
        registry._bootstrap_context = False

        yield

        # Clean up after test
        registry._components.clear()
        registry._protected_components.clear()
        registry._ready = False
        registry._manifest = None
        registry._dev_mode = False
        registry._bootstrap_context = False

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
        instance = registry.get("test_comp", namespace="test", init_params={"value": 100})
        assert instance.value == 100

    def test_namespace_searching(self):
        """Test that get() searches namespaces correctly."""

        class CoreNode:
            pass

        class UserNode:
            pass

        # Register in different namespaces
        registry.register(
            "my_node", CoreNode, ComponentType.NODE, namespace="core", privileged=True
        )
        registry.register("my_node", UserNode, ComponentType.NODE, namespace="user")

        # Without namespace, should get core first
        instance = registry.get("my_node")
        assert isinstance(instance, CoreNode)

        # With explicit namespace
        instance = registry.get("my_node", namespace="user")
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

        # Should always fail on duplicate (no replacement policy)
        with pytest.raises(ComponentAlreadyRegisteredError):
            registry.register("comp", Replacement, ComponentType.NODE, namespace="test")

    def test_core_protection(self):
        """Test that core namespace is protected."""

        class MyNode:
            pass

        # Should fail without privilege
        with pytest.raises(NamespacePermissionError):
            registry.register("my_node", MyNode, ComponentType.NODE, namespace="core")

        # Should work with privilege
        registry.register("my_node", MyNode, ComponentType.NODE, namespace="core", privileged=True)
        assert "core:my_node" in registry._protected_components


class TestDecorators:
    """Test decorator functionality - decorators only add metadata, no auto-registration."""

    def test_component_decorator_adds_metadata(self):
        """Test that decorators add metadata to classes."""

        @component(ComponentType.NODE, namespace="test")
        class TestNode:
            """A test node."""

            pass

        # Decorator should add metadata to the class
        assert hasattr(TestNode, "__hexdag_metadata__")
        metadata = TestNode.__hexdag_metadata__
        assert metadata.type == ComponentType.NODE
        assert metadata.name == "test_node"
        assert metadata.declared_namespace == "test"
        assert metadata.description == "A test node."

    def test_type_specific_decorators_add_metadata(self):
        """Test that type-specific decorators add correct metadata."""

        @node(namespace="test")
        class TestNode:
            pass

        @tool(namespace="test")
        class TestTool:
            pass

        @adapter(implements_port="test_port", namespace="test")
        class TestAdapter:
            pass

        # All should have metadata with correct types
        assert TestNode.__hexdag_metadata__.type == ComponentType.NODE
        assert TestTool.__hexdag_metadata__.type == ComponentType.TOOL
        assert TestAdapter.__hexdag_metadata__.type == ComponentType.ADAPTER

    def test_decorator_with_custom_name(self):
        """Test decorator with custom name."""

        @node(name="custom_name", namespace="test")
        class SomeClass:
            pass

        # Should use custom name in metadata
        assert SomeClass.__hexdag_metadata__.name == "custom_name"

    def test_decorator_with_subtype(self):
        """Test decorator with subtype."""
        from hexai.core.registry.models import NodeSubtype

        @node(namespace="test", subtype=NodeSubtype.LLM)
        class LLMNode:
            pass

        # Should have subtype in metadata
        assert LLMNode.__hexdag_metadata__.subtype == NodeSubtype.LLM


class TestValidation:
    """Test validation of names and namespaces."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clean registry before each test."""
        registry._components.clear()
        registry._protected_components.clear()
        yield
        registry._components.clear()
        registry._protected_components.clear()

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

        # Mixed case namespace is normalized to lowercase
        registry.register("MyComponent", TestComponent, "node", namespace="MyPlugin")
        # Namespace is normalized to lowercase internally
        instance = registry.get("MyComponent", namespace="myplugin")
        assert isinstance(instance, TestComponent)

    def test_invalid_namespace_names(self):
        """Test that invalid namespace names are rejected."""

        class TestComponent:
            pass

        # Namespace with hyphen should fail
        with pytest.raises(InvalidComponentError) as exc_info:
            registry.register("comp", TestComponent, "node", namespace="my-plugin")
        assert "must be alphanumeric" in str(exc_info.value)

        # Namespace with space should fail
        with pytest.raises(InvalidComponentError) as exc_info:
            registry.register("comp", TestComponent, "node", namespace="my plugin")
        assert "must be alphanumeric" in str(exc_info.value)

        # Namespace with dot should fail
        with pytest.raises(InvalidComponentError) as exc_info:
            registry.register("comp", TestComponent, "node", namespace="my.plugin")
        assert "must be alphanumeric" in str(exc_info.value)

        # Namespace with special characters should fail
        with pytest.raises(InvalidComponentError) as exc_info:
            registry.register("comp", TestComponent, "node", namespace="my@plugin")
        assert "must be alphanumeric" in str(exc_info.value)

    def test_invalid_component_names(self):
        """Test that invalid component names are rejected."""

        class TestComponent:
            pass

        # Component name with hyphen should fail
        with pytest.raises(InvalidComponentError) as exc_info:
            registry.register("my-component", TestComponent, "node", namespace="test")
        assert "must be alphanumeric" in str(exc_info.value)

        # Component name with space should fail
        with pytest.raises(InvalidComponentError) as exc_info:
            registry.register("my component", TestComponent, "node", namespace="test")
        assert "must be alphanumeric" in str(exc_info.value)

        # Component name with dot should fail
        with pytest.raises(InvalidComponentError) as exc_info:
            registry.register("my.component", TestComponent, "node", namespace="test")
        assert "must be alphanumeric" in str(exc_info.value)

        # Component name with special character should fail
        with pytest.raises(InvalidComponentError) as exc_info:
            registry.register("my@component", TestComponent, "node", namespace="test")
        assert "must be alphanumeric" in str(exc_info.value)

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
            "processor", CoreNode, ComponentType.NODE, namespace="core", privileged=True
        )

        yield

        registry._components.clear()
        registry._protected_components.clear()

    def test_plugin_can_have_same_name_in_different_namespace(self):
        """Test that plugins can have same name in different namespace."""

        class PluginNode:
            pass

        # Should work fine since it's a different namespace
        registry.register(
            "processor",
            PluginNode,
            ComponentType.NODE,
            namespace="my_plugin",
        )

        # Both should be accessible
        assert registry.get("processor", namespace="core") is not None
        assert registry.get("processor", namespace="my_plugin") is not None

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
        core_instance = registry.get("processor", namespace="core")
        assert not hasattr(core_instance, "is_plugin")

        # Plugin version accessible with explicit namespace
        plugin_instance = registry.get("processor", namespace="my_plugin")
        assert plugin_instance.is_plugin is True

        # Default should still get core
        default_instance = registry.get("processor")
        assert not hasattr(default_instance, "is_plugin")


class TestImprovedAPI:
    """Test the improved API features."""

    def setup_method(self):
        """Set up test fixtures."""
        from hexai.core.registry.registry import ComponentRegistry

        self.registry = ComponentRegistry()

    def test_ergonomic_locking_api(self):
        """Test the new ergonomic locking API."""
        # Should be able to use with registry._lock.read() and write()
        with self.registry._lock.read():
            # Can read
            assert self.registry._components is not None

        with self.registry._lock.write():
            # Can write
            self.registry._components["test"] = {}

    def test_separate_metadata_from_instantiation(self):
        """Test that get_metadata returns metadata without instantiation."""
        # Register a component that tracks instantiation
        instantiation_count = 0

        class TrackedComponent:
            def __init__(self):
                nonlocal instantiation_count
                instantiation_count += 1

        self.registry.register("tracked", TrackedComponent, "node")

        # Getting metadata should NOT instantiate
        metadata = self.registry.get_metadata("tracked")
        assert instantiation_count == 0
        assert metadata.raw_component is TrackedComponent
        assert metadata.name == "tracked"

        # Using get() should instantiate
        instance = self.registry.get("tracked")
        assert instantiation_count == 1
        assert isinstance(instance, TrackedComponent)

    def test_different_namespaces_allow_same_name(self):
        """Test that different namespaces can have same component name."""
        # Register a core component
        self.registry.register(
            "important", lambda: "core", "tool", namespace="core", privileged=True
        )

        # Register same name in plugin namespace (no error)
        self.registry.register(
            "important",
            lambda: "plugin",
            "tool",
            namespace="plugin",
        )

        # Both should be accessible
        assert self.registry.get("important", namespace="core")() == "core"
        assert self.registry.get("important", namespace="plugin")() == "plugin"

    def test_find_namespace_respects_priority(self):
        """Test that _find_namespace respects search priority."""
        # Register same component in multiple namespaces
        self.registry.register("shared", lambda: "user", "tool", namespace="user")
        self.registry.register("shared", lambda: "plugin", "tool", namespace="plugin")
        self.registry.register("shared", lambda: "custom", "tool", namespace="custom_ns")

        # _find_namespace should return based on priority
        found_ns = self.registry._find_namespace("shared")
        assert found_ns == "user"  # user has higher priority than plugin

        # Verify search order
        result = self.registry.get("shared")
        assert result() == "user"

    def test_custom_search_priority(self):
        """Test customizing search priority."""
        # Create registry with custom priority (using internal parameter for testing)
        from hexai.core.registry.registry import ComponentRegistry

        registry = ComponentRegistry(_search_priority=("plugin", "user", "core"))

        # Register in different namespaces
        registry.register("comp", lambda: "user", "tool", namespace="user")
        registry.register("comp", lambda: "plugin", "tool", namespace="plugin")

        # Should find plugin first due to custom priority
        result = registry.get("comp")
        assert result() == "plugin"

    def test_get_metadata_with_qualified_name(self):
        """Test get_metadata with qualified names."""
        self.registry.register("tool", lambda: 1, "tool", namespace="ns1")
        self.registry.register("tool", lambda: 2, "tool", namespace="ns2")

        # Get specific metadata using qualified name
        metadata1 = self.registry.get_metadata("ns1:tool")
        metadata2 = self.registry.get_metadata("ns2:tool")

        assert metadata1.raw_component() == 1
        assert metadata2.raw_component() == 2

    def test_metadata_lookup_error_handling(self):
        """Test error handling in metadata lookup."""
        # Non-existent component
        with pytest.raises(Exception) as exc_info:
            self.registry.get_metadata("nonexistent")

        assert "nonexistent" in str(exc_info.value)

        # Non-existent namespace
        with pytest.raises(ComponentNotFoundError):
            self.registry.get_metadata("nonexistent", namespace="bad_ns")


class TestAdapterRegistration:
    """Test adapter registration and validation in the registry."""

    @pytest.fixture
    def test_registry(self):
        """Create a fresh registry for testing."""
        reg = ComponentRegistry()
        reg._ready = True
        reg._dev_mode = True
        return reg

    @pytest.fixture
    def setup_test_port(self, test_registry):
        """Register a test port in the registry."""
        from hexai.core.registry.models import ClassComponent, ComponentMetadata, PortMetadata

        # Register a test port with required and optional methods
        port_meta = ComponentMetadata(
            name="llm_port",
            component_type=ComponentType.PORT,
            component=ClassComponent(value=type("LLMPort", (), {})),
            namespace="core",
            port_metadata=PortMetadata(
                protocol_class=type("LLMProtocol", (), {}),
                required_methods=["generate", "stream"],
                optional_methods=["embed", "tokenize"],
            ),
        )
        test_registry._components.setdefault("core", {})["llm_port"] = port_meta
        return test_registry

    def test_valid_adapter_registration(self, setup_test_port):
        """Test registering a valid adapter that implements all required methods."""
        reg = setup_test_port

        @adapter(implements_port="llm_port")
        class ValidAdapter:
            def generate(self, prompt: str) -> str:
                return "response"

            def stream(self, prompt: str):
                yield "response"

        reg.register(
            name="valid_adapter",
            component=ValidAdapter,
            component_type="adapter",
            namespace="test",
        )

        assert "test" in reg._components
        assert "valid_adapter" in reg._components["test"]

    def test_adapter_missing_required_method(self, setup_test_port):
        """Test that adapter missing required methods fails validation."""
        reg = setup_test_port

        @adapter(implements_port="llm_port")
        class InvalidAdapter:
            def generate(self, prompt: str) -> str:
                return "response"

        with pytest.raises(InvalidComponentError, match="does not implement required methods"):
            reg.register(
                name="invalid_adapter",
                component=InvalidAdapter,
                component_type="adapter",
                namespace="test",
            )

    def test_adapter_with_nonexistent_port(self, test_registry):
        """Test that adapter declaring nonexistent port fails validation."""
        reg = test_registry

        @adapter(implements_port="nonexistent_port")
        class OrphanAdapter:
            def some_method(self):
                pass

        with pytest.raises(InvalidComponentError, match="port 'nonexistent_port' does not exist"):
            reg.register(
                name="orphan_adapter",
                component=OrphanAdapter,
                component_type="adapter",
                namespace="test",
            )

    def test_get_adapters_for_port(self, setup_test_port):
        """Test getting all adapters that implement a specific port."""
        reg = setup_test_port

        @adapter(implements_port="llm_port")
        class Adapter1:
            def generate(self, prompt: str) -> str:
                return "adapter1"

            def stream(self, prompt: str):
                yield "adapter1"

        @adapter(implements_port="llm_port")
        class Adapter2:
            def generate(self, prompt: str) -> str:
                return "adapter2"

            def stream(self, prompt: str):
                yield "adapter2"

        reg.register("adapter1", Adapter1, "adapter", namespace="test")
        reg.register("adapter2", Adapter2, "adapter", namespace="test")

        adapters = reg.get_adapters_for_port("llm_port")
        assert len(adapters) == 2
        adapter_names = [a.name for a in adapters]
        assert "adapter1" in adapter_names
        assert "adapter2" in adapter_names

    def test_two_phase_discovery(self, test_registry):
        """Test that ports are registered before adapters in two-phase discovery."""
        reg = test_registry

        # First register a port
        @port(
            name="discovery_test_port",
            namespace="test",
            required_methods=["process"],
            optional_methods=["validate"],
        )
        class DiscoveryTestPort:
            def process(self, data: str) -> str: ...

        # Register the port first (simulating phase A)
        reg.register(
            name="discovery_test_port",
            component=DiscoveryTestPort,
            component_type="port",
            namespace="test",
        )

        # Now create and register an adapter that depends on it (simulating phase B)
        @adapter(implements_port="discovery_test_port")
        class DiscoveryTestAdapter:
            def process(self, data: str) -> str:
                return f"processed: {data}"

        # This should succeed because port exists
        reg.register(
            name="discovery_test_adapter",
            component=DiscoveryTestAdapter,
            component_type="adapter",
            namespace="test",
        )

        # Port should be available
        port_info = reg.get_info("discovery_test_port", namespace="test")
        assert port_info.component_type == ComponentType.PORT

        # Adapter should be available and validated against port
        adapter_info = reg.get_info("discovery_test_adapter", namespace="test")
        assert adapter_info.component_type == ComponentType.ADAPTER

        # Adapter should be listed for the port
        adapters = reg.get_adapters_for_port("discovery_test_port")
        assert len(adapters) == 1
        assert adapters[0].name == "discovery_test_adapter"

    def test_adapter_registration_before_port_fails(self, test_registry):
        """Test that registering adapter before its port fails."""
        reg = test_registry

        # Create an adapter without registering its port first
        @adapter(implements_port="not_yet_registered_port")
        class EarlyAdapter:
            def process(self, data: str) -> str:
                return "data"

        # This should fail because port doesn't exist yet
        with pytest.raises(InvalidComponentError, match="does not exist in registry"):
            reg.register(
                name="early_adapter",
                component=EarlyAdapter,
                component_type="adapter",
                namespace="test",
            )
