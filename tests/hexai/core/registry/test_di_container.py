"""Tests for the Dependency Injection Container."""

from __future__ import annotations

import pytest

from hexai.core import registry
from hexai.core.registry.di_container import (
    AdapterNotFoundError,
    DIContainer,
    DIContainerError,
    PortAdapterMismatchError,
    PortNotBoundError,
    PortRequirement,
)
from hexai.core.registry.models import (
    AdapterMetadata,
    ClassComponent,
    ComponentMetadata,
    ComponentType,
    PortMetadata,
)


@pytest.fixture
def container():
    """Create a fresh DI container for testing."""
    return DIContainer()


@pytest.fixture
def setup_test_components():
    """Register test ports and adapters in the registry."""
    # Ensure registry is ready for testing
    registry._ready = True

    # Clear registry first
    registry._components.clear()

    # Register a test port
    port_meta = ComponentMetadata(
        name="test_llm",
        component_type=ComponentType.PORT,
        component=ClassComponent(value=type("TestLLMPort", (), {})),
        namespace="test",
        port_metadata=PortMetadata(
            protocol_class=type("LLMProtocol", (), {}),
            required_methods=["generate", "stream"],
            optional_methods=["embed"],
        ),
    )
    registry._components.setdefault("test", {})["test_llm"] = port_meta

    # Register test adapters
    adapter1_meta = ComponentMetadata(
        name="adapter1",
        component_type=ComponentType.ADAPTER,
        component=ClassComponent(value=type("Adapter1", (), {
            "__init__": lambda self, **kwargs: setattr(self, "config", kwargs),
            "generate": lambda self, prompt: f"adapter1: {prompt}",
            "stream": lambda self, prompt: iter([f"adapter1: {prompt}"]),
        })),
        namespace="test",
        adapter_metadata=AdapterMetadata(
            implements_port="test_llm",
            capabilities=["streaming", "function_calling"],
            singleton=True,
        ),
    )
    registry._components["test"]["adapter1"] = adapter1_meta

    adapter2_meta = ComponentMetadata(
        name="adapter2",
        component_type=ComponentType.ADAPTER,
        component=ClassComponent(value=type("Adapter2", (), {
            "__init__": lambda self, **kwargs: setattr(self, "config", kwargs),
            "generate": lambda self, prompt: f"adapter2: {prompt}",
            "stream": lambda self, prompt: iter([f"adapter2: {prompt}"]),
        })),
        namespace="test",
        adapter_metadata=AdapterMetadata(
            implements_port="test_llm",
            capabilities=["streaming"],
            singleton=False,  # Not a singleton
        ),
    )
    registry._components["test"]["adapter2"] = adapter2_meta

    # Register adapter for different port
    other_adapter_meta = ComponentMetadata(
        name="other_adapter",
        component_type=ComponentType.ADAPTER,
        component=ClassComponent(value=type("OtherAdapter", (), {
            "__init__": lambda self, **kwargs: None,
        })),
        namespace="test",
        adapter_metadata=AdapterMetadata(
            implements_port="other_port",
            capabilities=[],
            singleton=True,
        ),
    )
    registry._components["test"]["other_adapter"] = other_adapter_meta

    yield

    # Cleanup
    registry._components.clear()


class TestBasicBinding:
    """Test basic port-adapter binding functionality."""

    def test_bind_port_success(self, container, setup_test_components):
        """Test successful port binding."""
        # Debug: Check registry state
        print(f"Registry components: {registry._components}")
        print(f"Registry ready: {registry._ready}")

        container.bind_port("test:test_llm", "test:adapter1")

        bindings = container.list_bindings()
        assert "test:test_llm" in bindings
        assert bindings["test:test_llm"] == "test:adapter1"

    def test_bind_port_with_config(self, container, setup_test_components):
        """Test binding with configuration."""
        config = {"api_key": "test_key", "timeout": 30}
        container.bind_port("test:test_llm", "test:adapter1", config=config)

        # Config should be stored
        assert "test:adapter1" in container._adapter_configs
        assert container._adapter_configs["test:adapter1"] == config

    def test_bind_port_invalid_adapter(self, container, setup_test_components):
        """Test binding with non-existent adapter."""
        with pytest.raises((AdapterNotFoundError, DIContainerError)):
            container.bind_port("test:test_llm", "test:nonexistent")

    def test_bind_port_mismatch(self, container, setup_test_components):
        """Test binding adapter to wrong port."""
        with pytest.raises(PortAdapterMismatchError):
            container.bind_port("test:test_llm", "test:other_adapter")


class TestMultipleAdapters:
    """Test multiple adapters per port functionality."""

    def test_bind_multiple_adapters_default(self, container, setup_test_components):
        """Test binding multiple adapters with default selection."""
        container.bind_port("test:test_llm", "test:adapter1")
        container.bind_port("test:test_llm", "test:adapter2", set_as_default=False)

        # adapter1 should remain default
        assert container.get_binding("test:test_llm") == "test:adapter1"

    def test_bind_multiple_adapters_named(self, container, setup_test_components):
        """Test named bindings for multiple adapters."""
        container.bind_port("test:test_llm", "test:adapter1", binding_name="primary")
        container.bind_port("test:test_llm", "test:adapter2", binding_name="fallback")

        named = container.list_named_bindings("test:test_llm")
        assert "primary" in named
        assert "fallback" in named
        assert named["primary"] == "test:adapter1"
        assert named["fallback"] == "test:adapter2"

    def test_get_all_bindings(self, container, setup_test_components):
        """Test getting all adapters bound to a port."""
        container.bind_port("test:test_llm", "test:adapter1", set_as_default=True)
        container.bind_port("test:test_llm", "test:adapter2", binding_name="backup", set_as_default=False)

        all_adapters = container.get_all_bindings("test:test_llm")
        assert len(all_adapters) == 2
        assert "test:adapter1" in all_adapters
        assert "test:adapter2" in all_adapters


class TestPortResolution:
    """Test port resolution to adapter instances."""

    def test_resolve_port_default(self, container, setup_test_components):
        """Test resolving port to default adapter."""
        container.bind_port("test:test_llm", "test:adapter1")

        requirement = PortRequirement(port_name="test:test_llm")
        adapter = container.resolve_port(requirement)

        assert adapter is not None
        assert hasattr(adapter, "generate")
        assert adapter.generate("test") == "adapter1: test"

    def test_resolve_port_named_binding(self, container, setup_test_components):
        """Test resolving port with named binding."""
        container.bind_port("test:test_llm", "test:adapter1", binding_name="primary")
        container.bind_port("test:test_llm", "test:adapter2", binding_name="fallback")

        requirement = PortRequirement(port_name="test:test_llm")
        adapter = container.resolve_port(requirement, binding_name="fallback")

        assert adapter is not None
        assert adapter.generate("test") == "adapter2: test"

    def test_resolve_port_explicit_adapter(self, container, setup_test_components):
        """Test resolving with explicit adapter override."""
        container.bind_port("test:test_llm", "test:adapter1")

        # Explicitly request adapter2
        requirement = PortRequirement(
            port_name="test:test_llm",
            adapter_name="test:adapter2"  # Override default
        )
        adapter = container.resolve_port(requirement)

        assert adapter is not None
        assert adapter.generate("test") == "adapter2: test"

    def test_resolve_optional_unbound(self, container, setup_test_components):
        """Test resolving optional port that's not bound."""
        requirement = PortRequirement(
            port_name="unbound_port",
            optional=True
        )
        adapter = container.resolve_port(requirement)
        assert adapter is None

    def test_resolve_required_unbound(self, container, setup_test_components):
        """Test resolving required port that's not bound."""
        requirement = PortRequirement(
            port_name="unbound_port",
            optional=False
        )
        with pytest.raises(PortNotBoundError):
            container.resolve_port(requirement)

    def test_resolve_with_capabilities(self, container, setup_test_components):
        """Test resolving with capability requirements."""
        container.bind_port("test:test_llm", "test:adapter1")

        # Request specific capabilities
        requirement = PortRequirement(
            port_name="test:test_llm",
            capabilities=["streaming", "function_calling"]
        )
        adapter = container.resolve_port(requirement)
        assert adapter is not None  # adapter1 has both capabilities

        # Request capability that adapter2 doesn't have
        container.bind_port("test:test_llm", "test:adapter2", set_as_default=True)
        with pytest.raises(DIContainerError, match="missing required capabilities"):
            container.resolve_port(requirement)


class TestSingletonManagement:
    """Test singleton instance management."""

    def test_singleton_caching(self, container, setup_test_components):
        """Test that singleton adapters are cached."""
        container.bind_port("test:test_llm", "test:adapter1")

        requirement = PortRequirement(port_name="test:test_llm")
        adapter1 = container.resolve_port(requirement)
        adapter2 = container.resolve_port(requirement)

        # Should be the same instance (singleton)
        assert adapter1 is adapter2

    def test_non_singleton_instances(self, container, setup_test_components):
        """Test that non-singleton adapters create new instances."""
        container.bind_port("test:test_llm", "test:adapter2")

        requirement = PortRequirement(port_name="test:test_llm")
        adapter1 = container.resolve_port(requirement)
        adapter2 = container.resolve_port(requirement)

        # Should be different instances (not singleton)
        assert adapter1 is not adapter2

    def test_clear_singletons(self, container, setup_test_components):
        """Test clearing singleton cache."""
        container.bind_port("test:test_llm", "test:adapter1")

        requirement = PortRequirement(port_name="test:test_llm")
        adapter1 = container.resolve_port(requirement)

        container.clear_singletons()

        adapter2 = container.resolve_port(requirement)
        # Should be different instance after clearing cache
        assert adapter1 is not adapter2


class TestComponentDependencies:
    """Test component creation with dependency injection."""

    def test_create_with_dependencies(self, container, setup_test_components):
        """Test creating component with port dependencies."""
        # Register a component that requires ports
        component_meta = ComponentMetadata(
            name="test_node",
            component_type=ComponentType.NODE,
            component=ClassComponent(value=type("TestNode", (), {
                "__init__": lambda self, llm_port=None, **kwargs: setattr(self, "llm", llm_port),
            })),
            namespace="test",
            port_requirements=[
                PortRequirement(port_name="test:test_llm", field_name="llm_port")
            ],
        )
        registry._components["test"]["test_node"] = component_meta

        # Bind the required port
        container.bind_port("test:test_llm", "test:adapter1")

        # Create component with dependencies
        node = container.create_with_dependencies("test:test_node")

        assert node is not None
        assert hasattr(node, "llm")
        assert node.llm is not None
        assert node.llm.generate("test") == "adapter1: test"

    def test_create_with_missing_required_dependency(self, container, setup_test_components):
        """Test creating component with missing required dependency."""
        component_meta = ComponentMetadata(
            name="test_node",
            component_type=ComponentType.NODE,
            component=ClassComponent(value=type("TestNode", (), {
                "__init__": lambda self, llm_port=None: None,
            })),
            namespace="test",
            port_requirements=[
                PortRequirement(port_name="missing_port", optional=False)
            ],
        )
        registry._components["test"]["test_node"] = component_meta

        with pytest.raises(PortNotBoundError):
            container.create_with_dependencies("test:test_node")

    def test_create_with_optional_dependency(self, container, setup_test_components):
        """Test creating component with optional dependency."""
        component_meta = ComponentMetadata(
            name="test_node",
            component_type=ComponentType.NODE,
            component=ClassComponent(value=type("TestNode", (), {
                "__init__": lambda self, memory_port=None, **kwargs:
                    setattr(self, "memory", memory_port),
            })),
            namespace="test",
            port_requirements=[
                PortRequirement(port_name="memory_port", optional=True, field_name="memory_port")
            ],
        )
        registry._components["test"]["test_node"] = component_meta

        # Create without binding the optional port
        node = container.create_with_dependencies("test:test_node")

        assert node is not None
        assert hasattr(node, "memory")
        assert node.memory is None  # Optional, not bound


class TestUtilityMethods:
    """Test utility methods of DIContainer."""

    def test_clear_bindings(self, container, setup_test_components):
        """Test clearing all bindings."""
        container.bind_port("test:test_llm", "test:adapter1", binding_name="primary")
        container.bind_port("test:test_llm", "test:adapter2", binding_name="backup")

        container.clear_bindings()

        assert len(container.list_bindings()) == 0
        assert len(container.list_named_bindings("test:test_llm")) == 0
        assert len(container._adapter_configs) == 0

    def test_get_binding(self, container, setup_test_components):
        """Test getting specific binding."""
        container.bind_port("test:test_llm", "test:adapter1")

        binding = container.get_binding("test:test_llm")
        assert binding == "test:adapter1"

        # Non-existent binding
        assert container.get_binding("nonexistent") is None

    def test_validate_adapter_implements_port(self, container, setup_test_components):
        """Test adapter-port validation."""
        # Valid implementation
        assert container.validate_adapter_implements_port("test:adapter1", "test:test_llm")

        # Invalid implementation
        assert not container.validate_adapter_implements_port("test:other_adapter", "test:test_llm")


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_named_binding(self, container, setup_test_components):
        """Test resolving with invalid named binding."""
        container.bind_port("test:test_llm", "test:adapter1", binding_name="primary")

        requirement = PortRequirement(port_name="test:test_llm")
        with pytest.raises(PortNotBoundError, match="No binding named 'invalid'"):
            container.resolve_port(requirement, binding_name="invalid")

    def test_port_not_found(self, container, setup_test_components):
        """Test binding to non-existent port."""
        with pytest.raises(DIContainerError, match="Port 'nonexistent' not found"):
            container.bind_port("nonexistent", "test:adapter1")

    def test_adapter_not_found(self, container, setup_test_components):
        """Test resolving non-existent adapter."""
        requirement = PortRequirement(
            port_name="test:test_llm",
            adapter_name="nonexistent"
        )
        with pytest.raises(AdapterNotFoundError):
            container.resolve_port(requirement)
