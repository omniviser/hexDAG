"""Tests for the simplified component registry."""

import asyncio
import os
import warnings
from abc import abstractmethod
from typing import Protocol, runtime_checkable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from hexdag.core.config.models import ManifestEntry
from hexdag.core.registry import adapter, component, node, port, registry, tool
from hexdag.core.registry.exceptions import (
    ComponentAlreadyRegisteredError,
    ComponentNotFoundError,
    InvalidComponentError,
)
from hexdag.core.registry.models import (
    ClassComponent,
    ComponentMetadata,
    ComponentType,  # Internal for tests
)
from hexdag.core.registry.registry import ComponentRegistry

# ============================================================================
# Schema Integration Tests
# ============================================================================


class TestRegistrySchemaIntegration:
    """Test registry.get_schema() method."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Setup test registry with sample component."""
        from hexdag.core.bootstrap import ensure_bootstrapped

        # Ensure registry is bootstrapped
        ensure_bootstrapped()
        yield
        # Don't reset - other tests may need the registry

    def test_get_schema_for_existing_component(self):
        """Test getting schema for an existing node."""
        # Get schema for llm_node
        schema = registry.get_schema("llm_node", namespace="core")

        assert isinstance(schema, dict)
        assert schema["type"] == "object"
        assert "properties" in schema
        # LLM node should have template parameter
        assert "template" in schema["properties"]

    def test_get_schema_yaml_format(self):
        """Test getting schema in YAML format."""
        import yaml

        schema_yaml = registry.get_schema("llm_node", namespace="core", format="yaml")

        assert isinstance(schema_yaml, str)
        # Should be valid YAML
        parsed = yaml.safe_load(schema_yaml)
        assert parsed["type"] == "object"

    def test_get_schema_json_format(self):
        """Test getting schema in JSON format."""
        import json

        schema_json = registry.get_schema("llm_node", namespace="core", format="json")

        assert isinstance(schema_json, str)
        # Should be valid JSON
        parsed = json.loads(schema_json)
        assert parsed["type"] == "object"

    def test_get_schema_nonexistent_component(self):
        """Test getting schema for non-existent component."""
        with pytest.raises(ComponentNotFoundError):
            registry.get_schema("nonexistent_node", namespace="core")

    def test_schema_caching(self):
        """Test that schemas are cached."""
        # First call
        schema1 = registry.get_schema("llm_node", namespace="core")

        # Second call should return cached version
        schema2 = registry.get_schema("llm_node", namespace="core")

        # Should be the same object (cached)
        assert schema1 is schema2

    def test_schema_cache_different_formats(self):
        """Test that different formats are cached separately."""
        schema_dict = registry.get_schema("llm_node", namespace="core", format="dict")
        schema_yaml = registry.get_schema("llm_node", namespace="core", format="yaml")

        # Should be different objects
        assert schema_dict != schema_yaml
        assert isinstance(schema_dict, dict)
        assert isinstance(schema_yaml, str)

    def test_get_schema_with_namespace_inference(self):
        """Test getting schema without specifying namespace."""
        # Should find llm_node in core namespace
        schema = registry.get_schema("llm_node")

        assert isinstance(schema, dict)
        assert "template" in schema["properties"]

    def test_schema_includes_constraints(self):
        """Test that Pydantic Field constraints are in schema."""
        # agent_node has max_steps with constraints
        schema = registry.get_schema("agent_node", namespace="core")

        # Check if max_steps has min/max constraints
        if "max_steps" in schema["properties"]:
            max_steps_prop = schema["properties"]["max_steps"]
            # Should have numeric constraints
            assert max_steps_prop["type"] == "integer"

    def test_schema_includes_descriptions(self):
        """Test that docstring descriptions are included."""
        schema = registry.get_schema("llm_node", namespace="core")

        # Check if any property has a description
        # (depends on whether LLMNode has docstrings)
        properties = schema["properties"]
        assert len(properties) > 0


class TestSchemaForCustomComponents:
    """Test schema generation for custom components."""

    def test_custom_component_schema(self):
        """Test schema generation for a custom registered component."""
        from typing import Annotated

        from hexdag.core.domain.dag import NodeSpec
        from hexdag.core.registry import node

        # Register a custom node
        @node(name="test_schema_node", namespace="test")
        class TestSchemaNode:
            def __call__(
                self,
                name: str,
                value: Annotated[int, Field(ge=0, le=100)],
                mode: str = "default",
            ) -> NodeSpec:
                pass

        # Ensure it's registered
        from hexdag.core.bootstrap import ensure_bootstrapped

        ensure_bootstrapped()

        # Try to register it if not already
        try:
            registry.register(
                "test_schema_node",
                TestSchemaNode,
                "node",
                namespace="test",
            )
        except Exception:
            pass  # Already registered

        # Get schema
        schema = registry.get_schema("test_schema_node", namespace="test")

        # Verify schema structure
        assert "value" in schema["properties"]
        assert schema["properties"]["value"]["minimum"] == 0
        assert schema["properties"]["value"]["maximum"] == 100

        # mode has default
        assert schema["properties"]["mode"]["default"] == "default"

        # value is required (no default)
        assert "value" in schema["required"]


class TestComponentRegistry:
    """Test the main ComponentRegistry class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clean registry before each test."""
        # Clear registry before each test
        registry._reset_for_testing()

        yield

        # Clean up after test
        registry._reset_for_testing()

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

    def test_namespace_collision_detection(self):
        """Test that duplicate names are rejected (flat registry)."""

        class CoreNode:
            pass

        class UserNode:
            pass

        # Register in core namespace
        registry.register("my_node", CoreNode, ComponentType.NODE, namespace="core")

        # Attempting to register same name in different namespace should fail
        with pytest.raises(ComponentAlreadyRegisteredError) as exc_info:
            registry.register("my_node", UserNode, ComponentType.NODE, namespace="user")

        # Error should mention the collision
        assert "my_node" in str(exc_info.value)
        assert "already registered" in str(exc_info.value)

        # Verify only the first registration exists
        instance = registry.get("my_node")
        assert isinstance(instance, CoreNode)

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

    def test_core_namespace_registration(self):
        """Test that any namespace can be used (protection removed)."""

        class MyNode:
            pass

        # Any namespace can be used - protection removed
        registry.register("my_node", MyNode, ComponentType.NODE, namespace="core")

        # Verify it was registered
        info = registry.get_info("my_node", "core")
        assert info.namespace == "core"
        assert not info.is_protected  # Protection feature removed


class TestDecorators:
    """Test decorator functionality - decorators only add metadata, no auto-registration."""

    def test_component_decorator_adds_metadata(self):
        """Test that decorators add metadata to classes."""

        @component(ComponentType.NODE, namespace="test")
        class TestNode:
            """A test node."""

            pass

        # Decorator should add attributes to the class
        assert hasattr(TestNode, "_hexdag_type")
        assert TestNode._hexdag_type == ComponentType.NODE
        assert TestNode._hexdag_name == "test_node"
        assert TestNode._hexdag_namespace == "test"
        assert TestNode._hexdag_description == "A test node."

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

        # All should have attributes with correct types
        assert TestNode._hexdag_type == ComponentType.NODE
        assert TestTool._hexdag_type == ComponentType.TOOL
        assert TestAdapter._hexdag_type == ComponentType.ADAPTER

    def test_decorator_with_custom_name(self):
        """Test decorator with custom name."""

        @node(name="custom_name", namespace="test")
        class SomeClass:
            pass

        # Should use custom name in attribute
        assert SomeClass._hexdag_name == "custom_name"

    def test_decorator_with_subtype(self):
        """Test decorator with subtype."""
        from hexdag.core.registry.models import NodeSubtype

        @node(namespace="test", subtype=NodeSubtype.LLM)
        class LLMNode:
            pass

        # Should have subtype in attribute
        assert LLMNode._hexdag_subtype == NodeSubtype.LLM


class TestValidation:
    """Test validation of names and namespaces."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clean registry before each test."""
        registry._reset_for_testing()
        yield
        registry._reset_for_testing()

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


@pytest.mark.skip(
    reason="Namespace shadowing removed - registry now uses flat storage with unique names"
)
class TestPluginShadowing:
    """Test plugin component shadowing behavior."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clean registry and set up core component."""
        registry._reset_for_testing()

        # Register a core component
        class CoreNode:
            pass

        registry.register("processor", CoreNode, ComponentType.NODE, namespace="core")

        yield

        registry._reset_for_testing()

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


@pytest.mark.skip(
    reason="Namespace priority and searching removed - registry now uses flat storage"
)
class TestImprovedAPI:
    """Test the improved API features."""

    def setup_method(self):
        """Set up test fixtures."""
        from hexdag.core.registry.registry import ComponentRegistry

        self.registry = ComponentRegistry()

    def test_ergonomic_locking_api(self):
        """Test that locking works during registration."""
        # With simplified threading.Lock, just test registration works
        self.registry.register("test_lock", lambda: "locked", "tool")
        result = self.registry.get("test_lock")
        assert result() == "locked"

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
        self.registry.register("important", lambda: "core", "tool", namespace="core")

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
        """Test that namespace search respects priority."""
        # Register same component in multiple namespaces
        self.registry.register("shared", lambda: "user", "tool", namespace="user")
        self.registry.register("shared", lambda: "plugin", "tool", namespace="plugin")
        self.registry.register("shared", lambda: "custom", "tool", namespace="custom_ns")

        # Search should find user first (default priority: core, user, plugin)
        result = self.registry.get("shared")
        assert result() == "user"  # user has higher priority than plugin

    def test_custom_search_priority(self):
        """Test customizing search priority."""
        # Create registry with custom priority (using internal parameter for testing)
        from hexdag.core.registry.registry import ComponentRegistry

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

        # Create a proper Protocol class with required methods
        @runtime_checkable
        class LLMPort(Protocol):
            @abstractmethod
            def generate(self, prompt: str) -> str: ...

            @abstractmethod
            def stream(self, prompt: str): ...

            # Optional methods (with default implementations)
            def embed(self, text: str) -> list[float]:
                return []

            def tokenize(self, text: str) -> list[str]:
                return text.split()

        # Register the port using the internal store (flat dict structure)
        port_meta = ComponentMetadata(
            name="llm_port",
            component_type=ComponentType.PORT,
            component=ClassComponent(value=LLMPort),
            namespace="core",
        )
        test_registry._components["llm_port"] = port_meta
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

        # Flat registry - component name is key
        assert "valid_adapter" in reg._components

    def test_adapter_missing_required_method(self, setup_test_port):
        """Test adapter registration succeeds - validation happens at runtime/type-check time."""
        reg = setup_test_port

        @adapter(implements_port="llm_port")
        class InvalidAdapter:
            def generate(self, prompt: str) -> str:
                return "response"

        # Registration succeeds - Pydantic/type checkers validate at usage time
        reg.register(
            name="invalid_adapter",
            component=InvalidAdapter,
            component_type="adapter",
            namespace="test",
        )
        # Note: Runtime error will occur when trying to use missing methods
        # Type checkers (mypy/pyright) will catch this before runtime

    def test_adapter_with_nonexistent_port(self, test_registry):
        """Test adapter with nonexistent port - now allowed, fails at usage time."""
        reg = test_registry

        @adapter(implements_port="nonexistent_port")
        class OrphanAdapter:
            def some_method(self):
                pass

        # Registration succeeds - port existence checked at usage time
        reg.register(
            name="orphan_adapter",
            component=OrphanAdapter,
            component_type="adapter",
            namespace="test",
        )
        # Note: Error occurs when trying to get adapters for nonexistent port

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

        @port(
            name="discovery_test_port",
            namespace="test",
        )
        @runtime_checkable
        class DiscoveryTestPort(Protocol):
            @abstractmethod
            def process(self, data: str) -> str: ...

            # Optional method with default implementation
            def validate(self, data: str) -> bool:
                return True

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
        """Test adapter registration succeeds even if port registered later."""
        reg = test_registry

        # Create an adapter without registering its port first
        @adapter(implements_port="not_yet_registered_port")
        class EarlyAdapter:
            def process(self, data: str) -> str:
                return "data"

        # Registration succeeds - port can be registered later
        reg.register(
            name="early_adapter",
            component=EarlyAdapter,
            component_type="adapter",
            namespace="test",
        )
        # Note: Port-adapter linkage validated at usage time, not registration


# ============================================================================
# Plugin System Tests
# ============================================================================


class TestRegistryPluginRequirements:
    """Tests for ComponentRegistry._check_plugin_requirements() method."""

    def test_check_plugin_requirements_no_requirements(self):
        """Test checking a module without special requirements."""
        registry = ComponentRegistry()

        # Module without requirements should return None (no skip reason)
        result = registry._check_plugin_requirements("hexdag.builtin.adapters.mock.mock_llm")
        assert result is None

    def test_check_plugin_requirements_missing_package(self):
        """Test checking when required package is not installed."""
        registry = ComponentRegistry()

        with patch("importlib.util.find_spec", return_value=None):
            result = registry._check_plugin_requirements("hexdag.adapters.llm.openai_adapter")

        assert result is not None
        assert "Module hexdag.adapters.llm.openai_adapter not found" in result

    def test_check_plugin_requirements_missing_env_var(self):
        """Test checking when module exists (no longer checks env vars)."""
        registry = ComponentRegistry()

        # Mock package installed - env vars are not checked at import time
        with (
            patch("importlib.util.find_spec", return_value=MagicMock()),
            patch.dict(os.environ, {}, clear=True),
        ):
            result = registry._check_plugin_requirements("hexdag.adapters.llm.openai_adapter")

        # Should return None since module exists (env vars checked at runtime)
        assert result is None

    def test_check_plugin_requirements_all_met(self):
        """Test when all requirements are met."""
        registry = ComponentRegistry()

        # Mock both package and env var present
        with (
            patch("importlib.util.find_spec", return_value=MagicMock()),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
        ):
            result = registry._check_plugin_requirements("hexdag.adapters.llm.openai_adapter")

        assert result is None  # No reason to skip

    @pytest.mark.parametrize(
        "module_path,package,env_var",
        [
            ("hexdag.adapters.llm.openai_adapter", "openai", "OPENAI_API_KEY"),
            ("hexdag.adapters.llm.anthropic_adapter", "anthropic", "ANTHROPIC_API_KEY"),
        ],
    )
    def test_check_plugin_requirements_parametrized(self, module_path, package, env_var):
        """Test requirement checking for different adapters (simplified)."""
        registry = ComponentRegistry()

        # Test missing module
        with patch("importlib.util.find_spec", return_value=None):
            result = registry._check_plugin_requirements(module_path)
            assert f"Module {module_path} not found" in result

        # Test module exists (env vars not checked at import time)
        with (
            patch("importlib.util.find_spec", return_value=MagicMock()),
            patch.dict(os.environ, {}, clear=True),
        ):
            result = registry._check_plugin_requirements(module_path)
            assert result is None  # Module exists, env vars checked at runtime

        # Test all requirements met
        with (
            patch("importlib.util.find_spec", return_value=MagicMock()),
            patch.dict(os.environ, {env_var: "test-key"}),
        ):
            result = registry._check_plugin_requirements(module_path)
            assert result is None


class TestRegistryManifestLoading:
    """Tests for ComponentRegistry._load_manifest_modules() method."""

    @pytest.fixture
    def clean_registry(self):
        """Provide a clean registry for each test."""
        registry = ComponentRegistry()
        yield registry
        # Cleanup
        registry._reset_for_testing()

    def test_load_manifest_with_plugins_all_available(self, clean_registry):
        """Test loading manifest when all plugins are available."""
        from hexdag.core.config.models import ManifestEntry

        manifest = [
            ManifestEntry(module="hexdag.core.ports", namespace="core"),
            ManifestEntry(module="hexdag.builtin.adapters.mock.mock_llm", namespace="plugin"),
        ]

        # Mock the register function to avoid actual imports
        # The function is imported as default_register_components in registry module
        with patch("hexdag.core.registry.registry.default_register_components") as mock_register:
            mock_register.return_value = 1  # Simulate 1 component registered

            total = clean_registry._load_manifest_modules(manifest, mock_register)

        assert total == 2  # Both modules loaded
        assert mock_register.call_count == 2

    def test_load_manifest_skip_unavailable_plugin(self, clean_registry):
        """Test that unavailable plugins are skipped gracefully."""
        from hexdag.core.config.models import ManifestEntry

        manifest = [
            ManifestEntry(module="hexdag.core.ports", namespace="core"),
            ManifestEntry(module="hexdag.adapters.llm.openai_adapter", namespace="plugin"),
        ]

        with (
            patch("hexdag.core.registry.registry.default_register_components") as mock_register,
            patch.object(clean_registry, "_check_plugin_requirements") as mock_check,
        ):
            # Only the second module (plugin) gets checked, and it's missing env var
            mock_check.return_value = "Missing environment variable"
            mock_register.return_value = 5  # Core ports registers 5 components

            total = clean_registry._load_manifest_modules(manifest, mock_register)

        assert total == 5  # Only core module loaded
        assert mock_register.call_count == 1  # Plugin was skipped
        assert mock_check.call_count == 1  # Only checked for the non-core module

    def test_load_manifest_core_module_failure(self, clean_registry):
        """Test that core module failures raise exceptions."""
        from hexdag.core.config.models import ManifestEntry

        manifest = [
            ManifestEntry(module="nonexistent.module", namespace="core"),
        ]

        with patch("hexdag.core.registry.registry.default_register_components") as mock_register:
            mock_register.side_effect = ImportError("Module not found")

            with pytest.raises(ImportError):
                clean_registry._load_manifest_modules(manifest, mock_register)

    def test_load_manifest_plugin_module_failure_continues(self, clean_registry):
        """Test that plugin module failures don't stop loading."""
        from hexdag.core.config.models import ManifestEntry

        manifest = [
            ManifestEntry(module="hexdag.core.ports", namespace="core"),
            ManifestEntry(module="broken.plugin", namespace="plugin"),
            ManifestEntry(module="hexdag.builtin.adapters.mock", namespace="plugin"),
        ]

        with patch("hexdag.core.registry.registry.default_register_components") as mock_register:
            # First succeeds, broken.plugin is skipped by _check_plugin_requirements, third succeeds
            mock_register.side_effect = [5, 3]  # Only called for valid modules

            total = clean_registry._load_manifest_modules(manifest, mock_register)

        assert total == 8  # 5 from core + 3 from mock
        assert mock_register.call_count == 2  # Only called for existing modules


class TestRegistryPluginBootstrap:
    """Integration tests for plugin-related bootstrap functionality."""

    @pytest.fixture
    def temp_config(self, tmp_path):
        """Create a temporary configuration file."""
        config_file = tmp_path / "hexdag.toml"
        config_content = """
# Test configuration
modules = [
    "hexdag.core.ports",
]

plugins = [
    "hexdag.builtin.adapters.llm.openai_adapter",
    "hexdag.builtin.adapters.llm.anthropic_adapter",
    "hexdag.builtin.adapters.mock",
]
"""
        config_file.write_text(config_content)
        return config_file

    def test_bootstrap_with_missing_plugins(self, temp_config):
        """Test bootstrap loads adapters even without environment variables.

        Adapters are loaded at bootstrap time if the module exists.
        Missing API keys are handled at runtime when the adapter is instantiated.
        """
        from hexdag.core.bootstrap import bootstrap_registry
        from hexdag.core.registry import registry as global_registry

        # Clear any existing registry
        if global_registry.ready:
            global_registry._reset_for_testing()

        # Bootstrap without API keys
        with patch.dict(os.environ, {}, clear=True):
            bootstrap_registry(config_path=temp_config)

        # Check that core components are loaded
        components = global_registry.list_components()
        assert any(c.name == "llm" and c.namespace == "core" for c in components)

        # Check that mock adapter is loaded (no requirements)
        assert any(c.name == "mock_llm" and c.namespace == "plugin" for c in components)

        # Check that OpenAI/Anthropic ARE loaded (modules exist, env vars checked at runtime)
        assert any(c.name == "openai" for c in components)
        assert any(c.name == "anthropic" for c in components)

        # Cleanup
        global_registry._reset_for_testing()

    def test_bootstrap_with_all_plugins_available(self, temp_config):
        """Test bootstrap when all plugin requirements are met."""
        from hexdag.core.bootstrap import bootstrap_registry
        from hexdag.core.registry import registry as global_registry

        # Clear any existing registry
        if global_registry.ready:
            global_registry._reset_for_testing()

        # Bootstrap with API keys set
        with patch.dict(
            os.environ, {"OPENAI_API_KEY": "test-key", "ANTHROPIC_API_KEY": "test-key"}
        ):
            bootstrap_registry(config_path=temp_config)

        # Check all adapters are loaded
        components = global_registry.list_components()
        adapter_names = [c.name for c in components if c.component_type.value == "adapter"]

        assert "mock_llm" in adapter_names
        assert "openai" in adapter_names
        assert "anthropic" in adapter_names

        # Cleanup
        global_registry._reset_for_testing()


class TestRegistryPluginScenarios:
    """End-to-end tests for plugin registration scenarios."""

    @pytest.mark.asyncio
    async def test_use_mock_adapter_always_available(self):
        """Test that mock adapters work without any setup."""
        from hexdag.builtin.adapters.mock.mock_llm import MockLLM
        from hexdag.core.ports.llm import Message

        mock_llm = MockLLM()
        response = await mock_llm.aresponse([Message(role="user", content="Test")])

        assert response is not None
        assert "Mock response" in response

    @pytest.mark.asyncio
    async def test_use_real_adapter_with_requirements(self):
        """Test using real adapter when requirements are met."""
        # This test will be skipped if openai is not installed
        pytest.importorskip("openai")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from hexdag.builtin.adapters.llm.openai_adapter import OpenAIAdapter
            from hexdag.core.ports.llm import Message

            # Create adapter
            adapter = OpenAIAdapter(model="gpt-4o-mini", max_tokens=10)

            # Mock the actual API call using AsyncMock
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"

            adapter.client.chat.completions.create = AsyncMock(return_value=mock_response)

            response = await adapter.aresponse([Message(role="user", content="Test")])

            assert response == "Test response"

    def test_registry_list_shows_available_plugins(self):
        """Test that registry list only shows available plugins."""
        from hexdag.core.bootstrap import bootstrap_registry
        from hexdag.core.config.models import HexDAGConfig
        from hexdag.core.registry import registry as global_registry

        # Clear registry
        if global_registry.ready:
            global_registry._reset_for_testing()

        # Create minimal config
        config = HexDAGConfig(
            modules=["hexdag.core.ports"],
            plugins=[
                "hexdag.builtin.adapters.llm.openai_adapter",
                "hexdag.builtin.adapters.mock",
            ],
        )

        # Bootstrap without OpenAI API key
        with (
            patch("hexdag.core.bootstrap.load_config", return_value=config),
            patch.dict(os.environ, {}, clear=True),
        ):
            bootstrap_registry()

        # List adapters
        components = global_registry.list_components()
        adapters = [c for c in components if c.component_type.value == "adapter"]

        # Mock should be present, OpenAI should not
        adapter_names = [a.name for a in adapters]
        assert "mock_llm" in adapter_names
        assert "openai" in adapter_names

        # Cleanup
        global_registry._reset_for_testing()


# ============================================================================
# Convention Over Configuration Tests
# Moved from test_registry_convention.py
# ============================================================================


class TestConventionIntegration:
    """Integration tests showing the full convention over configuration flow."""

    @pytest.fixture
    def registry(self):
        """Create a test registry."""
        reg = ComponentRegistry()
        reg.bootstrap(manifest=[], dev_mode=True)
        return reg

    def test_complete_port_adapter_workflow(self, registry):
        """Test complete workflow: define port -> create adapters -> validate."""
        from hexdag.core.registry.introspection import (
            extract_port_methods,
            infer_adapter_capabilities,
            validate_adapter_implementation,
        )

        # Define a port using Protocol
        @port(name="messaging", namespace="test")
        @runtime_checkable
        class MessagingPort(Protocol):
            """Messaging port with required and optional methods."""

            @abstractmethod
            async def send_message(self, to: str, message: str) -> bool:
                """Required: Send a message."""
                ...

            @abstractmethod
            async def receive_message(self, from_user: str) -> str | None:
                """Required: Receive a message."""
                ...

            def get_status(self) -> dict:
                """Optional: Get messaging status."""
                return {"status": "unknown"}

            def set_priority(self, level: int) -> None:
                """Optional: Set message priority."""
                pass

        # Extract methods automatically
        required, optional = extract_port_methods(MessagingPort)
        assert set(required) == {"send_message", "receive_message"}
        assert set(optional) == {"get_status", "set_priority"}

        # Register the port
        registry.register(
            name="messaging",
            component=MessagingPort,
            component_type="port",
            namespace="test",
        )

        # Create a full adapter with all features
        @adapter(implements_port="messaging", name="full_messenger")
        class FullMessenger:
            """Adapter with all optional methods."""

            async def send_message(self, to: str, message: str) -> bool:
                await asyncio.sleep(0.01)
                return True

            async def receive_message(self, from_user: str) -> str | None:
                await asyncio.sleep(0.01)
                return f"Hello from {from_user}"

            def get_status(self) -> dict:
                return {"status": "online", "queue": 5}

            def set_priority(self, level: int) -> None:
                self.priority = level

        # Validate it implements the port correctly
        is_valid, missing = validate_adapter_implementation(FullMessenger, MessagingPort)
        assert is_valid is True
        assert missing == []

        # Check capabilities
        capabilities = infer_adapter_capabilities(FullMessenger, MessagingPort)
        assert set(capabilities) == {"supports_get_status", "supports_set_priority"}

        # Register the adapter
        registry.register(
            name="full_messenger",
            component=FullMessenger,
            component_type="adapter",
            namespace="test",
        )

        # Get adapters for the port
        adapters = registry.get_adapters_for_port("messaging")
        assert len(adapters) == 1
        assert adapters[0].name == "full_messenger"


"""Tests for registry tracking of configurable components."""


class TestConfigurableRegistry:
    """Test the registry's ability to track configurable components."""

    def setup_method(self):
        """Reset registry before each test."""
        if registry.ready:
            registry._reset_for_testing()

    @pytest.mark.skip(
        reason="Configurable components tracking removed - using decorator pattern now"
    )
    def test_configurable_component_registration(self):
        """Test that configurable components are tracked by the registry."""

        # Create a test module with a configurable component
        import sys
        import types

        # Create a mock module with proper spec
        test_module = types.ModuleType("test_configurable_module")
        test_module.__spec__ = types.SimpleNamespace(
            name="test_configurable_module",
            loader=None,
            origin=None,
            submodule_search_locations=None,
        )

        # Create a configurable adapter class
        class TestConfig(BaseModel):
            """Test configuration."""

            api_key: str = Field(default="test-key", description="API key")
            timeout: int = Field(default=30, description="Timeout")

        class TestConfigurableAdapter:
            """Test adapter with configuration."""

            _hexdag_type = ComponentType.ADAPTER
            _hexdag_name = "test_configurable"
            # Don't declare a port to avoid validation issues
            # _hexdag_implements_port = "llm"

            @classmethod
            def get_config_class(cls) -> type[BaseModel]:
                return TestConfig

            async def aresponse(self, messages):
                return "test response"

        # Fix the __module__ attribute to match our mock module
        TestConfigurableAdapter.__module__ = "test_configurable_module"

        # Add the adapter to the module
        test_module.TestConfigurableAdapter = TestConfigurableAdapter
        sys.modules["test_configurable_module"] = test_module

        # Bootstrap with the test module
        entries = [ManifestEntry(namespace="test", module="test_configurable_module")]

        registry.bootstrap(entries, dev_mode=True)

        # Check that the configurable component was tracked
        configurable = registry.get_configurable_components()
        assert "test_configurable" in configurable

        # Check the configuration class is accessible
        info = configurable["test_configurable"]
        assert info["config_class"] == TestConfig
        assert info["name"] == "test_configurable"
        assert info["namespace"] == "test"

        # Clean up
        del sys.modules["test_configurable_module"]

    @pytest.mark.skip(
        reason="Configurable components tracking removed - using decorator pattern now"
    )
    def test_non_configurable_component_not_tracked(self):
        """Test that non-configurable components are not in configurable list."""

        # Create a test module with a non-configurable component
        import sys
        import types

        test_module = types.ModuleType("test_non_configurable_module")
        test_module.__spec__ = types.SimpleNamespace(
            name="test_non_configurable_module",
            loader=None,
            origin=None,
            submodule_search_locations=None,
        )

        class TestAdapter:
            """Test adapter without configuration."""

            _hexdag_type = ComponentType.ADAPTER
            _hexdag_name = "test_non_configurable"
            # Don't declare a port to avoid validation issues
            # _hexdag_implements_port = "llm"

            async def aresponse(self, messages):
                return "test response"

        # Fix the __module__ attribute to match our mock module
        TestAdapter.__module__ = "test_non_configurable_module"

        test_module.TestAdapter = TestAdapter
        sys.modules["test_non_configurable_module"] = test_module

        # Bootstrap with the test module
        entries = [ManifestEntry(namespace="test", module="test_non_configurable_module")]

        registry.bootstrap(entries, dev_mode=True)

        # Check that the non-configurable component is registered but not tracked
        components = registry.list_components()
        assert any(c.name == "test_non_configurable" for c in components)

        # But not in configurable components
        configurable = registry.get_configurable_components()
        assert "test_non_configurable" not in configurable

        # Clean up
        del sys.modules["test_non_configurable_module"]
