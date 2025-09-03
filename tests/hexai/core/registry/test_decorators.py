"""Tests for the decorators module."""

import pytest

from hexai.core.registry import ComponentRegistry, ComponentType
from hexai.core.registry.decorators import (
    _infer_component_type,
    _snake_case,
    adapter,
    component,
    get_component_metadata,
    memory,
    node,
    observer,
    policy,
    tool,
)


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
        assert _snake_case("HTMLToXML") == "html_to_xml"

    def test_already_snake_case(self):
        """Test that snake_case remains unchanged."""
        assert _snake_case("already_snake") == "already_snake"
        assert _snake_case("test_name") == "test_name"

    def test_single_word(self):
        """Test single word conversion."""
        assert _snake_case("Word") == "word"
        assert _snake_case("TEST") == "test"

    def test_numbers(self):
        """Test handling of numbers in names."""
        assert _snake_case("Test2Node") == "test2_node"
        assert _snake_case("V2Parser") == "v2_parser"

    def test_edge_cases(self):
        """Test edge cases."""
        assert _snake_case("") == ""
        assert _snake_case("A") == "a"
        assert _snake_case("_Leading") == "leading"
        assert _snake_case("__DoubleLeading") == "double_leading"


class TestInferComponentType:
    """Test component type inference."""

    def test_node_inference(self):
        """Test inference for node types."""

        class SomeNode:
            pass

        class DataProcessorNode:
            pass

        class NodeProcessor:
            pass

        assert _infer_component_type(SomeNode) == ComponentType.NODE
        assert _infer_component_type(DataProcessorNode) == ComponentType.NODE
        assert _infer_component_type(NodeProcessor) == ComponentType.NODE

    def test_adapter_inference(self):
        """Test inference for adapter types."""

        class DatabaseAdapter:
            pass

        class AdapterBase:
            pass

        class PostgresAdapter:
            pass

        assert _infer_component_type(DatabaseAdapter) == ComponentType.ADAPTER
        assert _infer_component_type(AdapterBase) == ComponentType.ADAPTER
        assert _infer_component_type(PostgresAdapter) == ComponentType.ADAPTER

    def test_tool_inference(self):
        """Test inference for tool types."""

        class WebScraperTool:
            pass

        class ToolBase:
            pass

        class DataFetchTool:
            pass

        assert _infer_component_type(WebScraperTool) == ComponentType.TOOL
        assert _infer_component_type(ToolBase) == ComponentType.TOOL
        assert _infer_component_type(DataFetchTool) == ComponentType.TOOL

    def test_policy_inference(self):
        """Test inference for policy types."""

        class RetryPolicy:
            pass

        class PolicyBase:
            pass

        class CachingPolicy:
            pass

        assert _infer_component_type(RetryPolicy) == ComponentType.POLICY
        assert _infer_component_type(PolicyBase) == ComponentType.POLICY
        assert _infer_component_type(CachingPolicy) == ComponentType.POLICY

    def test_memory_inference(self):
        """Test inference for memory types."""

        class ConversationMemory:
            pass

        class MemoryStore:
            pass

        class LongTermMemory:
            pass

        assert _infer_component_type(ConversationMemory) == ComponentType.MEMORY
        assert _infer_component_type(MemoryStore) == ComponentType.MEMORY
        assert _infer_component_type(LongTermMemory) == ComponentType.MEMORY

    def test_observer_inference(self):
        """Test inference for observer types."""

        class MetricsObserver:
            pass

        class ObserverBase:
            pass

        class LoggingObserver:
            pass

        assert _infer_component_type(MetricsObserver) == ComponentType.OBSERVER
        assert _infer_component_type(ObserverBase) == ComponentType.OBSERVER
        assert _infer_component_type(LoggingObserver) == ComponentType.OBSERVER

    def test_inheritance_inference(self):
        """Test type inference through inheritance."""

        class BaseNode:
            pass

        class DerivedProcessor(BaseNode):
            pass

        # Should infer from base class name
        assert _infer_component_type(DerivedProcessor) == ComponentType.NODE

    def test_no_inference(self):
        """Test when type cannot be inferred."""

        class RandomClass:
            pass

        class SomeProcessor:
            pass

        assert _infer_component_type(RandomClass) is None
        assert _infer_component_type(SomeProcessor) is None


class TestComponentDecorator:
    """Test the main component decorator."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear pending components before each test."""
        ComponentRegistry._pending_components.clear()
        yield
        ComponentRegistry._pending_components.clear()

    def test_basic_decoration(self):
        """Test basic component decoration."""

        @component(namespace="test", component_type=ComponentType.NODE)
        class TestComponent:
            """Test component."""

            pass

        # Check class attributes
        assert hasattr(TestComponent, "_hexdag_component")
        assert TestComponent._hexdag_component is True
        assert TestComponent._hexdag_namespace == "test"
        assert TestComponent._hexdag_name == "test_component"

        # Check pending registration
        assert len(ComponentRegistry._pending_components) == 1
        cls, metadata = ComponentRegistry._pending_components[0]
        assert cls is TestComponent
        assert metadata["name"] == "test_component"
        assert metadata["namespace"] == "test"
        assert metadata["component_type"] == ComponentType.NODE

    def test_custom_name(self):
        """Test custom component name."""

        @component(name="custom_name", namespace="test", component_type=ComponentType.NODE)
        class SomeClass:
            pass

        assert SomeClass._hexdag_name == "custom_name"
        _, metadata = ComponentRegistry._pending_components[0]
        assert metadata["name"] == "custom_name"

    def test_docstring_as_description(self):
        """Test using docstring as description."""

        @component(namespace="test", component_type=ComponentType.NODE)
        class DocumentedComponent:
            """This is the component description."""

            pass

        _, metadata = ComponentRegistry._pending_components[0]
        assert metadata["description"] == "This is the component description."

    def test_explicit_description(self):
        """Test explicit description overrides docstring."""

        @component(
            namespace="test", component_type=ComponentType.NODE, description="Explicit description"
        )
        class ComponentWithBoth:
            """Docstring description."""

            pass

        _, metadata = ComponentRegistry._pending_components[0]
        assert metadata["description"] == "Explicit description"

    def test_tags_and_dependencies(self):
        """Test tags and dependencies parameters."""

        @component(
            namespace="test",
            component_type=ComponentType.NODE,
            tags={"tag1", "tag2"},
            dependencies={"dep1", "dep2"},
        )
        class TaggedComponent:
            pass

        _, metadata = ComponentRegistry._pending_components[0]
        assert metadata["tags"] == {"tag1", "tag2"}
        assert metadata["dependencies"] == {"dep1", "dep2"}

    def test_all_parameters(self):
        """Test all decorator parameters."""

        @component(
            name="full_component",
            namespace="test_ns",
            component_type=ComponentType.TOOL,
            description="Full description",
            tags={"tag1", "tag2"},
            author="test_author",
            dependencies={"dep1"},
            replaceable=True,
            version="2.0.0",
        )
        class FullyConfigured:
            pass

        _, metadata = ComponentRegistry._pending_components[0]
        assert metadata["name"] == "full_component"
        assert metadata["namespace"] == "test_ns"
        assert metadata["component_type"] == ComponentType.TOOL
        assert metadata["description"] == "Full description"
        assert metadata["tags"] == {"tag1", "tag2"}
        assert metadata["author"] == "test_author"
        assert metadata["dependencies"] == {"dep1"}
        assert metadata["replaceable"] is True
        assert metadata["version"] == "2.0.0"

    def test_type_inference(self):
        """Test automatic type inference."""

        @component(namespace="test")
        class SomeNode:
            pass

        _, metadata = ComponentRegistry._pending_components[0]
        assert metadata["component_type"] == ComponentType.NODE

    def test_type_inference_failure(self):
        """Test error when type cannot be inferred."""
        with pytest.raises(ValueError, match="Cannot infer component type"):

            @component(namespace="test")
            class RandomClass:
                pass

    def test_default_namespace(self):
        """Test default namespace is 'core'."""

        @component(component_type=ComponentType.NODE)
        class CoreComponent:
            pass

        assert CoreComponent._hexdag_namespace == "core"
        _, metadata = ComponentRegistry._pending_components[0]
        assert metadata["namespace"] == "core"


class TestTypeSpecificDecorators:
    """Test type-specific decorator shortcuts."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear pending components before each test."""
        ComponentRegistry._pending_components.clear()
        yield
        ComponentRegistry._pending_components.clear()

    def test_node_decorator(self):
        """Test @node decorator."""

        @node(namespace="test")
        class TestNode:
            """Node description."""

            pass

        _, metadata = ComponentRegistry._pending_components[0]
        assert metadata["component_type"] == ComponentType.NODE
        assert metadata["name"] == "test_node"
        assert metadata["namespace"] == "test"

    def test_tool_decorator(self):
        """Test @tool decorator."""

        @tool(namespace="test", author="tool_author")
        class DataFetcher:
            pass

        _, metadata = ComponentRegistry._pending_components[0]
        assert metadata["component_type"] == ComponentType.TOOL
        assert metadata["name"] == "data_fetcher"
        assert metadata["author"] == "tool_author"

    def test_adapter_decorator(self):
        """Test @adapter decorator."""

        @adapter(namespace="test", version="1.5.0")
        class PostgresAdapter:
            pass

        _, metadata = ComponentRegistry._pending_components[0]
        assert metadata["component_type"] == ComponentType.ADAPTER
        assert metadata["name"] == "postgres_adapter"
        assert metadata["version"] == "1.5.0"

    def test_policy_decorator(self):
        """Test @policy decorator."""

        @policy(namespace="test", replaceable=True)
        class RetryPolicy:
            pass

        _, metadata = ComponentRegistry._pending_components[0]
        assert metadata["component_type"] == ComponentType.POLICY
        assert metadata["name"] == "retry_policy"
        assert metadata["replaceable"] is True

    def test_memory_decorator(self):
        """Test @memory decorator."""

        @memory(namespace="test", tags={"persistent"})
        class ConversationMemory:
            pass

        _, metadata = ComponentRegistry._pending_components[0]
        assert metadata["component_type"] == ComponentType.MEMORY
        assert metadata["name"] == "conversation_memory"
        assert "persistent" in metadata["tags"]

    def test_observer_decorator(self):
        """Test @observer decorator."""

        @observer(namespace="test", dependencies={"logger"})
        class MetricsObserver:
            pass

        _, metadata = ComponentRegistry._pending_components[0]
        assert metadata["component_type"] == ComponentType.OBSERVER
        assert metadata["name"] == "metrics_observer"
        assert "logger" in metadata["dependencies"]

    def test_decorator_default_namespace(self):
        """Test that type decorators default to 'core' namespace."""

        @node()
        class CoreNode:
            pass

        _, metadata = ComponentRegistry._pending_components[0]
        assert metadata["namespace"] == "core"


class TestGetComponentMetadata:
    """Test the get_component_metadata function."""

    def test_get_metadata_from_decorated(self):
        """Test getting metadata from decorated component."""

        @node(namespace="test")
        class DecoratedNode:
            pass

        metadata = get_component_metadata(DecoratedNode)
        assert metadata is not None
        assert metadata["name"] == "decorated_node"
        assert metadata["namespace"] == "test"

    def test_get_metadata_from_undecorated(self):
        """Test getting metadata from undecorated class."""

        class PlainClass:
            pass

        metadata = get_component_metadata(PlainClass)
        assert metadata is None

    def test_get_metadata_from_instance(self):
        """Test getting metadata from instance of decorated class."""

        @tool(namespace="test")
        class SomeTool:
            pass

        instance = SomeTool()
        metadata = get_component_metadata(instance)
        assert metadata is not None
        assert metadata["name"] == "some_tool"
        assert metadata["namespace"] == "test"
