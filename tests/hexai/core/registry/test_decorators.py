"""Tests for the simplified decorators module."""

import pytest

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
from hexai.core.registry.models import (
    ComponentType,  # Internal for tests
    NodeSubtype,
)


class TestSnakeCase:
    """Test the _snake_case function with various input patterns."""

    def test_simple_camel_case(self):
        """Test simple CamelCase conversions."""
        assert _snake_case("CamelCase") == "camel_case"
        assert _snake_case("SimpleCase") == "simple_case"
        assert _snake_case("TwoWords") == "two_words"
        assert _snake_case("SimpleTest") == "simple_test"

    def test_lower_camel_case(self):
        """Test lowerCamelCase (camelCase) conversions."""
        assert _snake_case("camelCase") == "camel_case"
        assert _snake_case("userId") == "user_id"
        assert _snake_case("firstName") == "first_name"

    def test_acronyms(self):
        """Test handling of acronyms."""
        assert _snake_case("XMLHttpRequest") == "xml_http_request"
        assert _snake_case("HTMLParser") == "html_parser"
        assert _snake_case("HTTPSConnection") == "https_connection"
        assert _snake_case("URLPattern") == "url_pattern"
        assert _snake_case("JSONDecoder") == "json_decoder"
        assert _snake_case("XMLDocument") == "xml_document"
        assert _snake_case("PDFReader") == "pdf_reader"
        assert _snake_case("IOError") == "io_error"
        assert _snake_case("HTTPServer") == "http_server"
        assert _snake_case("XMLParser") == "xml_parser"

    def test_mixed_acronyms(self):
        """Test mixed cases with acronyms."""
        assert _snake_case("SimpleHTTPServer") == "simple_http_server"
        assert _snake_case("MyAPIClass") == "my_api_class"
        assert _snake_case("getHTTPResponseCode") == "get_http_response_code"
        assert _snake_case("HTTPResponseCodeError") == "http_response_code_error"
        assert _snake_case("parseHTMLString") == "parse_html_string"
        assert _snake_case("XMLToHTMLConverter") == "xml_to_html_converter"

    def test_all_caps(self):
        """Test all capital letters."""
        assert _snake_case("HTTP") == "http"
        assert _snake_case("API") == "api"
        assert _snake_case("HTTPAPI") == "httpapi"
        assert _snake_case("XMLHTTP") == "xmlhttp"

    def test_numbers(self):
        """Test handling of numbers in names."""
        assert _snake_case("Base64Encoder") == "base64_encoder"
        assert _snake_case("SHA256Hash") == "sha256_hash"
        assert _snake_case("Python3Parser") == "python3_parser"
        assert _snake_case("HTML5Parser") == "html5_parser"

    def test_single_word(self):
        """Test single word conversions."""
        assert _snake_case("Word") == "word"
        assert _snake_case("word") == "word"
        assert _snake_case("WORD") == "word"
        assert _snake_case("w") == "w"
        assert _snake_case("W") == "w"

    def test_underscores_preserved(self):
        """Test that existing underscores are preserved."""
        assert _snake_case("Already_Snake_Case") == "already_snake_case"
        assert _snake_case("Mixed_CamelCase") == "mixed_camel_case"
        assert _snake_case("_PrivateMethod") == "_private_method"
        assert _snake_case("__DoublePrivate") == "__double_private"
        assert _snake_case("already_snake_case") == "already_snake_case"
        assert _snake_case("snake_case") == "snake_case"
        assert _snake_case("_Leading") == "_leading"
        assert _snake_case("__DoubleLeading") == "__double_leading"

    def test_consecutive_capitals(self):
        """Test handling of consecutive capital letters."""
        assert _snake_case("ABCDef") == "abc_def"
        assert _snake_case("ABCD") == "abcd"
        assert _snake_case("AbCdEf") == "ab_cd_ef"

    def test_empty_string(self):
        """Test empty string input."""
        assert _snake_case("") == ""

    def test_real_world_examples(self):
        """Test with real-world class names."""
        # Node types
        assert _snake_case("FunctionNode") == "function_node"
        assert _snake_case("LLMNode") == "llm_node"
        assert _snake_case("ReActAgentNode") == "re_act_agent_node"

        # Component types
        assert _snake_case("InMemoryMemory") == "in_memory_memory"
        assert _snake_case("ComponentRegistry") == "component_registry"
        assert _snake_case("NodeFactory") == "node_factory"

        # Event types
        assert _snake_case("NodeStartedEvent") == "node_started_event"
        assert _snake_case("ToolCalledEvent") == "tool_called_event"

        # Common patterns
        assert _snake_case("BaseNodeFactory") == "base_node_factory"
        assert _snake_case("AbstractBaseClass") == "abstract_base_class"

    def test_edge_cases(self):
        """Test edge cases and unusual patterns."""
        # Multiple transitions
        assert _snake_case("aB") == "a_b"
        assert _snake_case("aBc") == "a_bc"
        assert _snake_case("aBcD") == "a_bc_d"

        # Starting with lowercase
        assert _snake_case("iPhone") == "i_phone"
        assert _snake_case("eBay") == "e_bay"

        # Ending with acronym
        assert _snake_case("RequestHTTP") == "request_http"
        assert _snake_case("ParserXML") == "parser_xml"


class TestComponentDecorator:
    """Test the main component decorator."""

    def test_basic_decoration(self):
        """Test basic component decoration adds metadata."""

        @component(ComponentType.NODE, namespace="test")
        class TestComponent:
            """Test component."""

            pass

        # Decorator should add attributes to the class
        assert hasattr(TestComponent, "_hexdag_type")
        assert TestComponent._hexdag_type == ComponentType.NODE
        assert TestComponent._hexdag_name == "test_component"
        assert TestComponent._hexdag_namespace == "test"
        assert TestComponent._hexdag_description == "Test component."

    def test_custom_name(self):
        """Test decoration with custom name."""

        @component(ComponentType.NODE, name="custom", namespace="test")
        class TestComponent:
            pass

        # Should use custom name
        assert hasattr(TestComponent, "_hexdag_type")
        assert TestComponent._hexdag_name == "custom"

    def test_description_from_docstring(self):
        """Test description extraction from docstring."""

        @component(ComponentType.NODE, namespace="test")
        class DocumentedComponent:
            """Well-documented component.

            It has multiple lines.
            """

            pass

        # Check attributes directly on the class
        assert hasattr(DocumentedComponent, "_hexdag_type")
        assert "Well-documented component" in DocumentedComponent._hexdag_description

    def test_explicit_description(self):
        """Test explicit description parameter."""

        @component(ComponentType.NODE, namespace="test", description="Explicit description")
        class TestComponent:
            """Docstring description."""

            pass

        # Check attributes directly on the class
        assert hasattr(TestComponent, "_hexdag_type")
        assert TestComponent._hexdag_description == "Explicit description"

    def test_subtype_parameter(self):
        """Test subtype parameter for nodes."""

        @component(ComponentType.NODE, namespace="test", subtype=NodeSubtype.FUNCTION)
        class FunctionComponent:
            pass

        # Check attributes directly on the class
        assert hasattr(FunctionComponent, "_hexdag_type")
        assert FunctionComponent._hexdag_subtype == NodeSubtype.FUNCTION

    def test_core_namespace_privilege(self):
        """Test that core namespace gets privileged access."""

        @component(ComponentType.NODE, namespace="core")
        class CoreComponent:
            pass

        # Check attributes directly on the class
        assert hasattr(CoreComponent, "_hexdag_type")
        # Note: _hexdag_namespace is what's set by decorator
        # The actual namespace is determined during bootstrap
        assert CoreComponent._hexdag_namespace == "core"


class TestTypeSpecificDecorators:
    """Test type-specific decorator shortcuts."""

    def test_node_decorator(self):
        """Test @node decorator."""

        @node(namespace="test")
        class TestNode:
            pass

        assert hasattr(TestNode, "_hexdag_type")
        assert TestNode._hexdag_type == ComponentType.NODE

    def test_tool_decorator(self):
        """Test @tool decorator."""

        @tool(namespace="test")
        class TestTool:
            pass

        assert hasattr(TestTool, "_hexdag_type")
        assert TestTool._hexdag_type == ComponentType.TOOL

    def test_adapter_decorator(self):
        """Test @adapter decorator."""

        @adapter(implements_port="test_port", namespace="test")
        class TestAdapter:
            pass

        assert hasattr(TestAdapter, "_hexdag_type")
        assert TestAdapter._hexdag_type == ComponentType.ADAPTER
        # The implements_port is now stored directly on the class
        assert hasattr(TestAdapter, "_hexdag_implements_port")
        assert TestAdapter._hexdag_implements_port == "test_port"  # type: ignore[attr-defined]

    def test_adapter_decorator_with_class_reference(self):
        """Test @adapter decorator with direct port class reference."""
        from hexai.core.registry.decorators import port

        # Create a mock port class
        @port(name="mock_port")
        class MockPort:
            """A mock port for testing."""

            pass

        # Test with decorated port (has _hexdag_name)
        @adapter(implements_port=MockPort)
        class AdapterWithDecoratedPort:
            pass

        assert hasattr(AdapterWithDecoratedPort, "_hexdag_implements_port")
        assert AdapterWithDecoratedPort._hexdag_implements_port == "mock_port"  # type: ignore[attr-defined]

        # Test with undecorated port class
        class PlainPortClass:
            """Plain port class without decorator."""

            pass

        @adapter(implements_port=PlainPortClass)
        class AdapterWithPlainPort:
            pass

        assert hasattr(AdapterWithPlainPort, "_hexdag_implements_port")
        assert AdapterWithPlainPort._hexdag_implements_port == "plain_port_class"  # type: ignore[attr-defined]

        # Test with port class ending in 'Port'
        class DatabasePort:
            """Database port class."""

            pass

        @adapter(implements_port=DatabasePort)
        class DatabaseAdapter:
            pass

        assert hasattr(DatabaseAdapter, "_hexdag_implements_port")
        assert DatabaseAdapter._hexdag_implements_port == "database"  # type: ignore[attr-defined]

    def test_adapter_string_reference_deprecation(self):
        """Test that string-based port references emit deprecation warning."""
        import warnings

        # Test that string reference triggers deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Ensure all warnings are captured

            @adapter(implements_port="test_port")
            class StringBasedAdapter:
                pass

            # Check that a deprecation warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
            assert "test_port" in str(w[0].message)
            assert "string_based_adapter" in str(w[0].message)  # Name is converted to snake_case

    def test_policy_decorator(self):
        """Test @policy decorator."""

        @policy(namespace="test")
        class TestPolicy:
            pass

        assert hasattr(TestPolicy, "_hexdag_type")
        assert TestPolicy._hexdag_type == ComponentType.POLICY

    def test_memory_decorator(self):
        """Test @memory decorator."""

        @memory(namespace="test")
        class TestMemory:
            pass

        assert hasattr(TestMemory, "_hexdag_type")
        assert TestMemory._hexdag_type == ComponentType.MEMORY

    def test_observer_decorator(self):
        """Test @observer decorator."""

        @observer(namespace="test")
        class TestObserver:
            pass

        assert hasattr(TestObserver, "_hexdag_type")
        assert TestObserver._hexdag_type == ComponentType.OBSERVER

    def test_function_node_decorator(self):
        """Test @function_node decorator."""

        @function_node(namespace="test")
        class TestFunctionNode:
            pass

        assert hasattr(TestFunctionNode, "_hexdag_type")
        assert TestFunctionNode._hexdag_type == ComponentType.NODE
        assert TestFunctionNode._hexdag_subtype == NodeSubtype.FUNCTION

    def test_llm_node_decorator(self):
        """Test @llm_node decorator."""

        @llm_node(namespace="test")
        class TestLLMNode:
            pass

        assert hasattr(TestLLMNode, "_hexdag_type")
        assert TestLLMNode._hexdag_type == ComponentType.NODE
        assert TestLLMNode._hexdag_subtype == NodeSubtype.LLM

    def test_agent_node_decorator(self):
        """Test @agent_node decorator."""

        @agent_node(namespace="test")
        class TestAgentNode:
            pass

        assert hasattr(TestAgentNode, "_hexdag_type")
        assert TestAgentNode._hexdag_type == ComponentType.NODE
        assert TestAgentNode._hexdag_subtype == NodeSubtype.AGENT


class TestDecoratorMetadata:
    """Test that decorators correctly attach metadata."""

    def test_multiple_components_metadata(self):
        """Test that multiple components get correct metadata."""

        @node(namespace="test")
        class Node1:
            pass

        @node(namespace="test")
        class Node2:
            pass

        @tool(namespace="test")
        class Tool1:
            pass

        # All should have attributes attached
        assert hasattr(Node1, "_hexdag_type")
        assert Node1._hexdag_name == "node1"
        assert Node1._hexdag_type == ComponentType.NODE

        assert hasattr(Node2, "_hexdag_type")
        assert Node2._hexdag_name == "node2"
        assert Node2._hexdag_type == ComponentType.NODE

        assert hasattr(Tool1, "_hexdag_type")
        assert Tool1._hexdag_name == "tool1"
        assert Tool1._hexdag_type == ComponentType.TOOL

    def test_same_name_different_classes(self):
        """Test that classes can have same component name."""

        @node(name="shared_name", namespace="test")
        class FirstNode:
            pass

        @node(name="shared_name", namespace="other")
        class SecondNode:
            pass

        # Both should have the attributes with same name but different namespaces
        assert FirstNode._hexdag_name == "shared_name"
        assert FirstNode._hexdag_namespace == "test"

        assert SecondNode._hexdag_name == "shared_name"
        assert SecondNode._hexdag_namespace == "other"


class TestFunctionMetadata:
    """Test that function decorators work correctly."""

    def test_function_decorator(self):
        """Test that functions can be decorated."""

        @tool(namespace="test")
        def my_tool():
            """Tool that does something."""
            return "result"

        # Function should have attributes
        assert hasattr(my_tool, "_hexdag_type")
        assert my_tool._hexdag_name == "my_tool"
        assert my_tool._hexdag_type == ComponentType.TOOL
        assert my_tool._hexdag_namespace == "test"

    def test_function_with_parameters(self):
        """Test that functions with parameters get metadata."""

        @tool(namespace="test")
        def parameterized_tool(x: int, y: int = 5) -> int:
            """Tool with parameters."""
            return x + y

        # Function should have attributes
        assert hasattr(parameterized_tool, "_hexdag_type")
        assert parameterized_tool._hexdag_name == "parameterized_tool"

    def test_generator_function(self):
        """Test that generator functions can be decorated."""

        @tool(namespace="test")
        def generator_tool():
            """Yield values from generator."""
            yield 1
            yield 2
            yield 3

        # Generator function should have attributes
        assert hasattr(generator_tool, "_hexdag_type")
        assert generator_tool._hexdag_name == "generator_tool"
        assert generator_tool._hexdag_type == ComponentType.TOOL


class TestSnakeCaseInDecorator:
    """Test snake_case conversion when used in decorators."""

    def test_decorator_without_name(self):
        """Test that decorator uses snake_case of class name when name not provided."""

        @component(ComponentType.NODE, namespace="test")
        class XMLHttpRequest:
            pass

        assert XMLHttpRequest._hexdag_name == "xml_http_request"

    def test_decorator_with_explicit_name(self):
        """Test that explicit name overrides snake_case conversion."""

        @component(ComponentType.NODE, name="custom_name", namespace="test")
        class XMLHttpRequest:
            pass

        assert XMLHttpRequest._hexdag_name == "custom_name"

    def test_various_class_names(self):
        """Test decorator with various class name patterns."""

        @component(ComponentType.NODE, namespace="test")
        class SimpleHTTPServer:
            pass

        @component(ComponentType.TOOL, namespace="test")
        class JSONDecoder:
            pass

        @component(ComponentType.ADAPTER, namespace="test")
        class MyAPIClass:
            pass

        assert SimpleHTTPServer._hexdag_name == "simple_http_server"
        assert JSONDecoder._hexdag_name == "json_decoder"
        assert MyAPIClass._hexdag_name == "my_api_class"


class TestStringUsage:
    """Test string-based decorator usage for user-friendliness."""

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

        # All should have correct attributes
        assert StringNode._hexdag_type == ComponentType.NODE
        assert StringTool._hexdag_type == ComponentType.TOOL
        assert StringAdapter._hexdag_type == ComponentType.ADAPTER

    def test_string_namespace(self):
        """Test that namespaces can be strings."""

        @node(namespace="my_plugin")
        class PluginNode:
            pass

        # Should have declared namespace in attribute
        assert hasattr(PluginNode, "_hexdag_type")
        assert PluginNode._hexdag_namespace == "my_plugin"

    def test_string_subtype(self):
        """Test that subtypes can be strings."""

        @component("node", namespace="user", subtype="function")
        class FuncNode:
            pass

        @component("node", namespace="user", subtype="llm")
        class LLMNode:
            pass

        # Should have correct subtypes in attributes
        assert FuncNode._hexdag_subtype == "function"
        assert LLMNode._hexdag_subtype == "llm"

    def test_mixed_string_and_enum(self):
        """Test mixing strings and enums works."""

        @component(ComponentType.NODE, namespace="user")  # Enum type, string namespace
        class MixedNode1:
            pass

        @component("node", namespace="user")  # String type, string namespace
        class MixedNode2:
            pass

        # Both should have correct type in attributes
        assert MixedNode1._hexdag_type == ComponentType.NODE
        assert MixedNode2._hexdag_type == ComponentType.NODE

    def test_default_string_namespace(self):
        """Test that default namespace is 'user' string."""

        @node()  # No namespace specified, should default to "user"
        class DefaultNode:
            pass

        # Should have default namespace in attribute
        assert hasattr(DefaultNode, "_hexdag_type")
        assert DefaultNode._hexdag_namespace == "user"

    def test_invalid_component_type(self):
        """Test that invalid component types raise an error."""
        with pytest.raises(ValueError, match="Invalid component type 'invalid'"):

            @component("invalid", namespace="user")
            class InvalidComponent:
                pass
