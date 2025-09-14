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
from hexai.core.registry.models import ComponentType  # Internal for tests
from hexai.core.registry.models import NodeSubtype


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

        # Decorator should add metadata to the class
        assert hasattr(TestComponent, "__hexdag_metadata__")
        metadata = TestComponent.__hexdag_metadata__
        assert metadata.name == "test_component"
        assert metadata.type == ComponentType.NODE
        assert metadata.declared_namespace == "test"
        assert metadata.description == "Test component."

    def test_custom_name(self):
        """Test decoration with custom name."""

        @component(ComponentType.NODE, name="custom", namespace="test")
        class TestComponent:
            pass

        # Should use custom name in metadata
        assert hasattr(TestComponent, "__hexdag_metadata__")
        metadata = TestComponent.__hexdag_metadata__
        assert metadata.name == "custom"

    def test_description_from_docstring(self):
        """Test description extraction from docstring."""

        @component(ComponentType.NODE, namespace="test")
        class DocumentedComponent:
            """Well-documented component.

            It has multiple lines.
            """

            pass

        # Check metadata directly on the class
        assert hasattr(DocumentedComponent, "__hexdag_metadata__")
        metadata = DocumentedComponent.__hexdag_metadata__
        assert "Well-documented component" in metadata.description

    def test_explicit_description(self):
        """Test explicit description parameter."""

        @component(ComponentType.NODE, namespace="test", description="Explicit description")
        class TestComponent:
            """Docstring description."""

            pass

        # Check metadata directly on the class
        assert hasattr(TestComponent, "__hexdag_metadata__")
        metadata = TestComponent.__hexdag_metadata__
        assert metadata.description == "Explicit description"

    def test_subtype_parameter(self):
        """Test subtype parameter for nodes."""

        @component(ComponentType.NODE, namespace="test", subtype=NodeSubtype.FUNCTION)
        class FunctionComponent:
            pass

        # Check metadata directly on the class
        assert hasattr(FunctionComponent, "__hexdag_metadata__")
        metadata = FunctionComponent.__hexdag_metadata__
        assert metadata.subtype == NodeSubtype.FUNCTION

    def test_core_namespace_privilege(self):
        """Test that core namespace gets privileged access."""

        @component(ComponentType.NODE, namespace="core")
        class CoreComponent:
            pass

        # Check metadata directly on the class
        assert hasattr(CoreComponent, "__hexdag_metadata__")
        metadata = CoreComponent.__hexdag_metadata__
        # Note: declared_namespace is what's set by decorator
        # The actual namespace is determined during bootstrap
        assert metadata.declared_namespace == "core"


class TestTypeSpecificDecorators:
    """Test type-specific decorator shortcuts."""

    def test_node_decorator(self):
        """Test @node decorator."""

        @node(namespace="test")
        class TestNode:
            pass

        assert hasattr(TestNode, "__hexdag_metadata__")
        metadata = TestNode.__hexdag_metadata__
        assert metadata.type == ComponentType.NODE

    def test_tool_decorator(self):
        """Test @tool decorator."""

        @tool(namespace="test")
        class TestTool:
            pass

        assert hasattr(TestTool, "__hexdag_metadata__")
        metadata = TestTool.__hexdag_metadata__
        assert metadata.type == ComponentType.TOOL

    def test_adapter_decorator(self):
        """Test @adapter decorator."""

        @adapter(implements_port="test_port", namespace="test")
        class TestAdapter:
            pass

        assert hasattr(TestAdapter, "__hexdag_metadata__")
        metadata = TestAdapter.__hexdag_metadata__
        assert metadata.type == ComponentType.ADAPTER
        assert metadata.adapter_metadata is not None
        assert metadata.adapter_metadata.implements_port == "test_port"

    def test_policy_decorator(self):
        """Test @policy decorator."""

        @policy(namespace="test")
        class TestPolicy:
            pass

        assert hasattr(TestPolicy, "__hexdag_metadata__")
        metadata = TestPolicy.__hexdag_metadata__
        assert metadata.type == ComponentType.POLICY

    def test_memory_decorator(self):
        """Test @memory decorator."""

        @memory(namespace="test")
        class TestMemory:
            pass

        assert hasattr(TestMemory, "__hexdag_metadata__")
        metadata = TestMemory.__hexdag_metadata__
        assert metadata.type == ComponentType.MEMORY

    def test_observer_decorator(self):
        """Test @observer decorator."""

        @observer(namespace="test")
        class TestObserver:
            pass

        assert hasattr(TestObserver, "__hexdag_metadata__")
        metadata = TestObserver.__hexdag_metadata__
        assert metadata.type == ComponentType.OBSERVER

    def test_function_node_decorator(self):
        """Test @function_node decorator."""

        @function_node(namespace="test")
        class TestFunctionNode:
            pass

        assert hasattr(TestFunctionNode, "__hexdag_metadata__")
        metadata = TestFunctionNode.__hexdag_metadata__
        assert metadata.type == ComponentType.NODE
        assert metadata.subtype == NodeSubtype.FUNCTION

    def test_llm_node_decorator(self):
        """Test @llm_node decorator."""

        @llm_node(namespace="test")
        class TestLLMNode:
            pass

        assert hasattr(TestLLMNode, "__hexdag_metadata__")
        metadata = TestLLMNode.__hexdag_metadata__
        assert metadata.type == ComponentType.NODE
        assert metadata.subtype == NodeSubtype.LLM

    def test_agent_node_decorator(self):
        """Test @agent_node decorator."""

        @agent_node(namespace="test")
        class TestAgentNode:
            pass

        assert hasattr(TestAgentNode, "__hexdag_metadata__")
        metadata = TestAgentNode.__hexdag_metadata__
        assert metadata.type == ComponentType.NODE
        assert metadata.subtype == NodeSubtype.AGENT


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

        # All should have metadata attached
        assert hasattr(Node1, "__hexdag_metadata__")
        assert Node1.__hexdag_metadata__.name == "node1"
        assert Node1.__hexdag_metadata__.type == ComponentType.NODE

        assert hasattr(Node2, "__hexdag_metadata__")
        assert Node2.__hexdag_metadata__.name == "node2"
        assert Node2.__hexdag_metadata__.type == ComponentType.NODE

        assert hasattr(Tool1, "__hexdag_metadata__")
        assert Tool1.__hexdag_metadata__.name == "tool1"
        assert Tool1.__hexdag_metadata__.type == ComponentType.TOOL

    def test_same_name_different_classes(self):
        """Test that classes can have same component name."""

        @node(name="shared_name", namespace="test")
        class FirstNode:
            pass

        @node(name="shared_name", namespace="other")
        class SecondNode:
            pass

        # Both should have the metadata with same name but different namespaces
        assert FirstNode.__hexdag_metadata__.name == "shared_name"
        assert FirstNode.__hexdag_metadata__.declared_namespace == "test"

        assert SecondNode.__hexdag_metadata__.name == "shared_name"
        assert SecondNode.__hexdag_metadata__.declared_namespace == "other"


class TestFunctionMetadata:
    """Test that function decorators work correctly."""

    def test_function_decorator(self):
        """Test that functions can be decorated."""

        @tool(namespace="test")
        def my_tool():
            """Tool that does something."""
            return "result"

        # Function should have metadata
        assert hasattr(my_tool, "__hexdag_metadata__")
        assert my_tool.__hexdag_metadata__.name == "my_tool"
        assert my_tool.__hexdag_metadata__.type == ComponentType.TOOL
        assert my_tool.__hexdag_metadata__.declared_namespace == "test"

    def test_function_with_parameters(self):
        """Test that functions with parameters get metadata."""

        @tool(namespace="test")
        def parameterized_tool(x: int, y: int = 5) -> int:
            """Tool with parameters."""
            return x + y

        # Function should have metadata
        assert hasattr(parameterized_tool, "__hexdag_metadata__")
        assert parameterized_tool.__hexdag_metadata__.name == "parameterized_tool"

    def test_generator_function(self):
        """Test that generator functions can be decorated."""

        @tool(namespace="test")
        def generator_tool():
            """Yield values from generator."""
            yield 1
            yield 2
            yield 3

        # Generator function should have metadata
        assert hasattr(generator_tool, "__hexdag_metadata__")
        assert generator_tool.__hexdag_metadata__.name == "generator_tool"
        assert generator_tool.__hexdag_metadata__.type == ComponentType.TOOL


class TestSnakeCaseInDecorator:
    """Test snake_case conversion when used in decorators."""

    def test_decorator_without_name(self):
        """Test that decorator uses snake_case of class name when name not provided."""

        @component(ComponentType.NODE, namespace="test")
        class XMLHttpRequest:
            pass

        assert XMLHttpRequest.__hexdag_metadata__.name == "xml_http_request"

    def test_decorator_with_explicit_name(self):
        """Test that explicit name overrides snake_case conversion."""

        @component(ComponentType.NODE, name="custom_name", namespace="test")
        class XMLHttpRequest:
            pass

        assert XMLHttpRequest.__hexdag_metadata__.name == "custom_name"

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

        assert SimpleHTTPServer.__hexdag_metadata__.name == "simple_http_server"
        assert JSONDecoder.__hexdag_metadata__.name == "json_decoder"
        assert MyAPIClass.__hexdag_metadata__.name == "my_api_class"


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

        # All should have correct metadata
        assert StringNode.__hexdag_metadata__.type == ComponentType.NODE
        assert StringTool.__hexdag_metadata__.type == ComponentType.TOOL
        assert StringAdapter.__hexdag_metadata__.type == ComponentType.ADAPTER

    def test_string_namespace(self):
        """Test that namespaces can be strings."""

        @node(namespace="my_plugin")
        class PluginNode:
            pass

        # Should have declared namespace in metadata
        assert hasattr(PluginNode, "__hexdag_metadata__")
        assert PluginNode.__hexdag_metadata__.declared_namespace == "my_plugin"

    def test_string_subtype(self):
        """Test that subtypes can be strings."""

        @component("node", namespace="user", subtype="function")
        class FuncNode:
            pass

        @component("node", namespace="user", subtype="llm")
        class LLMNode:
            pass

        # Should have correct subtypes in metadata
        assert FuncNode.__hexdag_metadata__.subtype == "function"
        assert LLMNode.__hexdag_metadata__.subtype == "llm"

    def test_mixed_string_and_enum(self):
        """Test mixing strings and enums works."""

        @component(ComponentType.NODE, namespace="user")  # Enum type, string namespace
        class MixedNode1:
            pass

        @component("node", namespace="user")  # String type, string namespace
        class MixedNode2:
            pass

        # Both should have correct type in metadata
        assert MixedNode1.__hexdag_metadata__.type == ComponentType.NODE
        assert MixedNode2.__hexdag_metadata__.type == ComponentType.NODE

    def test_default_string_namespace(self):
        """Test that default namespace is 'user' string."""

        @node()  # No namespace specified, should default to "user"
        class DefaultNode:
            pass

        # Should have default namespace in metadata
        assert hasattr(DefaultNode, "__hexdag_metadata__")
        assert DefaultNode.__hexdag_metadata__.declared_namespace == "user"

    def test_invalid_component_type(self):
        """Test that invalid component types raise an error."""
        with pytest.raises(ValueError, match="Invalid component type 'invalid'"):

            @component("invalid", namespace="user")
            class InvalidComponent:
                pass
