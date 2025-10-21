"""Tests for the simplified decorators module."""

from __future__ import annotations

import asyncio

import pytest

from hexdag.core.registry.decorators import (
    _ASYNC_IO_WRAPPER_MARKER,
    _create_async_io_wrapper,
    _snake_case,
    _wrap_adapter_async_methods,
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
from hexdag.core.registry.models import (
    ComponentType,  # Internal for tests
    NodeSubtype,
)
from hexdag.core.utils.async_warnings import _is_in_async_context


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

        # Use TOOL type for testing metadata since tools don't require Config
        @component(ComponentType.TOOL, namespace="test")
        def test_component():
            """Test component."""
            pass

        # Decorator should add attributes to the function
        assert hasattr(test_component, "_hexdag_type")
        assert test_component._hexdag_type == ComponentType.TOOL
        assert test_component._hexdag_name == "test_component"
        assert test_component._hexdag_namespace == "test"
        assert test_component._hexdag_description == "Test component."

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


# ============================================================================
# Async I/O Monitoring Tests
# ============================================================================


class TestAsyncIOWrapperMarker:
    """Test the async I/O wrapper marker functionality."""

    def test_marker_constant_exists(self) -> None:
        """Test that marker constant is defined."""
        assert isinstance(_ASYNC_IO_WRAPPER_MARKER, str)
        assert _ASYNC_IO_WRAPPER_MARKER == "_hexdag_async_io_wrapped"


class TestWrapAdapterAsyncMethods:
    """Test _wrap_adapter_async_methods function."""

    def test_wraps_async_methods(self) -> None:
        """Test that async methods are wrapped."""

        class TestAdapter:
            async def aget_data(self) -> str:
                return "data"

            async def aset_data(self, value: str) -> None:
                pass

            def sync_method(self) -> str:
                return "sync"

        _wrap_adapter_async_methods(TestAdapter)

        # Async methods should be wrapped
        assert hasattr(TestAdapter.aget_data, _ASYNC_IO_WRAPPER_MARKER)
        assert hasattr(TestAdapter.aset_data, _ASYNC_IO_WRAPPER_MARKER)

        # Sync methods should not be wrapped
        assert not hasattr(TestAdapter.sync_method, _ASYNC_IO_WRAPPER_MARKER)

    def test_prevents_double_wrapping(self) -> None:
        """Test that methods are not double-wrapped."""

        class TestAdapter:
            async def aget_data(self) -> str:
                return "data"

        # Wrap once
        _wrap_adapter_async_methods(TestAdapter)
        first_method = TestAdapter.aget_data

        # Wrap again
        _wrap_adapter_async_methods(TestAdapter)
        second_method = TestAdapter.aget_data

        # Should be the same method object
        assert first_method is second_method

    def test_wraps_protocol_methods(self) -> None:
        """Test that async protocol methods are wrapped."""

        class TestAdapter:
            async def _private_method(self) -> str:
                return "private"

            async def aget_schema(self) -> dict:
                return {}

            async def aexecute_query(self, sql: str) -> list:
                return []

            async def acall_tool(self, name: str) -> str:
                return ""

        _wrap_adapter_async_methods(TestAdapter)

        # Private methods should not be wrapped (unless they match protocol patterns)
        assert not hasattr(TestAdapter._private_method, _ASYNC_IO_WRAPPER_MARKER)

        # Protocol methods should be wrapped
        assert hasattr(TestAdapter.aget_schema, _ASYNC_IO_WRAPPER_MARKER)
        assert hasattr(TestAdapter.aexecute_query, _ASYNC_IO_WRAPPER_MARKER)
        assert hasattr(TestAdapter.acall_tool, _ASYNC_IO_WRAPPER_MARKER)

    async def test_wrapped_method_still_works(self) -> None:
        """Test that wrapped methods still function correctly."""

        class TestAdapter:
            async def aget_value(self) -> int:
                await asyncio.sleep(0.001)
                return 42

        _wrap_adapter_async_methods(TestAdapter)

        adapter_instance = TestAdapter()
        result = await adapter_instance.aget_value()

        assert result == 42


class TestCreateAsyncIOWrapper:
    """Test _create_async_io_wrapper function."""

    async def test_wrapper_preserves_functionality(self) -> None:
        """Test that wrapper preserves original function behavior."""

        async def original_func(x: int, y: int) -> int:
            return x + y

        wrapped = _create_async_io_wrapper(
            original_func, "original_func", "TestClass", _is_in_async_context
        )

        result = await wrapped(5, 10)
        assert result == 15

    async def test_wrapper_preserves_metadata(self) -> None:
        """Test that wrapper preserves function metadata."""

        async def original_func() -> str:
            """Original docstring."""
            return "value"

        wrapped = _create_async_io_wrapper(
            original_func, "original_func", "TestClass", _is_in_async_context
        )

        assert wrapped.__name__ == "original_func"
        assert wrapped.__doc__ == "Original docstring."

    async def test_wrapper_has_marker(self) -> None:
        """Test that wrapper has the marker attribute."""

        async def original_func() -> None:
            pass

        wrapped = _create_async_io_wrapper(
            original_func, "original_func", "TestClass", _is_in_async_context
        )

        assert hasattr(wrapped, _ASYNC_IO_WRAPPER_MARKER)
        assert getattr(wrapped, _ASYNC_IO_WRAPPER_MARKER) is True

    async def test_wrapper_handles_exceptions(self) -> None:
        """Test that wrapper properly propagates exceptions."""

        async def failing_func() -> None:
            raise ValueError("Test error")

        wrapped = _create_async_io_wrapper(
            failing_func, "failing_func", "TestClass", _is_in_async_context
        )

        with pytest.raises(ValueError, match="Test error"):
            await wrapped()

    async def test_wrapper_handles_kwargs(self) -> None:
        """Test that wrapper handles keyword arguments."""

        async def func_with_kwargs(a: int, b: int = 10, c: int = 20) -> int:
            return a + b + c

        wrapped = _create_async_io_wrapper(
            func_with_kwargs, "func_with_kwargs", "TestClass", _is_in_async_context
        )

        result1 = await wrapped(5)
        assert result1 == 35  # 5 + 10 + 20

        result2 = await wrapped(5, b=15)
        assert result2 == 40  # 5 + 15 + 20

        result3 = await wrapped(5, c=25)
        assert result3 == 40  # 5 + 10 + 25


class TestAdapterDecoratorAsyncMonitoring:
    """Test adapter decorator with async I/O monitoring."""

    def test_adapter_default_wraps_methods(self) -> None:
        """Test that adapter decorator wraps async methods by default."""

        @adapter("test_port", name="test")
        class TestAdapter:
            async def aget_data(self) -> str:
                return "data"

        assert hasattr(TestAdapter.aget_data, _ASYNC_IO_WRAPPER_MARKER)

    def test_adapter_warn_sync_io_false(self) -> None:
        """Test that warn_sync_io=False prevents wrapping."""

        @adapter("test_port", name="test", warn_sync_io=False)
        class TestAdapter:
            async def aget_data(self) -> str:
                return "data"

        assert not hasattr(TestAdapter.aget_data, _ASYNC_IO_WRAPPER_MARKER)

    async def test_adapter_wrapped_methods_work(self) -> None:
        """Test that wrapped methods in adapter still work."""

        @adapter("test_port", name="test")
        class TestAdapter:
            async def aget_value(self, multiplier: int) -> int:
                await asyncio.sleep(0.001)
                return 42 * multiplier

        instance = TestAdapter()
        result = await instance.aget_value(2)

        assert result == 84

    def test_adapter_metadata_preserved(self) -> None:
        """Test that adapter decorator preserves class metadata."""

        @adapter("test_port", name="my_adapter", description="Test adapter")
        class TestAdapter:
            """Original docstring."""

            async def aget_data(self) -> str:
                return "data"

        assert TestAdapter._hexdag_name == "my_adapter"  # type: ignore[attr-defined]
        assert TestAdapter._hexdag_implements_port == "test_port"  # type: ignore[attr-defined]
        assert TestAdapter._hexdag_description == "Test adapter"  # type: ignore[attr-defined]

    def test_adapter_multiple_async_methods(self) -> None:
        """Test adapter with multiple async methods."""

        @adapter("database", name="test_db")
        class TestDatabaseAdapter:
            async def aexecute_query(self, sql: str) -> list:
                return []

            async def aget_schema(self) -> dict:
                return {}

            async def aconnect(self) -> None:
                pass

            def sync_helper(self) -> str:
                return "helper"

        # All async methods should be wrapped
        assert hasattr(TestDatabaseAdapter.aexecute_query, _ASYNC_IO_WRAPPER_MARKER)
        assert hasattr(TestDatabaseAdapter.aget_schema, _ASYNC_IO_WRAPPER_MARKER)
        assert hasattr(TestDatabaseAdapter.aconnect, _ASYNC_IO_WRAPPER_MARKER)

        # Sync method should not be wrapped
        assert not hasattr(TestDatabaseAdapter.sync_helper, _ASYNC_IO_WRAPPER_MARKER)


class TestAdapterDecoratorEdgeCases:
    """Test edge cases for adapter decorator async monitoring."""

    def test_adapter_with_no_async_methods(self) -> None:
        """Test adapter with only sync methods."""

        @adapter("test_port", name="sync_only")
        class SyncOnlyAdapter:
            def get_data(self) -> str:
                return "data"

            def set_data(self, value: str) -> None:
                pass

        # Should not crash, and no methods should be wrapped
        assert not hasattr(SyncOnlyAdapter.get_data, _ASYNC_IO_WRAPPER_MARKER)
        assert not hasattr(SyncOnlyAdapter.set_data, _ASYNC_IO_WRAPPER_MARKER)

    def test_adapter_with_properties(self) -> None:
        """Test adapter with properties."""

        @adapter("test_port", name="with_props")
        class AdapterWithProps:
            @property
            def value(self) -> str:
                return "value"

            async def aget_data(self) -> str:
                return self.value

        # Property should not be wrapped
        # Async method should be wrapped
        assert hasattr(AdapterWithProps.aget_data, _ASYNC_IO_WRAPPER_MARKER)

    def test_adapter_with_classmethod(self) -> None:
        """Test adapter with classmethods and staticmethods."""

        @adapter("test_port", name="with_classmethods")
        class AdapterWithClassMethods:
            @classmethod
            async def afrom_config(cls, config: dict) -> AdapterWithClassMethods:
                return cls()

            @staticmethod
            async def avalidate(data: dict) -> bool:
                return True

            async def aget_data(self) -> str:
                return "data"

        # All async methods should be wrapped (including class/static methods)
        assert hasattr(AdapterWithClassMethods.aget_data, _ASYNC_IO_WRAPPER_MARKER)


class TestAsyncIntegration:
    """Integration tests for decorator async monitoring."""

    async def test_full_adapter_lifecycle(self) -> None:
        """Test complete adapter lifecycle with async monitoring."""

        @adapter("database", name="integration_db")
        class IntegrationDatabaseAdapter:
            def __init__(self) -> None:
                self.connected = False

            async def aconnect(self) -> None:
                await asyncio.sleep(0.001)
                self.connected = True

            async def aexecute(self, sql: str) -> list[dict]:
                if not self.connected:
                    raise RuntimeError("Not connected")
                await asyncio.sleep(0.001)
                return [{"result": sql}]

            async def aclose(self) -> None:
                await asyncio.sleep(0.001)
                self.connected = False

        # Create instance and run through lifecycle
        db = IntegrationDatabaseAdapter()

        assert not db.connected

        await db.aconnect()
        assert db.connected

        results = await db.aexecute("SELECT * FROM test")
        assert len(results) == 1
        assert results[0]["result"] == "SELECT * FROM test"

        await db.aclose()
        assert not db.connected

    async def test_adapter_with_inheritance(self) -> None:
        """Test that wrapping works with adapter inheritance."""

        @adapter("database", name="base_db")
        class BaseDatabaseAdapter:
            async def aconnect(self) -> None:
                pass

        @adapter("database", name="derived_db")
        class DerivedDatabaseAdapter(BaseDatabaseAdapter):
            async def aexecute(self, sql: str) -> list:
                return []

        # Both async methods should be wrapped
        assert hasattr(DerivedDatabaseAdapter.aconnect, _ASYNC_IO_WRAPPER_MARKER)
        assert hasattr(DerivedDatabaseAdapter.aexecute, _ASYNC_IO_WRAPPER_MARKER)
