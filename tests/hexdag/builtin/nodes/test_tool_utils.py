"""Tests for native tool calling support."""

import pytest

from hexdag.builtin.adapters.llm.openai_adapter import OpenAIAdapter
from hexdag.builtin.nodes.tool_utils import ToolDefinition, ToolParameter, ToolSchemaConverter
from hexdag.core.ports.llm import LLMResponse, Message, MessageList, ToolCall


class TestToolSchemaConverter:
    """Test tool schema conversion to various formats."""

    def test_to_openai_schema(self):
        """Test conversion to OpenAI format."""
        tool_def = ToolDefinition(
            name="search_web",
            simplified_description="Search the web",
            detailed_description="Search the web for information on a given query",
            parameters=[
                ToolParameter(
                    name="query",
                    description="The search query",
                    param_type="str",
                    required=True,
                ),
                ToolParameter(
                    name="max_results",
                    description="Maximum number of results",
                    param_type="int",
                    required=False,
                    default=10,
                ),
            ],
            examples=["search_web(query='Python programming')"],
        )

        schema = ToolSchemaConverter.to_openai_schema(tool_def)

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search_web"
        assert (
            schema["function"]["description"] == "Search the web for information on a given query"
        )
        assert "parameters" in schema["function"]
        assert schema["function"]["parameters"]["type"] == "object"
        assert "query" in schema["function"]["parameters"]["properties"]
        assert "max_results" in schema["function"]["parameters"]["properties"]
        assert schema["function"]["parameters"]["required"] == ["query"]

    def test_to_anthropic_schema(self):
        """Test conversion to Anthropic format."""
        tool_def = ToolDefinition(
            name="calculate",
            simplified_description="Calculate math expression",
            detailed_description="Evaluate a mathematical expression and return the result",
            parameters=[
                ToolParameter(
                    name="expression",
                    description="The math expression to evaluate",
                    param_type="str",
                    required=True,
                )
            ],
            examples=["calculate(expression='2 + 2')"],
        )

        schema = ToolSchemaConverter.to_anthropic_schema(tool_def)

        assert schema["name"] == "calculate"
        assert schema["description"] == "Evaluate a mathematical expression and return the result"
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"
        assert "expression" in schema["input_schema"]["properties"]
        assert schema["input_schema"]["required"] == ["expression"]

    def test_python_type_mapping(self):
        """Test Python type to JSON Schema type mapping."""
        tool_def = ToolDefinition(
            name="test_types",
            simplified_description="Test all types",
            detailed_description="Test all parameter types",
            parameters=[
                ToolParameter(name="str_param", description="String", param_type="str"),
                ToolParameter(name="int_param", description="Integer", param_type="int"),
                ToolParameter(name="float_param", description="Float", param_type="float"),
                ToolParameter(name="bool_param", description="Boolean", param_type="bool"),
                ToolParameter(name="list_param", description="List", param_type="list"),
                ToolParameter(name="dict_param", description="Dict", param_type="dict"),
            ],
        )

        schema = ToolSchemaConverter.to_openai_schema(tool_def)
        props = schema["function"]["parameters"]["properties"]

        assert props["str_param"]["type"] == "string"
        assert props["int_param"]["type"] == "integer"
        assert props["float_param"]["type"] == "number"
        assert props["bool_param"]["type"] == "boolean"
        assert props["list_param"]["type"] == "array"
        assert props["dict_param"]["type"] == "object"


class TestLLMResponse:
    """Test LLMResponse model."""

    def test_llm_response_without_tools(self):
        """Test LLMResponse without tool calls."""
        response = LLMResponse(content="Hello, world!", tool_calls=None, finish_reason="stop")

        assert response.content == "Hello, world!"
        assert response.tool_calls is None
        assert response.finish_reason == "stop"

    def test_llm_response_with_tools(self):
        """Test LLMResponse with tool calls."""
        response = LLMResponse(
            content="Let me search for that",
            tool_calls=[
                ToolCall(id="call_123", name="search_web", arguments={"query": "Python"}),
                ToolCall(id="call_456", name="calculate", arguments={"expression": "2+2"}),
            ],
            finish_reason="tool_calls",
        )

        assert response.content == "Let me search for that"
        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].name == "search_web"
        assert response.tool_calls[0].arguments == {"query": "Python"}
        assert response.finish_reason == "tool_calls"


@pytest.mark.skip(reason="Requires OpenAI API key")
class TestOpenAIAdapterWithTools:
    """Test OpenAI adapter native tool calling (requires API key)."""

    @pytest.mark.asyncio
    async def test_aresponse_with_tools(self):
        """Test native tool calling with OpenAI."""
        OpenAIAdapter(api_key="test-key")

        MessageList([Message(role="user", content="What is 2 + 2?")])

        # This would make an actual API call
        # response = await adapter.aresponse_with_tools(messages, tools)
        # assert response.content is not None or response.tool_calls is not None
