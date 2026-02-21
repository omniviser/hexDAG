"""Tests for native tool calling support."""

import pytest

from hexdag.builtin.adapters.openai.openai_adapter import OpenAIAdapter
from hexdag.core.ports.llm import LLMResponse, Message, MessageList, ToolCall


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
