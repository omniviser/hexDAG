"""Tests for Ollama adapter."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from hexdag.kernel.ports.llm import Message
from hexdag.stdlib.adapters.ollama.ollama_adapter import OllamaAdapter


@pytest.fixture
def ollama_adapter():
    """Create Ollama adapter for testing."""
    return OllamaAdapter(
        model="qwen3.5:9b",
        base_url="http://localhost:11434",
        temperature=0.0,
        max_tokens=1024,
    )


class TestInitialization:
    def test_default_values(self):
        adapter = OllamaAdapter()
        assert adapter.model == "qwen3.5:9b"
        assert adapter.base_url == "http://localhost:11434"
        assert adapter.temperature == 0.0
        assert adapter.max_tokens == 1024
        assert adapter.timeout == 120.0

    def test_custom_values(self, ollama_adapter):
        assert ollama_adapter.model == "qwen3.5:9b"
        assert ollama_adapter.base_url == "http://localhost:11434"

    def test_url_building(self, ollama_adapter):
        url = ollama_adapter._build_url()
        assert url == "http://localhost:11434/v1/chat/completions"

    def test_message_conversion(self, ollama_adapter):
        messages = [
            Message(role="system", content="Be helpful"),
            Message(role="user", content="Hello"),
        ]
        result = ollama_adapter._convert_messages(messages)
        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "Be helpful"}
        assert result[1] == {"role": "user", "content": "Hello"}


@pytest.mark.asyncio
class TestAResponse:
    async def test_success(self, ollama_adapter):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from Ollama!"}}]
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        ollama_adapter._client = mock_client

        messages = [Message(role="user", content="Hello")]
        result = await ollama_adapter.aresponse(messages)

        assert result == "Hello from Ollama!"

    async def test_empty_choices(self, ollama_adapter):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"choices": []}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        ollama_adapter._client = mock_client

        messages = [Message(role="user", content="Hello")]
        result = await ollama_adapter.aresponse(messages)
        assert result is None

    async def test_http_error_returns_none(self, ollama_adapter):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_response.text = "Server Error"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=mock_response
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        ollama_adapter._client = mock_client

        messages = [Message(role="user", content="Hello")]
        result = await ollama_adapter.aresponse(messages)
        assert result is None


@pytest.mark.asyncio
class TestToolCalling:
    async def test_tool_calls(self, ollama_adapter):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "search",
                                    "arguments": {"query": "test"},
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        ollama_adapter._client = mock_client

        messages = [Message(role="user", content="Search")]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {},
                },
            }
        ]

        result = await ollama_adapter.aresponse_with_tools(messages, tools)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].id == "call_1"
        assert result.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_aclose(ollama_adapter):
    mock_client = AsyncMock()
    mock_client.is_closed = False
    ollama_adapter._client = mock_client

    await ollama_adapter.aclose()
    mock_client.aclose.assert_called_once()
    assert ollama_adapter._client is None
