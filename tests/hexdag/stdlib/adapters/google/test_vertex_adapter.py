"""Tests for Vertex AI (Google Gemini) adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from hexdag.kernel.ports.llm import Message, TokenUsage
from hexdag.stdlib.adapters.google.vertex_adapter import (
    VertexAIAdapter,
    _active_instances,
    _pydantic_schema_to_gemini,
    aclose_all_vertex_clients,
)


@pytest.fixture
def vertex_adapter():
    """Create Vertex AI adapter with API key auth for testing."""
    with patch.dict(
        "os.environ",
        {"GOOGLE_API_KEY": "test-api-key"},
        clear=False,
    ):
        adapter = VertexAIAdapter(
            api_key="test-api-key",
            model="gemini-2.0-flash",
            temperature=0.0,
            max_tokens=1024,
            cache_ttl=0,  # Disable cache for tests
        )
        # Force API key mode (not ADC)
        adapter._use_adc = False
        adapter._base_url = "https://aiplatform.googleapis.com/v1/publishers/google/models"
        return adapter


def _mock_client(response: MagicMock) -> AsyncMock:
    client = AsyncMock()
    client.post = AsyncMock(return_value=response)
    client.is_closed = False
    return client


class TestInitialization:
    def test_api_key_auth(self, vertex_adapter):
        assert vertex_adapter.api_key == "test-api-key"
        assert vertex_adapter.model == "gemini-2.0-flash"
        assert vertex_adapter.temperature == 0.0
        assert vertex_adapter.max_tokens == 1024

    def test_missing_credentials_raises(self):
        with (
            patch.object(VertexAIAdapter, "_try_init_adc", return_value=False),
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="No authentication"),
        ):
            VertexAIAdapter(api_key="", model="gemini-2.0-flash")

    def test_thinking_level_config(self, vertex_adapter):
        config = vertex_adapter._build_generation_config()
        assert config["thinkingConfig"] == {"thinkingLevel": "low"}

    def test_thinking_level_none(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "key"}, clear=False):
            adapter = VertexAIAdapter(api_key="key", thinking_level=None, cache_ttl=0)
            adapter._use_adc = False
            config = adapter._build_generation_config()
            assert "thinkingConfig" not in config

    def test_instance_tracked(self, vertex_adapter):
        assert vertex_adapter in _active_instances


class TestMessageConversion:
    def test_user_message(self, vertex_adapter):
        messages = [Message(role="user", content="Hello")]
        contents, sys = vertex_adapter._convert_messages(messages)
        assert len(contents) == 1
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"] == [{"text": "Hello"}]
        assert sys is None

    def test_system_message_extracted(self, vertex_adapter):
        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hello"),
        ]
        contents, sys = vertex_adapter._convert_messages(messages)
        assert len(contents) == 1
        assert sys == "You are helpful"

    def test_assistant_becomes_model(self, vertex_adapter):
        messages = [Message(role="assistant", content="Hi there")]
        contents, _ = vertex_adapter._convert_messages(messages)
        assert contents[0]["role"] == "model"


@pytest.mark.asyncio
class TestAResponse:
    async def test_success(self, vertex_adapter):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello from Gemini!"}]},
                    "finishReason": "STOP",
                }
            ]
        }
        vertex_adapter._client = _mock_client(mock_response)

        messages = [Message(role="user", content="Hello")]
        result = await vertex_adapter.aresponse(messages)

        assert result == "Hello from Gemini!"
        vertex_adapter._client.post.assert_called_once()

    async def test_skips_thinking_parts(self, vertex_adapter):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "thinking...", "thought": True},
                            {"text": "Actual answer"},
                        ]
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        vertex_adapter._client = _mock_client(mock_response)

        messages = [Message(role="user", content="Think")]
        result = await vertex_adapter.aresponse(messages)
        assert result == "Actual answer"

    async def test_http_error_raises(self, vertex_adapter):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=mock_response
        )
        vertex_adapter._client = _mock_client(mock_response)

        messages = [Message(role="user", content="Hello")]
        with pytest.raises(httpx.HTTPStatusError):
            await vertex_adapter.aresponse(messages)


@pytest.mark.asyncio
class TestUsageTracking:
    async def test_usage_captured(self, vertex_adapter):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hi"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 12,
                "candidatesTokenCount": 5,
                "totalTokenCount": 17,
            },
        }
        vertex_adapter._client = _mock_client(mock_response)

        assert vertex_adapter.get_last_usage() is None
        await vertex_adapter.aresponse([Message(role="user", content="Hello")])

        usage = vertex_adapter.get_last_usage()
        assert usage == TokenUsage(input_tokens=12, output_tokens=5, total_tokens=17)

    async def test_no_usage_metadata_leaves_last_usage(self, vertex_adapter):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hi"}]},
                    "finishReason": "STOP",
                }
            ]
        }
        vertex_adapter._client = _mock_client(mock_response)

        await vertex_adapter.aresponse([Message(role="user", content="Hello")])
        assert vertex_adapter.get_last_usage() is None


@pytest.mark.asyncio
class TestToolCalling:
    async def test_tool_calls_parsed(self, vertex_adapter):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "search",
                                    "args": {"query": "cats"},
                                }
                            }
                        ]
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        vertex_adapter._client = _mock_client(mock_response)

        messages = [Message(role="user", content="Search for cats")]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        result = await vertex_adapter.aresponse_with_tools(messages, tools)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"query": "cats"}


class TestToolConversion:
    def test_tool_conversion(self, vertex_adapter):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
        result = vertex_adapter._convert_tools_to_gemini(tools)
        assert len(result) == 1
        decls = result[0]["functionDeclarations"]
        assert len(decls) == 1
        assert decls[0]["name"] == "get_weather"


class TestSchemaConversion:
    def test_simple_object(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        result = _pydantic_schema_to_gemini(schema)
        assert result["type"] == "OBJECT"
        assert result["properties"]["name"]["type"] == "STRING"
        assert result["properties"]["age"]["type"] == "INTEGER"
        assert result["required"] == ["name"]

    def test_nullable_field(self):
        schema = {
            "type": "object",
            "properties": {"name": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
        }
        result = _pydantic_schema_to_gemini(schema)
        assert result["properties"]["name"]["type"] == "STRING"
        assert result["properties"]["name"]["nullable"] is True

    def test_array_type(self):
        schema = {
            "type": "array",
            "items": {"type": "string"},
        }
        result = _pydantic_schema_to_gemini(schema)
        assert result["type"] == "ARRAY"
        assert result["items"]["type"] == "STRING"

    def test_ref_resolution(self):
        schema = {
            "type": "object",
            "properties": {
                "item": {"$ref": "#/$defs/Item"},
            },
            "$defs": {
                "Item": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                }
            },
        }
        result = _pydantic_schema_to_gemini(schema)
        assert result["properties"]["item"]["type"] == "OBJECT"
        assert result["properties"]["item"]["properties"]["id"]["type"] == "INTEGER"


def test_url_building(vertex_adapter):
    url = vertex_adapter._build_url()
    assert "gemini-2.0-flash:generateContent" in url
    assert "key=test-api-key" in url

    url_fb = vertex_adapter._build_url(fallback=True)
    assert "key=test-api-key" in url_fb


@pytest.mark.asyncio
async def test_aclose_all_vertex_clients(vertex_adapter):
    mock_client = AsyncMock()
    mock_client.is_closed = False
    vertex_adapter._client = mock_client

    await aclose_all_vertex_clients()

    mock_client.aclose.assert_called_once()
    assert vertex_adapter._client is None
    assert vertex_adapter not in _active_instances
