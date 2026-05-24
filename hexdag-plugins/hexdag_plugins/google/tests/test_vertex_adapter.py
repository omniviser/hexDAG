"""Tests for Vertex AI (Google Gemini) adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from hexdag.kernel.ports.llm import Message

from hexdag_plugins.google.adapters.vertex import (
    VertexAIAdapter,
    _pydantic_schema_to_gemini,
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

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        vertex_adapter._client = mock_client

        messages = [Message(role="user", content="Hello")]
        result = await vertex_adapter.aresponse(messages)

        assert result == "Hello from Gemini!"
        mock_client.post.assert_called_once()

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

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        vertex_adapter._client = mock_client

        messages = [Message(role="user", content="Think")]
        result = await vertex_adapter.aresponse(messages)
        assert result == "Actual answer"

    async def test_http_error_returns_none(self, vertex_adapter):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=mock_response
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        vertex_adapter._client = mock_client

        messages = [Message(role="user", content="Hello")]
        result = await vertex_adapter.aresponse(messages)
        assert result is None


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

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        vertex_adapter._client = mock_client

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


@pytest.mark.asyncio
async def test_url_building(vertex_adapter):
    url = vertex_adapter._build_url()
    assert "gemini-2.0-flash:generateContent" in url
    assert "key=test-api-key" in url

    url_fb = vertex_adapter._build_url(fallback=True)
    assert "key=test-api-key" in url_fb
