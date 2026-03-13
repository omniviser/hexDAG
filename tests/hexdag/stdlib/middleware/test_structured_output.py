"""Tests for StructuredOutputFallback middleware."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from hexdag.kernel.ports.llm import (
    Message,
    MessageList,
    SupportsGeneration,
    SupportsStructuredOutput,
)
from hexdag.stdlib.middleware.structured_output import (
    StructuredOutputFallback,
    _build_schema_instruction,
    _inject_schema_instruction,
    _parse_json_response,
)


class _FakeAdapter:
    """Minimal adapter implementing SupportsGeneration."""

    def __init__(self, response: str = '{"result": "ok"}') -> None:
        self._response = response
        self.last_messages: MessageList | None = None

    async def aresponse(self, messages: MessageList) -> str:
        self.last_messages = messages
        return self._response


class _OutputModel(BaseModel):
    result: str
    score: float = 0.0


class TestStructuredOutputFallback:
    """Test the StructuredOutputFallback middleware."""

    def test_isinstance_supports_generation(self) -> None:
        fallback = StructuredOutputFallback(_FakeAdapter())
        assert isinstance(fallback, SupportsGeneration)

    def test_isinstance_supports_structured_output(self) -> None:
        fallback = StructuredOutputFallback(_FakeAdapter())
        assert isinstance(fallback, SupportsStructuredOutput)

    @pytest.mark.asyncio
    async def test_aresponse_passthrough(self) -> None:
        """aresponse() should forward to the inner adapter."""
        inner = _FakeAdapter(response="hello")
        fallback = StructuredOutputFallback(inner)
        messages = [Message(role="user", content="test")]
        result = await fallback.aresponse(messages)
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_aresponse_structured_with_pydantic_model(self) -> None:
        """aresponse_structured() should inject schema and parse JSON."""
        inner = _FakeAdapter(response='{"result": "ok", "score": 0.95}')
        fallback = StructuredOutputFallback(inner)
        messages = [Message(role="user", content="analyze this")]

        result = await fallback.aresponse_structured(messages, _OutputModel)

        assert result == {"result": "ok", "score": 0.95}
        # Verify schema instruction was injected into the message
        assert inner.last_messages is not None
        last_user_msg = [m for m in inner.last_messages if m.role == "user"][-1]
        assert "Output Format" in last_user_msg.content

    @pytest.mark.asyncio
    async def test_aresponse_structured_with_dict_schema(self) -> None:
        """aresponse_structured() should work with dict schemas."""
        inner = _FakeAdapter(response='{"name": "test"}')
        fallback = StructuredOutputFallback(inner)
        messages = [Message(role="user", content="test")]

        schema: dict[str, Any] = {
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        result = await fallback.aresponse_structured(messages, schema)
        assert result == {"name": "test"}

    @pytest.mark.asyncio
    async def test_aresponse_structured_json_in_markdown(self) -> None:
        """Should parse JSON from markdown code blocks."""
        inner = _FakeAdapter(response='Here is the result:\n```json\n{"result": "parsed"}\n```')
        fallback = StructuredOutputFallback(inner)
        messages = [Message(role="user", content="test")]

        result = await fallback.aresponse_structured(messages, {"properties": {}})
        assert result == {"result": "parsed"}

    def test_getattr_forwards_to_inner(self) -> None:
        """__getattr__ should forward to inner adapter."""
        inner = _FakeAdapter()
        inner.custom_attr = "custom_value"  # type: ignore[attr-defined]
        fallback = StructuredOutputFallback(inner)
        assert fallback.custom_attr == "custom_value"  # type: ignore[attr-defined]


class TestBuildSchemaInstruction:
    """Test the _build_schema_instruction helper."""

    def test_basic_schema(self) -> None:
        schema = {
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "description": "User age"},
            }
        }
        instruction = _build_schema_instruction(schema)
        assert "name: string" in instruction
        assert "age: integer - User age" in instruction
        assert "Output Format" in instruction

    def test_empty_properties(self) -> None:
        schema: dict[str, Any] = {"properties": {}}
        instruction = _build_schema_instruction(schema)
        assert "no specific fields defined" in instruction


class TestInjectSchemaInstruction:
    """Test the _inject_schema_instruction helper."""

    def test_appends_to_last_user_message(self) -> None:
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Analyze this"),
        ]
        result = _inject_schema_instruction(messages, "\n## Schema")
        assert result[-1].content == "Analyze this\n## Schema"
        assert result[0].content == "You are helpful."

    def test_creates_user_message_if_none(self) -> None:
        messages = [Message(role="system", content="sys")]
        result = _inject_schema_instruction(messages, "schema instruction")
        assert len(result) == 2
        assert result[-1].role == "user"

    def test_empty_messages(self) -> None:
        result = _inject_schema_instruction([], "instruction")
        assert len(result) == 1
        assert result[0].role == "user"


class TestParseJsonResponse:
    """Test the _parse_json_response helper."""

    def test_plain_json(self) -> None:
        assert _parse_json_response('{"key": "value"}') == {"key": "value"}

    def test_json_in_code_block(self) -> None:
        text = '```json\n{"key": "value"}\n```'
        assert _parse_json_response(text) == {"key": "value"}

    def test_json_with_surrounding_text(self) -> None:
        text = 'Here: {"a": 1} done.'
        result = _parse_json_response(text)
        assert result["a"] == 1

    def test_invalid_json_raises(self) -> None:
        from hexdag.kernel.exceptions import ParseError

        with pytest.raises((json.JSONDecodeError, ParseError)):
            _parse_json_response("not json")
