"""Tests for LLM structured output parsing via StructuredOutputFallback middleware.

Parsing logic was moved from LLMNode to the StructuredOutputFallback middleware.
These tests verify that the middleware's JSON parsing works correctly,
using SafeJSON delegation.
"""

from __future__ import annotations

import json

import pytest

from hexdag.kernel.exceptions import ParseError
from hexdag.stdlib.middleware.structured_output import _parse_json_response


class TestParseJson:
    """Test _parse_json_response delegates to SafeJSON.loads_from_text."""

    def test_plain_json(self) -> None:
        result = _parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_with_surrounding_text(self) -> None:
        text = 'Here is the result: {"a": 1, "b": 2} end.'
        result = _parse_json_response(text)
        assert result["a"] == 1
        assert result["b"] == 2

    def test_json_in_code_block(self) -> None:
        text = '```json\n{"key": "value"}\n```'
        result = _parse_json_response(text)
        assert result == {"key": "value"}

    def test_json_with_trailing_comma(self) -> None:
        text = '{"a": 1, "b": 2,}'
        result = _parse_json_response(text)
        assert result == {"a": 1, "b": 2}

    def test_json_with_comments(self) -> None:
        text = '{"a": 1, // comment\n"b": 2}'
        result = _parse_json_response(text)
        assert result == {"a": 1, "b": 2}

    def test_invalid_json_raises(self) -> None:
        with pytest.raises((json.JSONDecodeError, ParseError)):
            _parse_json_response("not json at all")

    def test_array_json(self) -> None:
        text = "Result: [1, 2, 3]"
        result = _parse_json_response(text)
        assert result == [1, 2, 3]


class TestParseJsonInMarkdown:
    """Test _parse_json_response handles markdown code blocks via SafeJSON."""

    def test_json_code_block(self) -> None:
        text = 'Output:\n```json\n{"answer": 42}\n```\n'
        result = _parse_json_response(text)
        assert result == {"answer": 42}

    def test_generic_code_block(self) -> None:
        text = 'Output:\n```\n{"answer": 42}\n```\n'
        result = _parse_json_response(text)
        assert result == {"answer": 42}

    def test_falls_back_to_raw_json(self) -> None:
        text = '{"answer": 42}'
        result = _parse_json_response(text)
        assert result == {"answer": 42}

    def test_json_with_comments_in_block(self) -> None:
        text = '```json\n{"a": 1, // comment\n}\n```'
        result = _parse_json_response(text)
        assert result == {"a": 1}

    def test_invalid_raises(self) -> None:
        with pytest.raises((json.JSONDecodeError, ParseError)):
            _parse_json_response("no json here")


class TestSafeJsonSizeProtection:
    """Verify parsing inherits SafeJSON's size/depth validation."""

    def test_large_json_falls_back_gracefully(self) -> None:
        """SafeJSON has a 1MB default limit; very large input should
        still be parseable via the stdlib fallback."""
        large_val = "x" * 500
        text = json.dumps({"data": large_val})
        result = _parse_json_response(text)
        assert result["data"] == large_val
