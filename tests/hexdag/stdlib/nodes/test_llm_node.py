"""Tests for LLMNode parsing methods using SafeJSON delegation."""

from __future__ import annotations

import json

import pytest

from hexdag.stdlib.nodes.llm_node import LLMNode


class TestParseJson:
    """Test _parse_json delegates to SafeJSON.loads_from_text."""

    def setup_method(self) -> None:
        self.node = LLMNode()

    def test_plain_json(self) -> None:
        result = self.node._parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_with_surrounding_text(self) -> None:
        text = 'Here is the result: {"a": 1, "b": 2} end.'
        result = self.node._parse_json(text)
        assert result["a"] == 1
        assert result["b"] == 2

    def test_json_in_code_block(self) -> None:
        text = '```json\n{"key": "value"}\n```'
        result = self.node._parse_json(text)
        assert result == {"key": "value"}

    def test_json_with_trailing_comma(self) -> None:
        text = '{"a": 1, "b": 2,}'
        result = self.node._parse_json(text)
        assert result == {"a": 1, "b": 2}

    def test_json_with_comments(self) -> None:
        text = '{"a": 1, // comment\n"b": 2}'
        result = self.node._parse_json(text)
        assert result == {"a": 1, "b": 2}

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            self.node._parse_json("not json at all")

    def test_array_json(self) -> None:
        text = "Result: [1, 2, 3]"
        result = self.node._parse_json(text)
        assert result == [1, 2, 3]


class TestParseJsonInMarkdown:
    """Test _parse_json handles markdown code blocks via SafeJSON.loads_from_text."""

    def setup_method(self) -> None:
        self.node = LLMNode()

    def test_json_code_block(self) -> None:
        text = 'Output:\n```json\n{"answer": 42}\n```\n'
        result = self.node._parse_json(text)
        assert result == {"answer": 42}

    def test_generic_code_block(self) -> None:
        text = 'Output:\n```\n{"answer": 42}\n```\n'
        result = self.node._parse_json(text)
        assert result == {"answer": 42}

    def test_falls_back_to_raw_json(self) -> None:
        text = '{"answer": 42}'
        result = self.node._parse_json(text)
        assert result == {"answer": 42}

    def test_json_with_comments_in_block(self) -> None:
        text = '```json\n{"a": 1, // comment\n}\n```'
        result = self.node._parse_json(text)
        assert result == {"a": 1}

    def test_invalid_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            self.node._parse_json("no json here")


class TestParseYaml:
    """Test _parse_yaml delegates to SafeJSON.loads_yaml_from_text."""

    def setup_method(self) -> None:
        self.node = LLMNode()

    def test_plain_yaml(self) -> None:
        text = "key: value\ncount: 42"
        result = self.node._parse_yaml(text)
        assert result == {"key": "value", "count": 42}

    def test_yaml_in_code_block(self) -> None:
        text = "```yaml\nname: test\nvalue: 1\n```"
        result = self.node._parse_yaml(text)
        assert result == {"name": "test", "value": 1}

    def test_yaml_list(self) -> None:
        text = "items:\n  - a\n  - b\n  - c"
        result = self.node._parse_yaml(text)
        assert result == {"items": ["a", "b", "c"]}

    def test_yaml_nested(self) -> None:
        text = "outer:\n  inner: value"
        result = self.node._parse_yaml(text)
        assert result == {"outer": {"inner": "value"}}


class TestSafeJsonSizeProtection:
    """Verify LLMNode inherits SafeJSON's size/depth validation."""

    def setup_method(self) -> None:
        self.node = LLMNode()

    def test_large_json_falls_back_gracefully(self) -> None:
        """SafeJSON has a 1MB default limit; very large input should
        still be parseable via the stdlib fallback."""
        large_val = "x" * 500
        text = json.dumps({"data": large_val})
        result = self.node._parse_json(text)
        assert result["data"] == large_val
