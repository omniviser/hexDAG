"""Tests for PipelineResult and resolve_output."""

from __future__ import annotations

import pytest

from hexdag.kernel.domain.pipeline_result import PipelineResult, resolve_output

# ---------------------------------------------------------------------------
# resolve_output
# ---------------------------------------------------------------------------


class TestResolveOutput:
    def test_simple_field(self) -> None:
        results = {"router": {"action": "counter", "text": "hi"}}
        out = resolve_output(results, {"action": "router.action"})
        assert out == {"action": "counter"}

    def test_nested_field(self) -> None:
        results = {"extraction": {"data": {"rate": 4500, "currency": "USD"}}}
        out = resolve_output(results, {"rate": "extraction.data.rate"})
        assert out == {"rate": 4500}

    def test_missing_node_returns_none(self) -> None:
        results = {"router": {"action": "counter"}}
        out = resolve_output(results, {"x": "nonexistent.field"})
        assert out == {"x": None}

    def test_missing_field_returns_none(self) -> None:
        results = {"router": {"action": "counter"}}
        out = resolve_output(results, {"x": "router.missing"})
        assert out == {"x": None}

    def test_multiple_fields(self) -> None:
        results = {
            "router": {"action": "counter", "body": "email text"},
            "extraction": {"confidence": 0.95},
        }
        out = resolve_output(
            results,
            {
                "action": "router.action",
                "email_body": "router.body",
                "confidence": "extraction.confidence",
            },
        )
        assert out == {"action": "counter", "email_body": "email text", "confidence": 0.95}

    def test_empty_mapping(self) -> None:
        results = {"router": {"action": "counter"}}
        out = resolve_output(results, {})
        assert out == {}


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------


class TestPipelineResult:
    def test_basic_creation(self) -> None:
        result = PipelineResult(
            node_results={"a": 1, "b": 2},
            output={"x": "hello"},
            pipeline_name="test",
        )
        assert result.output == {"x": "hello"}
        assert result.node_results == {"a": 1, "b": 2}
        assert result.pipeline_name == "test"

    def test_dict_style_access(self) -> None:
        """Backwards-compatible dict-style access works."""
        result = PipelineResult(node_results={"router": {"action": "counter"}})
        assert result["router"] == {"action": "counter"}
        assert "router" in result
        assert len(result) == 1
        assert list(result.keys()) == ["router"]

    def test_get_method(self) -> None:
        result = PipelineResult(node_results={"a": 1})
        assert result.get("a") == 1
        assert result.get("missing", "default") == "default"

    def test_iteration(self) -> None:
        result = PipelineResult(node_results={"a": 1, "b": 2})
        assert set(result) == {"a", "b"}

    def test_items(self) -> None:
        result = PipelineResult(node_results={"a": 1})
        assert list(result.items()) == [("a", 1)]

    def test_default_empty_output(self) -> None:
        result = PipelineResult(node_results={"a": 1})
        assert result.output == {}
        assert result.pipeline_name == ""

    def test_getitem_raises_keyerror(self) -> None:
        result = PipelineResult(node_results={"a": 1})
        with pytest.raises(KeyError):
            _ = result["missing"]
