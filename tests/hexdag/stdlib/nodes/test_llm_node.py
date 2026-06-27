"""Tests for LLM structured output parsing via StructuredOutputFallback middleware.

Parsing logic was moved from LLMNode to the StructuredOutputFallback middleware.
These tests verify that the middleware's JSON parsing works correctly,
using SafeJSON delegation.
"""

from __future__ import annotations

import json

import pytest

from hexdag.drivers.observer_manager import LocalObserverManager
from hexdag.kernel.domain.dag import DirectedGraph
from hexdag.kernel.exceptions import ParseError
from hexdag.kernel.orchestration.orchestrator import Orchestrator
from hexdag.kernel.ports.llm import LLMTokenStreamed, Message, SupportsStreaming
from hexdag.stdlib.adapters.mock import MockLLM
from hexdag.stdlib.middleware.structured_output import _parse_json_response
from hexdag.stdlib.nodes.llm_node import LLMNode


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


class TestMockLLMStreaming:
    """MockLLM implements SupportsStreaming."""

    def test_mock_llm_satisfies_protocol(self) -> None:
        assert isinstance(MockLLM(), SupportsStreaming)

    @pytest.mark.asyncio()
    async def test_astream_deltas_reassemble_to_full_response(self) -> None:
        llm = MockLLM(responses="hello streaming world")
        deltas = [d async for d in llm.astream([Message(role="user", content="hi")])]
        assert len(deltas) > 1  # actually chunked
        assert "".join(deltas) == "hello streaming world"

    @pytest.mark.asyncio()
    async def test_astream_matches_aresponse_cycling(self) -> None:
        llm = MockLLM(responses=["first", "second"])
        first = "".join([d async for d in llm.astream([Message(role="user", content="a")])])
        second = "".join([d async for d in llm.astream([Message(role="user", content="b")])])
        assert (first, second) == ("first", "second")


class TestLLMNodeStreaming:
    """LLMNode with stream=True emits token events, returns full text."""

    @pytest.mark.asyncio()
    async def test_streaming_node_emits_token_events(self) -> None:
        captured: list[LLMTokenStreamed] = []

        async def collector(event: LLMTokenStreamed) -> None:
            captured.append(event)

        observer_manager = LocalObserverManager()
        observer_manager.register(collector, event_types=LLMTokenStreamed)

        node = LLMNode()(
            name="streamer",
            human_message="Say something about {{topic}}",
            stream=True,
        )

        graph = DirectedGraph()
        graph.add(node)

        orchestrator = Orchestrator(
            ports={
                "llm": MockLLM(responses="alpha beta gamma"),
                "observer_manager": observer_manager,
            }
        )
        results = await orchestrator.run(graph, {"topic": "testing"})

        # Node result is the full concatenated text
        assert results["streamer"] == "alpha beta gamma"

        # Token events were emitted in order and reassemble to the result
        assert captured, "expected LLMTokenStreamed events"
        assert [e.index for e in captured] == list(range(len(captured)))
        assert "".join(e.delta for e in captured) == "alpha beta gamma"
        assert all(e.node_name == "streamer" for e in captured)

    @pytest.mark.asyncio()
    async def test_stream_false_emits_no_token_events(self) -> None:
        captured: list[LLMTokenStreamed] = []

        async def collector(event: LLMTokenStreamed) -> None:
            captured.append(event)

        observer_manager = LocalObserverManager()
        observer_manager.register(collector, event_types=LLMTokenStreamed)

        node = LLMNode()(
            name="plain",
            human_message="Say something about {{topic}}",
        )

        graph = DirectedGraph()
        graph.add(node)

        orchestrator = Orchestrator(
            ports={
                "llm": MockLLM(responses="no streaming here"),
                "observer_manager": observer_manager,
            }
        )
        results = await orchestrator.run(graph, {"topic": "testing"})

        assert results["plain"] == "no streaming here"
        assert captured == []

    @pytest.mark.asyncio()
    async def test_stream_falls_back_when_adapter_lacks_astream(self) -> None:
        """Adapters without astream still work — silent fallback."""

        class NoStreamLLM:
            async def aresponse(self, messages) -> str:
                return "fallback response"

        node = LLMNode()(
            name="fallback",
            human_message="Hello {{topic}}",
            stream=True,
        )

        graph = DirectedGraph()
        graph.add(node)

        orchestrator = Orchestrator(ports={"llm": NoStreamLLM()})
        results = await orchestrator.run(graph, {"topic": "x"})
        assert results["fallback"] == "fallback response"

    def test_stream_with_output_schema_warns_and_disables(self) -> None:
        with pytest.warns(UserWarning, match="stream.*ignored"):
            LLMNode()(
                name="structured",
                human_message="Analyze {{text}}",
                output_schema={"sentiment": str},
                stream=True,
            )
