"""Tests for ObservableLLM middleware."""

from __future__ import annotations

from typing import Any

import pytest

from hexdag.kernel.ports.llm import (
    MessageList,
    SupportsGeneration,
    SupportsStructuredOutput,
)
from hexdag.stdlib.middleware.observable import ObservableLLM


class _FakeAdapter:
    """Minimal adapter for testing."""

    def __init__(self, response: str = "hello") -> None:
        self._response = response

    async def aresponse(self, messages: MessageList) -> str:
        return self._response

    async def aresponse_structured(
        self, messages: MessageList, output_schema: Any
    ) -> dict[str, Any]:
        return {"result": self._response}


class TestObservableLLM:
    """Test ObservableLLM middleware."""

    def test_isinstance_supports_generation(self) -> None:
        obs = ObservableLLM(_FakeAdapter())
        assert isinstance(obs, SupportsGeneration)

    def test_isinstance_supports_structured_output(self) -> None:
        obs = ObservableLLM(_FakeAdapter())
        assert isinstance(obs, SupportsStructuredOutput)

    @pytest.mark.asyncio
    async def test_aresponse_forwards(self) -> None:
        from hexdag.kernel.ports.llm import Message

        obs = ObservableLLM(_FakeAdapter(response="world"))
        messages = [Message(role="user", content="test")]
        result = await obs.aresponse(messages)
        assert result == "world"

    @pytest.mark.asyncio
    async def test_aresponse_structured_forwards(self) -> None:
        from hexdag.kernel.ports.llm import Message

        obs = ObservableLLM(_FakeAdapter(response="test"))
        messages = [Message(role="user", content="test")]
        result = await obs.aresponse_structured(messages, {"properties": {}})
        assert result == {"result": "test"}

    def test_getattr_forwards(self) -> None:
        inner = _FakeAdapter()
        inner.model = "gpt-4"  # type: ignore[attr-defined]
        obs = ObservableLLM(inner)
        assert obs.model == "gpt-4"  # type: ignore[attr-defined]

    def test_astream_absent_when_inner_not_streaming(self) -> None:
        """hasattr(obs, 'astream') is False when the inner adapter can't stream."""
        obs = ObservableLLM(_FakeAdapter())
        assert not hasattr(obs, "astream")

    @pytest.mark.asyncio
    async def test_astream_forwards_and_emits_event(self, monkeypatch) -> None:
        from hexdag.kernel.ports.llm import LLMPortCall, Message

        class _StreamingAdapter(_FakeAdapter):
            model = "gpt-4"

            async def astream(self, messages: MessageList):
                for chunk in ("hel", "lo"):
                    yield chunk

        events: list[Any] = []

        class _Mgr:
            async def notify(self, event: Any) -> None:
                events.append(event)

        monkeypatch.setattr(
            "hexdag.stdlib.middleware.observable.get_observer_manager", lambda: _Mgr()
        )

        obs = ObservableLLM(_StreamingAdapter())
        assert hasattr(obs, "astream")
        messages = [Message(role="user", content="hi")]
        collected = [delta async for delta in obs.astream(messages)]

        assert collected == ["hel", "lo"]
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, LLMPortCall)
        assert event.method == "astream"
        assert event.response == "hello"
        assert event.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_astream_emits_event_on_early_exit(self, monkeypatch) -> None:
        from hexdag.kernel.ports.llm import LLMPortCall, Message

        class _StreamingAdapter(_FakeAdapter):
            async def astream(self, messages: MessageList):
                for chunk in ("a", "b", "c"):
                    yield chunk

        events: list[Any] = []

        class _Mgr:
            async def notify(self, event: Any) -> None:
                events.append(event)

        monkeypatch.setattr(
            "hexdag.stdlib.middleware.observable.get_observer_manager", lambda: _Mgr()
        )

        obs = ObservableLLM(_StreamingAdapter())
        messages = [Message(role="user", content="hi")]
        gen = obs.astream(messages)
        first = await gen.__anext__()
        await gen.aclose()  # consumer stops early

        assert first == "a"
        assert len(events) == 1
        assert isinstance(events[0], LLMPortCall)
        assert events[0].response == "a"
