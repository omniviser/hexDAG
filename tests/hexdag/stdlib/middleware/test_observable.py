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
