"""Tests for Round-Robin LLM middleware."""

from unittest.mock import AsyncMock

import pytest

from hexdag.kernel.ports.llm import LLMResponse, Message
from hexdag.stdlib.middleware.round_robin import RoundRobin


def _make_adapter(response: str = "ok") -> AsyncMock:
    """Create a mock adapter with SupportsGeneration interface."""
    adapter = AsyncMock()
    adapter.aresponse = AsyncMock(return_value=response)
    adapter.aresponse_structured = AsyncMock(return_value={"result": response})
    adapter.aresponse_with_tools = AsyncMock(
        return_value=LLMResponse(content=response, tool_calls=None)
    )
    adapter.aclose = AsyncMock()
    adapter.clear_cache = AsyncMock()
    return adapter


@pytest.fixture
def messages():
    return [Message(role="user", content="Hello")]


class TestInitialization:
    def test_requires_at_least_one_adapter(self):
        with pytest.raises(ValueError, match="at least one"):
            RoundRobin(adapters=[])

    def test_single_adapter(self):
        adapter = _make_adapter()
        rr = RoundRobin(adapters=[adapter])
        assert len(rr._adapters) == 1

    def test_multiple_adapters(self):
        adapters = [_make_adapter() for _ in range(3)]
        rr = RoundRobin(adapters=adapters)
        assert len(rr._adapters) == 3


@pytest.mark.asyncio
class TestRoundRobinDistribution:
    async def test_distributes_across_adapters(self, messages):
        a1 = _make_adapter("response-1")
        a2 = _make_adapter("response-2")
        a3 = _make_adapter("response-3")
        rr = RoundRobin(adapters=[a1, a2, a3])

        r1 = await rr.aresponse(messages)
        r2 = await rr.aresponse(messages)
        r3 = await rr.aresponse(messages)
        r4 = await rr.aresponse(messages)

        assert r1 == "response-1"
        assert r2 == "response-2"
        assert r3 == "response-3"
        assert r4 == "response-1"  # wraps around

    async def test_single_adapter_always_used(self, messages):
        a1 = _make_adapter("only")
        rr = RoundRobin(adapters=[a1])

        for _ in range(5):
            result = await rr.aresponse(messages)
            assert result == "only"


@pytest.mark.asyncio
class TestFailover:
    async def test_failover_on_exception(self, messages):
        a1 = _make_adapter()
        a1.aresponse = AsyncMock(side_effect=RuntimeError("fail"))
        a2 = _make_adapter("fallback")

        rr = RoundRobin(adapters=[a1, a2])
        result = await rr.aresponse(messages)
        assert result == "fallback"

    async def test_failover_on_none_response(self, messages):
        a1 = _make_adapter()
        a1.aresponse = AsyncMock(return_value=None)
        a2 = _make_adapter("success")

        rr = RoundRobin(adapters=[a1, a2])
        result = await rr.aresponse(messages)
        assert result == "success"

    async def test_all_fail_returns_none_for_aresponse(self, messages):
        a1 = _make_adapter()
        a1.aresponse = AsyncMock(side_effect=RuntimeError("fail"))
        a2 = _make_adapter()
        a2.aresponse = AsyncMock(side_effect=RuntimeError("fail"))

        rr = RoundRobin(adapters=[a1, a2])
        result = await rr.aresponse(messages)
        assert result is None

    async def test_all_fail_raises_for_tools(self, messages):
        a1 = _make_adapter()
        a1.aresponse_with_tools = AsyncMock(side_effect=RuntimeError("fail"))
        a2 = _make_adapter()
        a2.aresponse_with_tools = AsyncMock(side_effect=RuntimeError("fail"))

        rr = RoundRobin(adapters=[a1, a2])
        with pytest.raises(RuntimeError, match="fail"):
            await rr.aresponse_with_tools(messages, tools=[])

    async def test_no_failover_mode(self, messages):
        a1 = _make_adapter()
        a1.aresponse = AsyncMock(side_effect=RuntimeError("fail"))
        a2 = _make_adapter("success")

        rr = RoundRobin(adapters=[a1, a2], failover=False)
        # Without failover, should return None (aresponse error path)
        result = await rr.aresponse(messages)
        assert result is None


@pytest.mark.asyncio
class TestStructuredOutput:
    async def test_structured_output_forwarded(self, messages):
        a1 = _make_adapter()
        a1.aresponse_structured = AsyncMock(return_value={"name": "test", "age": 25})
        rr = RoundRobin(adapters=[a1])

        result = await rr.aresponse_structured(messages, output_schema={"type": "object"})
        assert result == {"name": "test", "age": 25}


@pytest.mark.asyncio
class TestLifecycle:
    async def test_aclose_closes_all(self):
        adapters = [_make_adapter() for _ in range(3)]
        rr = RoundRobin(adapters=adapters)

        await rr.aclose()
        for adapter in adapters:
            adapter.aclose.assert_called_once()

    async def test_clear_cache_clears_all(self):
        adapters = [_make_adapter() for _ in range(2)]
        rr = RoundRobin(adapters=adapters)

        await rr.clear_cache()
        for adapter in adapters:
            adapter.clear_cache.assert_called_once()

    async def test_aclose_handles_missing_method(self):
        """Adapters without aclose should not raise."""
        adapter = AsyncMock(spec=[])  # No methods
        adapter.aresponse = AsyncMock(return_value="ok")
        rr = RoundRobin(adapters=[adapter])
        await rr.aclose()  # Should not raise
