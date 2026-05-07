"""Tests for BatchGeneration middleware."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from hexdag.kernel.ports.llm import (
    BatchItemResult,
    BatchItemStatus,
    BatchResult,
    MessageList,
    TokenUsage,
)
from hexdag.stdlib.middleware.batch_generation import BatchGeneration


class _FakeAdapter:
    """Fake adapter for testing."""

    def __init__(self, delay: float = 0.0) -> None:
        self.call_count = 0
        self.model = "test-model"
        self._delay = delay

    async def aresponse(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        self.call_count += 1
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        return f"response-{self.call_count}"


class _FakeAdapterWithUsage(_FakeAdapter):
    """Fake adapter that tracks token usage."""

    def __init__(self, delay: float = 0.0) -> None:
        super().__init__(delay)
        self._last_usage: TokenUsage | None = None

    async def aresponse(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        result = await super().aresponse(*args, **kwargs)
        self._last_usage = TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30)
        return result

    def get_last_usage(self) -> TokenUsage | None:
        return self._last_usage


class _FailingAdapter:
    """Adapter that fails on specific call indices."""

    def __init__(self, fail_on: set[int]) -> None:
        self.call_count = 0
        self._fail_on = fail_on

    async def aresponse(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        self.call_count += 1
        if self.call_count in self._fail_on:
            raise ValueError(f"Deliberate failure on call {self.call_count}")
        return f"response-{self.call_count}"


class TestBatchGeneration:
    """Tests for BatchGeneration middleware."""

    # -- Single call passthrough --

    @pytest.mark.asyncio
    async def test_aresponse_passthrough(self) -> None:
        """Single aresponse() call passes through to inner adapter."""
        adapter = _FakeAdapter()
        mw = BatchGeneration(adapter, max_concurrency=10)

        result = await mw.aresponse([])
        assert result == "response-1"
        assert adapter.call_count == 1

    @pytest.mark.asyncio
    async def test_total_calls_counter(self) -> None:
        """total_calls increments for each call."""
        adapter = _FakeAdapter()
        mw = BatchGeneration(adapter)

        await mw.aresponse([])
        await mw.aresponse([])
        assert mw.total_calls == 2
        assert mw.active_calls == 0

    # -- Semaphore concurrency limiting --

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self) -> None:
        """Concurrent aresponse() calls are limited by semaphore."""
        adapter = _FakeAdapter(delay=0.1)
        mw = BatchGeneration(adapter, max_concurrency=2)

        # Fire 4 concurrent calls with max_concurrency=2
        # Should take ~0.2s (2 batches of 2), not ~0.1s (all 4 at once)
        import time

        start = time.monotonic()
        results = await asyncio.gather(
            mw.aresponse([]),
            mw.aresponse([]),
            mw.aresponse([]),
            mw.aresponse([]),
        )
        elapsed = time.monotonic() - start

        assert len(results) == 4
        assert adapter.call_count == 4
        # With max_concurrency=2, should take at least 2 rounds × 0.1s
        assert elapsed >= 0.18

    # -- aresponse_batch --

    @pytest.mark.asyncio
    async def test_aresponse_batch_returns_ordered_results(self) -> None:
        """aresponse_batch returns results in input order."""
        adapter = _FakeAdapter()
        mw = BatchGeneration(adapter)

        msgs: list[MessageList] = [[] for _ in range(5)]
        result = await mw.aresponse_batch(msgs)

        assert isinstance(result, BatchResult)
        assert len(result.items) == 5
        assert all(item.status == BatchItemStatus.COMPLETED for item in result.items)
        # contents preserves input order
        contents = result.contents
        assert len(contents) == 5
        assert all(c is not None for c in contents)

    @pytest.mark.asyncio
    async def test_aresponse_batch_respects_semaphore(self) -> None:
        """Batch calls are gated by the semaphore."""
        adapter = _FakeAdapter(delay=0.1)
        mw = BatchGeneration(adapter, max_concurrency=2)

        import time

        start = time.monotonic()
        result = await mw.aresponse_batch([[] for _ in range(6)])
        elapsed = time.monotonic() - start

        assert len(result.items) == 6
        assert adapter.call_count == 6
        # 6 items with max_concurrency=2 → 3 rounds × 0.1s ≈ 0.3s
        assert elapsed >= 0.28

    @pytest.mark.asyncio
    async def test_batch_item_failure_isolated(self) -> None:
        """One item failure doesn't abort the entire batch."""
        adapter = _FailingAdapter(fail_on={2})
        mw = BatchGeneration(adapter, max_concurrency=10)

        result = await mw.aresponse_batch([[], [], []])

        completed = [i for i in result.items if i.status == BatchItemStatus.COMPLETED]
        failed = [i for i in result.items if i.status == BatchItemStatus.FAILED]

        assert len(completed) == 2
        assert len(failed) == 1
        assert failed[0].error is not None
        assert "Deliberate failure" in failed[0].error

    @pytest.mark.asyncio
    async def test_batch_usage_aggregation(self) -> None:
        """Token usage is aggregated across batch items."""
        adapter = _FakeAdapterWithUsage()
        mw = BatchGeneration(adapter)

        result = await mw.aresponse_batch([[], [], []])

        assert result.total_usage is not None
        assert result.total_usage.input_tokens == 30  # 3 × 10
        assert result.total_usage.output_tokens == 60  # 3 × 20
        assert result.total_usage.total_tokens == 90  # 3 × 30

    @pytest.mark.asyncio
    async def test_batch_provider_is_gather(self) -> None:
        """BatchResult.provider is 'gather' for the middleware."""
        adapter = _FakeAdapter()
        mw = BatchGeneration(adapter)

        result = await mw.aresponse_batch([[]])
        assert result.provider == "gather"

    @pytest.mark.asyncio
    async def test_batch_empty_list(self) -> None:
        """Empty message list returns empty batch result."""
        adapter = _FakeAdapter()
        mw = BatchGeneration(adapter)

        result = await mw.aresponse_batch([])
        assert len(result.items) == 0
        assert result.contents == []

    # -- Passthrough methods --

    @pytest.mark.asyncio
    async def test_aresponse_with_tools_passthrough(self) -> None:
        """aresponse_with_tools delegates to inner."""
        inner = AsyncMock()
        inner.aresponse_with_tools = AsyncMock(return_value={"ok": True})
        mw = BatchGeneration(inner)

        result = await mw.aresponse_with_tools([], [])
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_aresponse_structured_passthrough(self) -> None:
        """aresponse_structured delegates to inner."""
        inner = AsyncMock()
        inner.aresponse_structured = AsyncMock(return_value={"data": 1})
        mw = BatchGeneration(inner)

        result = await mw.aresponse_structured([], {})
        assert result == {"data": 1}

    @pytest.mark.asyncio
    async def test_acall_tool_passthrough(self) -> None:
        """acall_tool delegates to inner."""
        inner = AsyncMock()
        inner.acall_tool = AsyncMock(return_value={"result": True})
        mw = BatchGeneration(inner)

        result = await mw.acall_tool("search", {"q": "test"})
        assert result == {"result": True}

    # -- __getattr__ --

    @pytest.mark.asyncio
    async def test_getattr_passthrough(self) -> None:
        """Attributes are forwarded to inner adapter."""
        adapter = _FakeAdapter()
        mw = BatchGeneration(adapter)

        assert mw.model == "test-model"


class TestBatchResultModel:
    """Tests for BatchResult domain model."""

    def test_contents_property_ordered(self) -> None:
        """contents returns items sorted by index."""
        result = BatchResult(
            items=[
                BatchItemResult(index=2, content="c"),
                BatchItemResult(index=0, content="a"),
                BatchItemResult(index=1, content="b"),
            ],
        )
        assert result.contents == ["a", "b", "c"]

    def test_contents_with_none(self) -> None:
        """contents includes None for failed items."""
        result = BatchResult(
            items=[
                BatchItemResult(index=0, content="ok"),
                BatchItemResult(index=1, content=None, status=BatchItemStatus.FAILED, error="fail"),
            ],
        )
        assert result.contents == ["ok", None]

    def test_aggregate_usage_sums(self) -> None:
        """aggregate_usage sums across items."""
        usages = [
            TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30),
            TokenUsage(input_tokens=5, output_tokens=15, total_tokens=20),
            None,
        ]
        total = BatchResult.aggregate_usage(usages)
        assert total is not None
        assert total.input_tokens == 15
        assert total.output_tokens == 35
        assert total.total_tokens == 50

    def test_aggregate_usage_all_none(self) -> None:
        """aggregate_usage returns None when all items have no usage."""
        assert BatchResult.aggregate_usage([None, None]) is None
