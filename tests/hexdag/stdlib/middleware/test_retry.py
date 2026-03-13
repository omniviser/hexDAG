"""Tests for RetryWithBackoff middleware."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from hexdag.stdlib.middleware.retry import RetryWithBackoff


class _FakeAdapter:
    """Fake adapter for testing."""

    def __init__(self, responses: list[str | Exception] | None = None) -> None:
        self._responses = list(responses or ["ok"])
        self._call_count = 0
        self.model = "test-model"

    async def aresponse(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        self._call_count += 1
        resp = self._responses[min(self._call_count - 1, len(self._responses) - 1)]
        if isinstance(resp, Exception):
            raise resp
        return resp


class TestRetryWithBackoff:
    """Tests for RetryWithBackoff middleware."""

    @pytest.mark.asyncio
    async def test_no_retry_on_success(self) -> None:
        """Successful call returns immediately without retry."""
        adapter = _FakeAdapter(["hello"])
        mw = RetryWithBackoff(adapter, max_retries=3, base_delay=0.01)

        result = await mw.aresponse([])
        assert result == "hello"
        assert mw.total_retries == 0

    @pytest.mark.asyncio
    async def test_retries_on_failure_then_succeeds(self) -> None:
        """Retries on failure, then returns on success."""
        adapter = _FakeAdapter([ValueError("fail"), ValueError("fail"), "ok"])
        mw = RetryWithBackoff(adapter, max_retries=3, base_delay=0.01)

        result = await mw.aresponse([])
        assert result == "ok"
        assert mw.total_retries == 2

    @pytest.mark.asyncio
    async def test_exhausts_retries_then_raises(self) -> None:
        """Raises after exhausting all retries."""
        adapter = _FakeAdapter([ValueError("fail")] * 4)
        mw = RetryWithBackoff(adapter, max_retries=2, base_delay=0.01)

        with pytest.raises(ValueError, match="fail"):
            await mw.aresponse([])
        assert mw.total_retries == 2

    @pytest.mark.asyncio
    async def test_retryable_exceptions_filter(self) -> None:
        """Only retries on specified exception types."""
        adapter = _FakeAdapter([TypeError("wrong type")])
        mw = RetryWithBackoff(
            adapter,
            max_retries=3,
            base_delay=0.01,
            retryable_exceptions=(ValueError,),
        )

        # TypeError is not retryable, should raise immediately
        with pytest.raises(TypeError, match="wrong type"):
            await mw.aresponse([])
        assert mw.total_retries == 0

    @pytest.mark.asyncio
    async def test_retryable_exceptions_does_retry(self) -> None:
        """Retries on matching exception types."""
        adapter = _FakeAdapter([ValueError("retry me"), "ok"])
        mw = RetryWithBackoff(
            adapter,
            max_retries=3,
            base_delay=0.01,
            retryable_exceptions=(ValueError,),
        )

        result = await mw.aresponse([])
        assert result == "ok"
        assert mw.total_retries == 1

    @pytest.mark.asyncio
    async def test_getattr_passthrough(self) -> None:
        """Attributes are forwarded to inner adapter."""
        adapter = _FakeAdapter()
        mw = RetryWithBackoff(adapter, base_delay=0.01)

        assert mw.model == "test-model"

    @pytest.mark.asyncio
    async def test_delay_computation(self) -> None:
        """Delay doubles with each attempt (exponential backoff)."""
        adapter = _FakeAdapter()
        mw = RetryWithBackoff(adapter, base_delay=1.0, max_delay=60.0, jitter=False)

        assert mw._compute_delay(0) == 1.0
        assert mw._compute_delay(1) == 2.0
        assert mw._compute_delay(2) == 4.0
        assert mw._compute_delay(3) == 8.0

    @pytest.mark.asyncio
    async def test_delay_capped_at_max(self) -> None:
        """Delay is capped at max_delay."""
        adapter = _FakeAdapter()
        mw = RetryWithBackoff(adapter, base_delay=1.0, max_delay=5.0, jitter=False)

        assert mw._compute_delay(10) == 5.0

    @pytest.mark.asyncio
    async def test_acall_tool_retries(self) -> None:
        """acall_tool is also retried."""
        inner = AsyncMock()
        inner.acall_tool = AsyncMock(side_effect=[ValueError("err"), {"result": "ok"}])
        mw = RetryWithBackoff(inner, max_retries=2, base_delay=0.01)

        result = await mw.acall_tool("search", {"q": "test"})
        assert result == {"result": "ok"}
        assert mw.total_retries == 1
