"""Tests for hexdag.kernel.validation.retry module."""

import time
from typing import Any

import pytest

from hexdag.kernel.validation.retry import RetryConfig, execute_with_retry


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = RetryConfig()
        assert cfg.max_retries == 1
        assert cfg.delay == 1.0
        assert cfg.backoff == 2.0
        assert cfg.max_delay == 60.0

    def test_custom_values(self) -> None:
        cfg = RetryConfig(max_retries=5, delay=0.5, backoff=3.0, max_delay=30.0)
        assert cfg.max_retries == 5
        assert cfg.delay == 0.5
        assert cfg.backoff == 3.0
        assert cfg.max_delay == 30.0

    def test_from_node_spec_fields_all_set(self) -> None:
        cfg = RetryConfig.from_node_spec_fields(
            max_retries=3, retry_delay=2.0, retry_backoff=1.5, retry_max_delay=10.0
        )
        assert cfg.max_retries == 3
        assert cfg.delay == 2.0
        assert cfg.backoff == 1.5
        assert cfg.max_delay == 10.0

    def test_from_node_spec_fields_none_uses_defaults(self) -> None:
        cfg = RetryConfig.from_node_spec_fields()
        assert cfg.max_retries == 1
        assert cfg.delay == 1.0
        assert cfg.backoff == 2.0
        assert cfg.max_delay == 60.0

    def test_from_node_spec_fields_partial(self) -> None:
        cfg = RetryConfig.from_node_spec_fields(max_retries=3)
        assert cfg.max_retries == 3
        assert cfg.delay == 1.0  # default

    def test_has_retries_true(self) -> None:
        assert RetryConfig(max_retries=2).has_retries is True

    def test_has_retries_false(self) -> None:
        assert RetryConfig(max_retries=1).has_retries is False

    def test_compute_delay_exponential(self) -> None:
        cfg = RetryConfig(delay=1.0, backoff=2.0, max_delay=100.0)
        assert cfg.compute_delay(1) == 1.0
        assert cfg.compute_delay(2) == 2.0
        assert cfg.compute_delay(3) == 4.0
        assert cfg.compute_delay(4) == 8.0

    def test_compute_delay_capped(self) -> None:
        cfg = RetryConfig(delay=1.0, backoff=10.0, max_delay=5.0)
        assert cfg.compute_delay(1) == 1.0
        assert cfg.compute_delay(2) == 5.0  # 10.0 capped to 5.0
        assert cfg.compute_delay(3) == 5.0  # 100.0 capped to 5.0

    def test_frozen(self) -> None:
        cfg = RetryConfig()
        with pytest.raises(AttributeError):
            cfg.max_retries = 5  # type: ignore[misc]


class TestExecuteWithRetry:
    """Tests for execute_with_retry function."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self) -> None:
        async def ok() -> int:
            return 42

        result = await execute_with_retry(ok, RetryConfig())
        assert result == 42

    @pytest.mark.asyncio
    async def test_success_after_retries(self) -> None:
        call_count = 0

        async def fail_twice() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"attempt {call_count}")
            return "ok"

        cfg = RetryConfig(max_retries=3, delay=0.01)
        result = await execute_with_retry(fail_twice, cfg)
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_exhausted_retries(self) -> None:
        call_count = 0

        async def always_fail() -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("permanent")

        cfg = RetryConfig(max_retries=3, delay=0.01)
        with pytest.raises(RuntimeError, match="permanent"):
            await execute_with_retry(always_fail, cfg)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_when_max_retries_is_one(self) -> None:
        call_count = 0

        async def fail() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        cfg = RetryConfig(max_retries=1)
        with pytest.raises(ValueError, match="fail"):
            await execute_with_retry(fail, cfg)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_on_retry_callback_called(self) -> None:
        call_count = 0
        retry_calls: list[tuple[int, int, str, float]] = []

        async def fail_once() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("transient")
            return "ok"

        def on_retry(attempt: int, max_retries: int, error: Exception, delay: float) -> None:
            retry_calls.append((attempt, max_retries, str(error), delay))

        cfg = RetryConfig(max_retries=3, delay=0.01)
        result = await execute_with_retry(fail_once, cfg, on_retry=on_retry)
        assert result == "ok"
        assert len(retry_calls) == 1
        assert retry_calls[0][0] == 1  # attempt
        assert retry_calls[0][1] == 3  # max_retries

    @pytest.mark.asyncio
    async def test_on_retry_not_called_on_success(self) -> None:
        retry_calls: list[Any] = []

        async def ok() -> int:
            return 1

        def on_retry(attempt: int, max_retries: int, error: Exception, delay: float) -> None:
            retry_calls.append(attempt)

        await execute_with_retry(ok, RetryConfig(max_retries=3), on_retry=on_retry)
        assert retry_calls == []

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self) -> None:
        call_times: list[float] = []
        call_count = 0

        async def fail_twice() -> str:
            nonlocal call_count
            call_times.append(time.monotonic())
            call_count += 1
            if call_count < 3:
                raise ValueError("transient")
            return "ok"

        cfg = RetryConfig(max_retries=3, delay=0.05, backoff=2.0, max_delay=1.0)
        await execute_with_retry(fail_twice, cfg)

        assert len(call_times) == 3
        # First retry delay: ~0.05s
        first_delay = call_times[1] - call_times[0]
        assert first_delay >= 0.04
        # Second retry delay: ~0.10s (0.05 * 2.0)
        second_delay = call_times[2] - call_times[1]
        assert second_delay >= 0.08

    @pytest.mark.asyncio
    async def test_max_delay_cap(self) -> None:
        call_times: list[float] = []
        call_count = 0

        async def fail_many() -> str:
            nonlocal call_count
            call_times.append(time.monotonic())
            call_count += 1
            if call_count < 4:
                raise ValueError("transient")
            return "ok"

        cfg = RetryConfig(max_retries=4, delay=0.05, backoff=10.0, max_delay=0.08)
        await execute_with_retry(fail_many, cfg)

        assert len(call_times) == 4
        # Second delay onwards should be capped at ~0.08s
        for i in range(1, len(call_times) - 1):
            delay = call_times[i + 1] - call_times[i]
            assert delay < 0.15  # capped + tolerance

    @pytest.mark.asyncio
    async def test_preserves_exception_type(self) -> None:
        async def fail() -> None:
            raise TypeError("type error")

        with pytest.raises(TypeError, match="type error"):
            await execute_with_retry(fail, RetryConfig(max_retries=1))

    @pytest.mark.asyncio
    async def test_timeout_error_propagated(self) -> None:
        async def timeout_fn() -> None:
            raise TimeoutError("timed out")

        cfg = RetryConfig(max_retries=2, delay=0.01)
        with pytest.raises(TimeoutError, match="timed out"):
            await execute_with_retry(timeout_fn, cfg)
