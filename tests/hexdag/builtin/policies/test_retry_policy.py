"""Tests for retry_policy module.

This module tests retry policies for automatic error recovery.
"""

from __future__ import annotations

import pytest

from hexdag.builtin.policies.retry_policy import (
    ExponentialBackoffPolicy,
    ParseRetryPolicy,
    RetryContext,
)
from hexdag.core.exceptions import ParseError


class TestRetryContext:
    """Tests for RetryContext model."""

    def test_basic_context(self) -> None:
        """Test creating a basic retry context."""
        ctx = RetryContext(
            node_name="test_node",
            attempt=0,
            max_retries=3,
            error="Parse error",
            original_input={"prompt": "Hello"},
        )
        assert ctx.node_name == "test_node"
        assert ctx.attempt == 0
        assert ctx.max_retries == 3
        assert ctx.error == "Parse error"
        assert ctx.original_input == {"prompt": "Hello"}
        assert ctx.previous_output is None

    def test_context_with_previous_output(self) -> None:
        """Test context with previous output."""
        ctx = RetryContext(
            node_name="test",
            attempt=1,
            max_retries=3,
            error="Validation error",
            original_input={},
            previous_output={"data": "value"},
        )
        assert ctx.previous_output == {"data": "value"}


class TestParseRetryPolicy:
    """Tests for ParseRetryPolicy."""

    def test_default_initialization(self) -> None:
        """Test default policy initialization."""
        policy = ParseRetryPolicy()
        assert policy.max_retries == 2
        assert policy.backoff_multiplier == 1.5
        assert policy.retry_prompt_builder is not None

    def test_custom_initialization(self) -> None:
        """Test custom policy initialization."""
        policy = ParseRetryPolicy(
            max_retries=5,
            backoff_multiplier=2.0,
        )
        assert policy.max_retries == 5
        assert policy.backoff_multiplier == 2.0

    def test_custom_prompt_builder(self) -> None:
        """Test custom prompt builder."""

        def custom_builder(ctx: RetryContext) -> str:
            return f"Retry: {ctx.error}"

        policy = ParseRetryPolicy(retry_prompt_builder=custom_builder)
        assert policy.retry_prompt_builder is custom_builder

    @pytest.mark.asyncio
    async def test_should_retry_parse_error(self) -> None:
        """Test that ParseError triggers retry."""
        policy = ParseRetryPolicy(max_retries=3)
        ctx = RetryContext(
            node_name="parser",
            attempt=0,
            max_retries=3,
            error="Invalid JSON",
            original_input={},
        )
        error = ParseError("Invalid JSON")

        result = await policy.should_retry(error, ctx)
        assert result is True

    @pytest.mark.asyncio
    async def test_should_not_retry_other_errors(self) -> None:
        """Test that non-ParseError doesn't trigger retry."""
        policy = ParseRetryPolicy(max_retries=3)
        ctx = RetryContext(
            node_name="node",
            attempt=0,
            max_retries=3,
            error="Runtime error",
            original_input={},
        )
        error = RuntimeError("Some error")

        result = await policy.should_retry(error, ctx)
        assert result is False

    @pytest.mark.asyncio
    async def test_should_not_retry_max_attempts(self) -> None:
        """Test that max retries prevents retry."""
        policy = ParseRetryPolicy(max_retries=2)
        ctx = RetryContext(
            node_name="parser",
            attempt=2,  # Already at max
            max_retries=2,
            error="Parse error",
            original_input={},
        )
        error = ParseError("Invalid JSON")

        result = await policy.should_retry(error, ctx)
        assert result is False

    @pytest.mark.asyncio
    async def test_prepare_retry_input_with_template(self) -> None:
        """Test preparing retry input with template field."""

        def simple_builder(ctx: RetryContext) -> str:
            return "improved prompt"

        policy = ParseRetryPolicy(retry_prompt_builder=simple_builder)
        ctx = RetryContext(
            node_name="node",
            attempt=0,
            max_retries=3,
            error="Error",
            original_input={"template": "original", "other": "value"},
        )

        result = await policy.prepare_retry_input(ctx)
        assert result["template"] == "improved prompt"
        assert result["other"] == "value"

    @pytest.mark.asyncio
    async def test_prepare_retry_input_with_prompt(self) -> None:
        """Test preparing retry input with prompt field."""

        def simple_builder(ctx: RetryContext) -> str:
            return "improved"

        policy = ParseRetryPolicy(retry_prompt_builder=simple_builder)
        ctx = RetryContext(
            node_name="node",
            attempt=0,
            max_retries=3,
            error="Error",
            original_input={"prompt": "original"},
        )

        result = await policy.prepare_retry_input(ctx)
        assert result["prompt"] == "improved"

    @pytest.mark.asyncio
    async def test_prepare_retry_input_with_text(self) -> None:
        """Test preparing retry input with text field."""

        def simple_builder(ctx: RetryContext) -> str:
            return "improved"

        policy = ParseRetryPolicy(retry_prompt_builder=simple_builder)
        ctx = RetryContext(
            node_name="node",
            attempt=0,
            max_retries=3,
            error="Error",
            original_input={"text": "original"},
        )

        result = await policy.prepare_retry_input(ctx)
        assert result["text"] == "improved"

    @pytest.mark.asyncio
    async def test_prepare_retry_input_fallback(self) -> None:
        """Test preparing retry input with no known field."""

        def simple_builder(ctx: RetryContext) -> str:
            return "improved"

        policy = ParseRetryPolicy(retry_prompt_builder=simple_builder)
        ctx = RetryContext(
            node_name="node",
            attempt=0,
            max_retries=3,
            error="Error",
            original_input={"data": "value"},
        )

        result = await policy.prepare_retry_input(ctx)
        assert result["retry_prompt"] == "improved"
        assert result["data"] == "value"

    def test_classify_error_parse(self) -> None:
        """Test error classification for JSON parse errors."""
        policy = ParseRetryPolicy()
        error_type = policy._classify_error("JSON decode error: Expecting property name")
        assert error_type == "parse"

    def test_classify_error_validation(self) -> None:
        """Test error classification for validation errors."""
        policy = ParseRetryPolicy()
        error_type = policy._classify_error("Schema validation failed: required field missing")
        assert error_type == "validation"

    def test_classify_error_markdown(self) -> None:
        """Test error classification for markdown errors."""
        policy = ParseRetryPolicy()
        error_type = policy._classify_error("Could not find code block in markdown")
        assert error_type == "markdown"

    def test_classify_error_generic(self) -> None:
        """Test error classification for unknown errors."""
        policy = ParseRetryPolicy()
        error_type = policy._classify_error("Some random error occurred")
        assert error_type == "generic"

    def test_extract_error_summary_short(self) -> None:
        """Test error summary extraction for short messages."""
        policy = ParseRetryPolicy()
        summary = policy._extract_error_summary("Short error message")
        assert summary == "Short error message"

    def test_extract_error_summary_multiline(self) -> None:
        """Test error summary extraction for multiline messages."""
        policy = ParseRetryPolicy()
        error = "Line 1\nLine 2\nLine 3\nRetry hints: ignore this"
        summary = policy._extract_error_summary(error)
        assert "Line 1" in summary
        assert "Line 2" in summary
        assert "Line 3" in summary
        assert "Retry hints" not in summary

    def test_extract_error_summary_truncation(self) -> None:
        """Test error summary truncation for long messages."""
        policy = ParseRetryPolicy()
        long_error = "x" * 1000
        summary = policy._extract_error_summary(long_error)
        assert len(summary) <= 503  # 500 + "..."
        assert summary.endswith("...")


class TestExponentialBackoffPolicy:
    """Tests for ExponentialBackoffPolicy."""

    def test_default_initialization(self) -> None:
        """Test default policy initialization."""
        policy = ExponentialBackoffPolicy()
        assert policy.max_retries == 5
        assert policy.base_delay == 1.0
        assert policy.max_delay == 60.0
        assert policy.multiplier == 2.0

    def test_custom_initialization(self) -> None:
        """Test custom policy initialization."""
        policy = ExponentialBackoffPolicy(
            max_retries=3,
            base_delay=0.5,
            max_delay=30.0,
            multiplier=3.0,
        )
        assert policy.max_retries == 3
        assert policy.base_delay == 0.5
        assert policy.max_delay == 30.0
        assert policy.multiplier == 3.0

    @pytest.mark.asyncio
    async def test_should_retry_rate_limit(self) -> None:
        """Test that rate limit errors trigger retry."""
        policy = ExponentialBackoffPolicy()
        ctx = RetryContext(
            node_name="api",
            attempt=0,
            max_retries=5,
            error="Rate limit exceeded",
            original_input={},
        )
        error = Exception("rate limit exceeded")

        result = await policy.should_retry(error, ctx)
        assert result is True

    @pytest.mark.asyncio
    async def test_should_retry_429(self) -> None:
        """Test that 429 errors trigger retry."""
        policy = ExponentialBackoffPolicy()
        ctx = RetryContext(
            node_name="api",
            attempt=0,
            max_retries=5,
            error="429 Too Many Requests",
            original_input={},
        )
        error = Exception("429 error")

        result = await policy.should_retry(error, ctx)
        assert result is True

    @pytest.mark.asyncio
    async def test_should_not_retry_other_errors(self) -> None:
        """Test that non-rate-limit errors don't trigger retry."""
        policy = ExponentialBackoffPolicy()
        ctx = RetryContext(
            node_name="api",
            attempt=0,
            max_retries=5,
            error="Connection error",
            original_input={},
        )
        error = Exception("Connection timeout")

        result = await policy.should_retry(error, ctx)
        assert result is False

    @pytest.mark.asyncio
    async def test_should_not_retry_max_attempts(self) -> None:
        """Test that max retries prevents retry."""
        policy = ExponentialBackoffPolicy(max_retries=3)
        ctx = RetryContext(
            node_name="api",
            attempt=3,
            max_retries=3,
            error="Rate limit",
            original_input={},
        )
        error = Exception("rate limit")

        result = await policy.should_retry(error, ctx)
        assert result is False

    def test_calculate_delay_first_attempt(self) -> None:
        """Test delay calculation for first attempt."""
        policy = ExponentialBackoffPolicy(base_delay=1.0, multiplier=2.0)
        delay = policy.calculate_delay(0)
        assert delay == 1.0

    def test_calculate_delay_second_attempt(self) -> None:
        """Test delay calculation for second attempt."""
        policy = ExponentialBackoffPolicy(base_delay=1.0, multiplier=2.0)
        delay = policy.calculate_delay(1)
        assert delay == 2.0

    def test_calculate_delay_exponential(self) -> None:
        """Test exponential delay calculation."""
        policy = ExponentialBackoffPolicy(base_delay=1.0, multiplier=2.0)
        delays = [policy.calculate_delay(i) for i in range(5)]
        assert delays == [1.0, 2.0, 4.0, 8.0, 16.0]

    def test_calculate_delay_max_cap(self) -> None:
        """Test delay capped at max_delay."""
        policy = ExponentialBackoffPolicy(base_delay=1.0, max_delay=10.0, multiplier=2.0)
        delay = policy.calculate_delay(10)  # Would be 1024 without cap
        assert delay == 10.0
