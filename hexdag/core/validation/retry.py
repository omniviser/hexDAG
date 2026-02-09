"""Retry strategies with exponential backoff for node execution.

This module extracts retry logic from NodeExecutor into reusable, testable
components. The primary interface is ``execute_with_retry`` which wraps an
async callable with configurable exponential backoff.

Examples
--------
Basic usage with default config::

    config = RetryConfig(max_retries=3)
    result = await execute_with_retry(my_async_fn, config)

Custom backoff::

    config = RetryConfig(
        max_retries=5,
        delay=0.5,
        backoff=3.0,
        max_delay=30.0,
    )
    result = await execute_with_retry(my_async_fn, config)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hexdag.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Configuration for retry behavior with exponential backoff.

    Parameters
    ----------
    max_retries : int
        Total number of attempts. 1 means no retries (single attempt).
    delay : float
        Initial delay in seconds before the first retry.
    backoff : float
        Multiplier applied to the delay after each retry.
    max_delay : float
        Maximum delay cap in seconds.
    """

    max_retries: int = 1
    delay: float = 1.0
    backoff: float = 2.0
    max_delay: float = 60.0

    @classmethod
    def from_node_spec_fields(
        cls,
        max_retries: int | None = None,
        retry_delay: float | None = None,
        retry_backoff: float | None = None,
        retry_max_delay: float | None = None,
    ) -> RetryConfig:
        """Build a RetryConfig from NodeSpec-style fields.

        Parameters
        ----------
        max_retries : int | None
            Total attempts (None → 1, no retries).
        retry_delay : float | None
            Initial delay (None → 1.0s).
        retry_backoff : float | None
            Backoff multiplier (None → 2.0).
        retry_max_delay : float | None
            Max delay cap (None → 60.0s).

        Returns
        -------
        RetryConfig

        Examples
        --------
        >>> cfg = RetryConfig.from_node_spec_fields(max_retries=3, retry_delay=0.5)
        >>> cfg.max_retries
        3
        >>> cfg.delay
        0.5
        """
        return cls(
            max_retries=max_retries or 1,
            delay=retry_delay or 1.0,
            backoff=retry_backoff or 2.0,
            max_delay=retry_max_delay or 60.0,
        )

    @property
    def has_retries(self) -> bool:
        """Whether this config enables retries (max_retries > 1)."""
        return self.max_retries > 1

    def compute_delay(self, attempt: int) -> float:
        """Compute the delay for a given attempt number (1-indexed).

        Parameters
        ----------
        attempt : int
            The attempt number that just failed (1-indexed).

        Returns
        -------
        float
            Delay in seconds before the next attempt.

        Examples
        --------
        >>> cfg = RetryConfig(delay=1.0, backoff=2.0, max_delay=10.0)
        >>> cfg.compute_delay(1)
        1.0
        >>> cfg.compute_delay(2)
        2.0
        >>> cfg.compute_delay(3)
        4.0
        >>> cfg.compute_delay(10)
        10.0
        """
        return min(self.delay * (self.backoff ** (attempt - 1)), self.max_delay)


async def execute_with_retry(
    fn: Callable[..., Awaitable[Any]],
    config: RetryConfig,
    *,
    on_retry: Callable[[int, int, Exception, float], Any] | None = None,
) -> Any:
    """Execute an async callable with retry and exponential backoff.

    Parameters
    ----------
    fn : Callable[..., Awaitable[Any]]
        Zero-argument async callable to execute. Callers should use
        ``functools.partial`` or a lambda to bind arguments.
    config : RetryConfig
        Retry configuration.
    on_retry : callable, optional
        Callback invoked before each retry sleep. Receives
        ``(attempt, max_retries, error, delay)``. Useful for logging
        or metrics from the caller's context.

    Returns
    -------
    Any
        The return value of *fn*.

    Examples
    --------
    >>> import asyncio
    >>> async def ok(): return 42
    >>> asyncio.run(execute_with_retry(ok, RetryConfig()))
    42
    """
    last_error: Exception | None = None

    for attempt in range(1, config.max_retries + 1):
        try:
            return await fn()
        except Exception as exc:
            last_error = exc
            if attempt < config.max_retries:
                delay = config.compute_delay(attempt)
                if on_retry is not None:
                    on_retry(attempt, config.max_retries, exc, delay)
                await asyncio.sleep(delay)
                continue
            raise

    # Should not be reached, but satisfy the type checker
    if last_error is not None:  # pragma: no cover
        raise last_error
    return None  # pragma: no cover
