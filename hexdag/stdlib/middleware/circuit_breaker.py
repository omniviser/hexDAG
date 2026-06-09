"""Circuit breaker middleware — prevents cascade failures.

Tracks consecutive failures on the inner port. After ``failure_threshold``
failures the breaker **opens** and immediately rejects calls with
``CircuitBreakerOpenError`` for ``reset_timeout`` seconds. After the
timeout the breaker enters **half-open** state and allows up to
``half_open_max_calls`` probe calls. If a probe succeeds the breaker
**closes**; if it fails the breaker re-opens.

State diagram::

    CLOSED ──[failure_threshold reached]──► OPEN
      ▲                                        │
      │                                   [reset_timeout elapsed]
      │                                        ▼
      └──────[probe succeeds]───── HALF_OPEN ──[probe fails]──► OPEN

Example YAML::

    spec:
      ports:
        llm:
          adapter: openai
          middleware:
            - hexdag.stdlib.middleware.CircuitBreaker:
                failure_threshold: 5
                reset_timeout: 60.0
"""

from __future__ import annotations

import asyncio
import enum
import time
from typing import Any

from hexdag.kernel.exceptions import HexDAGError
from hexdag.kernel.logging import get_logger

logger = get_logger(__name__)

# Defaults
_DEFAULT_FAILURE_THRESHOLD = 5
_DEFAULT_RESET_TIMEOUT = 60.0
_DEFAULT_HALF_OPEN_MAX_CALLS = 1


class CircuitBreakerOpenError(HexDAGError):
    """Raised when a call is rejected because the circuit breaker is open."""


class _State(enum.Enum):
    """Circuit breaker states: closed, open, or half-open."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Middleware that prevents cascade failures via the circuit breaker pattern.

    Parameters
    ----------
    inner : Any
        The port/adapter to wrap.
    failure_threshold : int
        Number of consecutive failures before the breaker opens (default 5).
    reset_timeout : float
        Seconds to wait in open state before allowing probe calls (default 60.0).
    half_open_max_calls : int
        Maximum concurrent probe calls in half-open state (default 1).
    """

    def __init__(
        self,
        inner: Any,
        failure_threshold: int = _DEFAULT_FAILURE_THRESHOLD,
        reset_timeout: float = _DEFAULT_RESET_TIMEOUT,
        half_open_max_calls: int = _DEFAULT_HALF_OPEN_MAX_CALLS,
    ) -> None:
        """Initialize the circuit breaker wrapping the given inner port."""
        self._inner = inner
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._half_open_max_calls = half_open_max_calls

        self._state = _State.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float = 0.0
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

        # Observation counters
        self._trip_count = 0
        self._success_count = 0
        self._failure_count = 0

    # -- Observation properties --

    @property
    def state(self) -> str:
        """Current circuit breaker state: 'closed', 'open', or 'half_open'."""
        if self._state == _State.OPEN and self._timeout_elapsed():
            return _State.HALF_OPEN.value
        return self._state.value

    @property
    def trip_count(self) -> int:
        """Number of times the breaker has tripped from closed to open."""
        return self._trip_count

    @property
    def success_count(self) -> int:
        """Total successful calls."""
        return self._success_count

    @property
    def failure_count(self) -> int:
        """Total failed calls (includes calls that tripped the breaker)."""
        return self._failure_count

    # -- Internal state management --

    def _timeout_elapsed(self) -> bool:
        """Return True if the reset timeout has elapsed since the breaker opened."""
        return time.monotonic() - self._opened_at >= self._reset_timeout

    def _trip(self) -> None:
        """Transition from CLOSED/HALF_OPEN to OPEN."""
        self._state = _State.OPEN
        self._opened_at = time.monotonic()
        self._half_open_calls = 0
        self._trip_count += 1
        logger.warning(
            "Circuit breaker OPENED after {} consecutive failures on {}",
            self._consecutive_failures,
            type(self._inner).__name__,
        )

    def _close(self) -> None:
        """Transition to CLOSED."""
        self._state = _State.CLOSED
        self._consecutive_failures = 0
        self._half_open_calls = 0
        logger.info("Circuit breaker CLOSED on {}", type(self._inner).__name__)

    def _record_success(self) -> None:
        """Record a successful call, closing the breaker if half-open."""
        self._success_count += 1
        if self._state == _State.HALF_OPEN:
            self._close()
        else:
            self._consecutive_failures = 0

    def _record_failure(self) -> None:
        """Record a failed call, tripping the breaker if the threshold is reached."""
        self._failure_count += 1
        self._consecutive_failures += 1
        if self._state == _State.HALF_OPEN or self._consecutive_failures >= self._failure_threshold:
            self._trip()

    # -- Core call logic --

    async def _call_with_breaker(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Call a method on the inner port with circuit breaker protection."""
        async with self._lock:
            # Check state transitions
            if self._state == _State.OPEN:
                if self._timeout_elapsed():
                    self._state = _State.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info(
                        "Circuit breaker HALF_OPEN on {} — allowing probe calls",
                        type(self._inner).__name__,
                    )
                else:
                    remaining = self._reset_timeout - (time.monotonic() - self._opened_at)
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is open for {type(self._inner).__name__}. "
                        f"Retry after {remaining:.1f}s."
                    )

            if self._state == _State.HALF_OPEN:
                if self._half_open_calls >= self._half_open_max_calls:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is half-open with max probe calls "
                        f"({self._half_open_max_calls}) in flight for {type(self._inner).__name__}."
                    )
                self._half_open_calls += 1

        # Execute outside the lock so we don't block concurrent closed-state calls
        method = getattr(self._inner, method_name)
        try:
            result = await method(*args, **kwargs)
        except Exception:
            async with self._lock:
                self._record_failure()
            raise
        else:
            async with self._lock:
                self._record_success()
            return result

    # -- LLM protocol methods --

    async def aresponse(self, *args: Any, **kwargs: Any) -> Any:
        """Circuit-breaker-wrapped aresponse."""
        return await self._call_with_breaker("aresponse", *args, **kwargs)

    async def aresponse_with_tools(self, *args: Any, **kwargs: Any) -> Any:
        """Circuit-breaker-wrapped aresponse_with_tools."""
        return await self._call_with_breaker("aresponse_with_tools", *args, **kwargs)

    async def aresponse_structured(self, *args: Any, **kwargs: Any) -> Any:
        """Circuit-breaker-wrapped aresponse_structured."""
        return await self._call_with_breaker("aresponse_structured", *args, **kwargs)

    # -- ToolRouter protocol methods --

    async def acall_tool(self, *args: Any, **kwargs: Any) -> Any:
        """Circuit-breaker-wrapped acall_tool."""
        return await self._call_with_breaker("acall_tool", *args, **kwargs)

    # -- Passthrough --

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the inner port."""
        return getattr(self._inner, name)
