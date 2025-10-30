"""Simple policy implementations."""

import asyncio
import time
from typing import Any

from hexdag.core.logging import get_logger
from hexdag.core.orchestration.events.events import NodeCompleted, NodeFailed, NodeStarted
from hexdag.core.orchestration.policies.models import PolicyContext, PolicyResponse, PolicySignal
from hexdag.core.registry.decorators import policy

logger = get_logger(__name__)


class Retry:
    """Configuration for Retry Policy.

    Attributes
    ----------
    max_retries : int
        Maximum number of retry attempts (default: 3)
    """

    max_retries: int = 3


@policy(name="retry", description="Retry failed operations up to a maximum number of attempts")
class RetryPolicy:
    """Simple retry policy."""

    def __init__(self, max_retries: int = 3, **kwargs: Any):
        """Initialize retry policy with configuration."""
        self.max_retries = max_retries

    @property
    def name(self) -> str:
        """Policy name."""
        return f"retry_{self.max_retries}"

    @property
    def priority(self) -> int:
        """Priority - retry early."""
        return 10

    async def evaluate(self, context: PolicyContext) -> PolicyResponse:
        """Evaluate retry decision."""
        if context.error and context.attempt <= self.max_retries:
            return PolicyResponse(
                signal=PolicySignal.RETRY,
                data={"attempt": context.attempt},
                metadata={"max_retries": self.max_retries},
            )
        return PolicyResponse(signal=PolicySignal.PROCEED)


class CircuitBreaker:
    """Configuration for Circuit Breaker Policy.

    Attributes
    ----------
    failure_threshold : int
        Number of failures before opening circuit (default: 5)
    """

    failure_threshold: int = 5


@policy(
    name="circuit_breaker",
    description="Fail fast after reaching failure threshold to prevent cascading failures",
)
class CircuitBreakerPolicy:
    """Simple circuit breaker."""

    def __init__(self, failure_threshold: int = 5, **kwargs: Any):
        """Initialize circuit breaker policy with configuration."""
        self.failure_threshold = failure_threshold
        self.failure_count = 0
        self.is_open = False

    @property
    def name(self) -> str:
        """Policy name."""
        return f"circuit_breaker_{self.failure_threshold}"

    @property
    def priority(self) -> int:
        """Priority - check circuit before retry."""
        return 5

    async def evaluate(self, context: PolicyContext) -> PolicyResponse:
        """Evaluate circuit state."""
        if self.is_open:
            return PolicyResponse(signal=PolicySignal.FAIL, data="Circuit breaker is open")

        if context.error:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
                return PolicyResponse(
                    signal=PolicySignal.FAIL,
                    data=f"Circuit opened after {self.failure_count} failures",
                )
        else:
            # Success resets the count
            self.failure_count = 0

        return PolicyResponse(signal=PolicySignal.PROCEED)


class Fallback:
    """Configuration for Fallback Policy.

    Attributes
    ----------
    fallback_value : Any
        Value to return when operation fails
    """

    fallback_value: Any = None


@policy(name="fallback", description="Provide fallback value when operations fail")
class FallbackPolicy:
    """Simple fallback policy."""

    def __init__(self, fallback_value: Any, **kwargs: Any):
        """Initialize fallback policy with configuration."""
        self.fallback_value = fallback_value

    @property
    def name(self) -> str:
        """Policy name."""
        return "fallback"

    @property
    def priority(self) -> int:
        """Priority - fallback last."""
        return 30

    async def evaluate(self, context: PolicyContext) -> PolicyResponse:
        """Provide fallback on error."""
        if context.error:
            return PolicyResponse(signal=PolicySignal.FALLBACK, data=self.fallback_value)
        return PolicyResponse(signal=PolicySignal.PROCEED)


# ==============================================================================
# ADVANCED CONTROL POLICIES
# ==============================================================================


class Timeout:
    """Configuration for Timeout Policy.

    Attributes
    ----------
    timeout_seconds : float
        Maximum allowed execution time in seconds
    """

    timeout_seconds: float = 30.0


@policy(name="timeout", description="Enforce time limits on node execution")
class TimeoutPolicy:
    """Policy that enforces execution time limits.

    Prevents nodes from running indefinitely by signaling failure
    when execution time exceeds the configured threshold.

    Parameters
    ----------
    timeout_seconds : float
        Maximum allowed execution time in seconds

    Example
    -------
        >>> from hexdag.builtin.adapters.local import LocalPolicyManager
        >>> from hexdag.builtin.policies import TimeoutPolicy
        >>> policy_manager = LocalPolicyManager()
        >>> timeout_policy = TimeoutPolicy(timeout_seconds=30.0)
        >>> policy_manager.subscribe(timeout_policy, SubscriberType.USER)  # doctest: +SKIP
    """

    def __init__(self, timeout_seconds: float = 30.0, **kwargs: Any):
        """Initialize timeout policy with configuration."""
        self.timeout_seconds = timeout_seconds
        self.start_times: dict[str, float] = {}

    @property
    def name(self) -> str:
        """Policy name."""
        return f"timeout_{self.timeout_seconds}s"

    @property
    def priority(self) -> int:
        """Priority - check timeout early."""
        return 8

    async def evaluate(self, context: PolicyContext) -> PolicyResponse:
        """Check for timeout violations.

        Parameters
        ----------
        context : PolicyContext
            Policy evaluation context

        Returns
        -------
        PolicyResponse
            FAIL if timeout exceeded, PROCEED otherwise
        """
        if isinstance(context.event, NodeStarted):
            if context.node_id:
                self.start_times[context.node_id] = context.event.timestamp.timestamp()
            return PolicyResponse(signal=PolicySignal.PROCEED)

        start_time = self.start_times.get(context.node_id) if context.node_id else None
        if start_time and context.event:
            elapsed = context.event.timestamp.timestamp() - start_time
            if elapsed > self.timeout_seconds:
                return PolicyResponse(
                    signal=PolicySignal.FAIL,
                    data={
                        "reason": "timeout",
                        "elapsed_seconds": elapsed,
                        "limit_seconds": self.timeout_seconds,
                    },
                )

        return PolicyResponse(signal=PolicySignal.PROCEED)


class RateLimit:
    """Configuration for Rate Limit Policy."""

    max_executions: int = 10
    window_seconds: float = 60.0


@policy(name="rate_limit", description="Enforce rate limits on node execution")
class RateLimitPolicy:
    """Policy that enforces rate limiting on node execution.

    Prevents too many executions within a time window by tracking
    execution frequency and signaling SKIP when limit is exceeded.

    Parameters
    ----------
    max_executions : int
        Maximum number of executions allowed
    window_seconds : float
        Time window in seconds

    Example
    -------
        >>> rate_limit = RateLimitPolicy(max_executions=10, window_seconds=60.0)
        >>> policy_manager.subscribe(rate_limit, SubscriberType.USER)  # doctest: +SKIP
    """

    def __init__(self, max_executions: int = 10, window_seconds: float = 60.0, **kwargs: Any):
        """Initialize rate limit policy."""
        self.max_executions = max_executions
        self.window_seconds = window_seconds
        self.executions: list[float] = []

    @property
    def name(self) -> str:
        """Policy name."""
        return f"rate_limit_{self.max_executions}_{self.window_seconds}s"

    @property
    def priority(self) -> int:
        """Priority - check rate limits early."""
        return 7

    async def evaluate(self, context: PolicyContext) -> PolicyResponse:
        """Check rate limit compliance.

        Parameters
        ----------
        context : PolicyContext
            Policy evaluation context

        Returns
        -------
        PolicyResponse
            SKIP if rate limit exceeded, PROCEED otherwise
        """
        if not isinstance(context.event, NodeStarted):
            return PolicyResponse(signal=PolicySignal.PROCEED)

        current_time = time.time()

        if self.executions and (
            len(self.executions) >= self.max_executions
            or current_time - self.executions[0] >= self.window_seconds
        ):
            self.executions = [t for t in self.executions if current_time - t < self.window_seconds]

        if len(self.executions) >= self.max_executions:
            return PolicyResponse(
                signal=PolicySignal.SKIP,
                data={
                    "reason": "rate_limit_exceeded",
                    "executions_in_window": len(self.executions),
                    "limit": self.max_executions,
                },
            )

        self.executions.append(current_time)
        return PolicyResponse(signal=PolicySignal.PROCEED)


class ConditionalSkip:
    """Configuration for Conditional Skip Policy."""

    policy_name: str = "conditional_skip"
    # Note: should_skip is a callable, can't be in frozen Config
    # Will be passed as parameter


@policy(name="conditional_skip", description="Skip nodes based on custom conditions")
class ConditionalSkipPolicy:
    """Policy that skips nodes based on custom conditions.

    Provides flexible node filtering based on a user-defined predicate function.

    Parameters
    ----------
    should_skip : callable
        Function(context: PolicyContext) -> bool that returns True to skip

    Example
    -------
        >>> # Skip test nodes
        >>> def skip_tests(ctx):
        ...     return "test" in ctx.node_id.lower()
        >>> skip_policy = ConditionalSkipPolicy(should_skip=skip_tests)
        >>> policy_manager.subscribe(skip_policy, SubscriberType.USER)  # doctest: +SKIP
    """

    def __init__(self, should_skip: Any, policy_name: str = "conditional_skip", **kwargs: Any):
        """Initialize conditional skip policy."""
        self.should_skip = should_skip
        self.policy_name = policy_name

    @property
    def name(self) -> str:
        """Policy name."""
        return self.policy_name

    @property
    def priority(self) -> int:
        """Priority - conditional skips early."""
        return 15

    async def evaluate(self, context: PolicyContext) -> PolicyResponse:
        """Evaluate skip condition.

        Parameters
        ----------
        context : PolicyContext
            Policy evaluation context

        Returns
        -------
        PolicyResponse
            SKIP if condition met, PROCEED otherwise
        """
        if not isinstance(context.event, NodeStarted):
            return PolicyResponse(signal=PolicySignal.PROCEED)

        try:
            if self.should_skip(context):
                return PolicyResponse(
                    signal=PolicySignal.SKIP,
                    data={"reason": "conditional_skip", "policy": self.name},
                )
        except Exception as e:
            # If predicate fails, don't skip (fail safe)
            logger.warning(f"ConditionalSkipPolicy predicate failed: {e}")

        return PolicyResponse(signal=PolicySignal.PROCEED)


class ResourceThrottle:
    """Configuration for Resource Throttle Policy."""

    max_concurrent: int = 5
    resource_cost: int = 1


@policy(name="resource_throttle", description="Throttle execution based on resource availability")
class ResourceThrottlePolicy:
    """Policy that throttles execution based on simulated resource limits.

    Prevents excessive concurrent execution by tracking "resource units"
    and signaling SKIP when resources are exhausted.

    Parameters
    ----------
    max_concurrent : int
        Maximum number of concurrent executions
    resource_cost : int
        Resource cost per execution (default: 1)

    Example
    -------
        >>> throttle = ResourceThrottlePolicy(max_concurrent=5, resource_cost=1)
        >>> policy_manager.subscribe(throttle, SubscriberType.USER)  # doctest: +SKIP
    """

    def __init__(self, max_concurrent: int = 5, resource_cost: int = 1, **kwargs: Any):
        """Initialize resource throttle policy."""
        self.max_concurrent = max_concurrent
        self.resource_cost = resource_cost
        self.current_usage = 0

    @property
    def name(self) -> str:
        """Policy name."""
        return f"resource_throttle_{self.max_concurrent}"

    @property
    def priority(self) -> int:
        """Priority - throttle early."""
        return 6

    async def evaluate(self, context: PolicyContext) -> PolicyResponse:
        """Check resource availability.

        Parameters
        ----------
        context : PolicyContext
            Policy evaluation context

        Returns
        -------
        PolicyResponse
            SKIP if resources exhausted, PROCEED if available
        """
        if isinstance(context.event, NodeStarted):
            if self.current_usage + self.resource_cost > self.max_concurrent:
                return PolicyResponse(
                    signal=PolicySignal.SKIP,
                    data={
                        "reason": "resource_exhausted",
                        "current_usage": self.current_usage,
                        "limit": self.max_concurrent,
                    },
                )
            self.current_usage += self.resource_cost

        elif isinstance(context.event, (NodeCompleted, NodeFailed)):
            self.current_usage = max(0, self.current_usage - self.resource_cost)

        return PolicyResponse(signal=PolicySignal.PROCEED)


class ExponentialBackoff:
    """Configuration for Exponential Backoff Policy."""

    max_retries: int = 5
    initial_delay_ms: float = 100.0
    backoff_factor: float = 2.0
    max_delay_ms: float = 10000.0


@policy(
    name="exponential_backoff",
    description="Retry with exponential backoff on failures",
)
class ExponentialBackoffPolicy:
    """Policy that implements exponential backoff retry strategy.

    Increases wait time exponentially between retries to reduce load
    on failing services.

    Parameters
    ----------
    max_retries : int
        Maximum number of retry attempts
    initial_delay_ms : float
        Initial delay in milliseconds
    backoff_factor : float
        Multiplier for each retry (default: 2.0)
    max_delay_ms : float
        Maximum delay cap in milliseconds

    Example
    -------
        >>> backoff = ExponentialBackoffPolicy(
        ...     max_retries=5,
        ...     initial_delay_ms=100.0,
        ...     backoff_factor=2.0,
        ...     max_delay_ms=10000.0
        ... )
        >>> policy_manager.subscribe(backoff, SubscriberType.USER)  # doctest: +SKIP
    """

    def __init__(
        self,
        max_retries: int = 5,
        initial_delay_ms: float = 100.0,
        backoff_factor: float = 2.0,
        max_delay_ms: float = 10000.0,
        **kwargs: Any,
    ):
        """Initialize exponential backoff policy."""
        self.max_retries = max_retries
        self.initial_delay_ms = initial_delay_ms
        self.backoff_factor = backoff_factor
        self.max_delay_ms = max_delay_ms

    @property
    def name(self) -> str:
        """Policy name."""
        return f"exponential_backoff_{self.max_retries}"

    @property
    def priority(self) -> int:
        """Priority - backoff retries mid-priority."""
        return 12

    async def evaluate(self, context: PolicyContext) -> PolicyResponse:
        """Evaluate retry with exponential backoff.

        Parameters
        ----------
        context : PolicyContext
            Policy evaluation context

        Returns
        -------
        PolicyResponse
            RETRY with delay if attempts remaining, PROCEED otherwise
        """
        if context.error and context.attempt <= self.max_retries:
            # Calculate delay with exponential backoff
            delay_ms = min(
                self.initial_delay_ms * (self.backoff_factor ** (context.attempt - 1)),
                self.max_delay_ms,
            )

            await asyncio.sleep(delay_ms / 1000.0)

            return PolicyResponse(
                signal=PolicySignal.RETRY,
                data={
                    "attempt": context.attempt,
                    "delay_ms": delay_ms,
                    "next_delay_ms": min(delay_ms * self.backoff_factor, self.max_delay_ms),
                },
                metadata={
                    "backoff_strategy": "exponential",
                    "max_retries": self.max_retries,
                },
            )

        return PolicyResponse(signal=PolicySignal.PROCEED)
