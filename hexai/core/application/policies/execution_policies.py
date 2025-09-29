"""Simple policy implementations."""

from typing import Any

from hexai.core.registry.decorators import policy

from .models import PolicyContext, PolicyResponse, PolicySignal


@policy(name="retry", description="Retry failed operations up to a maximum number of attempts")
class RetryPolicy:
    """Simple retry policy."""

    def __init__(self, max_retries: int = 3):
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


@policy(
    name="circuit_breaker",
    description="Fail fast after reaching failure threshold to prevent cascading failures",
)
class CircuitBreakerPolicy:
    """Simple circuit breaker."""

    def __init__(self, failure_threshold: int = 5):
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


@policy(name="fallback", description="Provide fallback value when operations fail")
class FallbackPolicy:
    """Simple fallback policy."""

    def __init__(self, fallback_value: Any):
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
