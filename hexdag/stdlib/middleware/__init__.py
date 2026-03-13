"""Port middleware — transparent layers stacked on adapters.

Middleware falls into two categories:

**Auto-middleware** (framework-managed, always outermost):
    ``StructuredOutputFallback``, ``ObservableLLM``, ``ObservableToolRouter``.
    These explicitly implement protocol interfaces for ``isinstance`` checks.

**User middleware** (declared in YAML, applied inner-to-outer):
    ``RetryWithBackoff``, ``RateLimiter``, ``ResponseCache``, ``Timeout``.
    Configured via ``spec.ports.<name>.middleware:`` or ``kind: Middleware``.

Full stack order (inner → outer)::

    adapter → [Cache → RateLimiter → Retry → Timeout]
            → StructuredOutputFallback (if needed) → ObservableLLM

Example
-------
::

    from hexdag.stdlib.middleware import compose

    port = compose(
        MockLLM(),
        ResponseCache,             # cache identical calls
        RateLimiter,               # throttle API calls
        RetryWithBackoff,          # retry transient failures
        Timeout,                   # enforce time limits
        StructuredOutputFallback,  # adds SupportsStructuredOutput if missing
        ObservableLLM,             # adds event emission (always outermost)
    )
"""

from hexdag.stdlib.middleware.compose import compose
from hexdag.stdlib.middleware.observable import ObservableLLM
from hexdag.stdlib.middleware.observable_tool_router import ObservableToolRouter
from hexdag.stdlib.middleware.rate_limiter import RateLimiter
from hexdag.stdlib.middleware.response_cache import ResponseCache
from hexdag.stdlib.middleware.retry import RetryWithBackoff
from hexdag.stdlib.middleware.timeout import Timeout

__all__ = [
    # Auto-middleware (framework-managed)
    "ObservableLLM",
    "ObservableToolRouter",
    # User middleware (YAML-configurable)
    "RetryWithBackoff",
    "RateLimiter",
    "ResponseCache",
    "Timeout",
    # Utilities
    "compose",
]
