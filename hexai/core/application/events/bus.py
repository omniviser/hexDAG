"""Control bus for execution control - can veto/modify execution."""

import logging
from typing import Any, Awaitable, Callable, Protocol

logger = logging.getLogger(__name__)

# Handler returns True to allow, False to veto
Handler = Callable[[Any], Awaitable[bool]]


class ControlHandler(Protocol):
    """Protocol for control handlers that can veto execution."""

    async def check(self, event: Any) -> bool:
        """Check if event should be allowed.

        Returns
        -------
            True to allow execution, False to veto.
        """
        ...


class EventBus:
    """Manages control handlers that can affect execution.

    Key principles:
    - Handlers can VETO execution by returning False
    - Used for policies, circuit breakers, rate limiting
    - Handlers are executed in priority order
    - All handlers must approve for execution to continue
    """

    def __init__(self) -> None:
        # Simple list of handlers - all handlers see all events
        self._handlers: list[Handler | ControlHandler] = []

    def register(self, handler: Handler | ControlHandler) -> None:
        """Register a control handler.

        Args
        ----
            handler: Function or ControlHandler that returns True to allow, False to veto
        """
        self._handlers.append(handler)

    def unregister(self, handler: Handler | ControlHandler) -> None:
        """Remove a handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)

    async def check(self, event: Any) -> bool:
        """Check if event is allowed by all control handlers.

        Returns
        -------
            True if all handlers approve (or no handlers), False if any vetoes
        """
        # No handlers means allow all
        if not self._handlers:
            return True

        for handler in self._handlers:
            try:
                # Check if it's a ControlHandler class
                if hasattr(handler, "check"):
                    result = await handler.check(event)
                else:
                    # It's a function
                    result = await handler(event)

                if not result:
                    handler_name = getattr(handler, "__name__", handler.__class__.__name__)
                    logger.info(f"Handler vetoed {event.__class__.__name__}: {handler_name}")
                    return False
            except Exception as e:
                handler_name = getattr(handler, "__name__", handler.__class__.__name__)
                logger.error(f"Control handler {handler_name} failed: {e}")
                # On error, we veto for safety
                return False

        return True

    def clear(self) -> None:
        """Clear all handlers."""
        self._handlers.clear()

    def has_handlers(self) -> bool:
        """Check if there are any handlers registered."""
        return bool(self._handlers)


# Example handlers
async def circuit_breaker(event: Any) -> bool:
    """Circuit breaker that can stop execution on too many failures."""
    # This would check failure rates and return False if circuit is open
    return True


async def rate_limiter(event: Any) -> bool:
    """Rate limiter for API calls."""
    # This would check rate limits and return False if exceeded
    return True


async def policy_checker(event: Any) -> bool:
    """Check if event violates any policies."""
    # This would check business rules and return False if violated
    return True
