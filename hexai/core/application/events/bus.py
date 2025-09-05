"""Event bus for control-flow event handling with return values."""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import Any, Callable

from .base import BasePriorityEventDispatcher, HandlerFunction, PipelineEvent

logger = logging.getLogger(__name__)


class HandlerPriority(IntEnum):
    """Priority levels for event bus handlers."""

    CRITICAL = -1000  # System-critical handlers (circuit breakers, security)
    HIGH = -100  # High priority handlers (validation, authentication)
    NORMAL = 0  # Default priority (business logic)
    LOW = 100  # Low priority handlers (logging, metrics)
    BACKGROUND = 1000  # Background tasks (cleanup, archiving)


class EventBus(BasePriorityEventDispatcher):
    """Event bus for control-flow handlers that can modify execution.

    Unlike PipelineEventManager (observability), this bus is designed for
    handlers that can affect pipeline execution - policies, circuit breakers,
    validation, etc. Handlers execute sequentially by priority and can return
    values to influence control flow.
    """

    def subscribe(
        self,
        event_type: type[PipelineEvent] | None,
        handler: HandlerFunction,
        priority: HandlerPriority | int = HandlerPriority.NORMAL,
        name: str = "",
        filter_func: Callable[[PipelineEvent], bool] | None = None,
    ) -> None:
        """Subscribe a handler with priority.

        Parameters
        ----------
        event_type : type[PipelineEvent] or None
            Event class to subscribe to (None for all events)
        handler : HandlerFunction
            Async handler function
        priority : HandlerPriority or int
            Handler priority (enum or int)
        name : str
            Optional handler name for debugging
        filter_func : callable or None
            Optional custom filter function
        """
        if isinstance(priority, HandlerPriority):
            priority_value = priority.value
        else:
            priority_value = priority

        handler_name = name or getattr(handler, "__name__", "anonymous")

        self._add_handler(
            handler=handler,
            event_filter=event_type,
            priority=priority_value,
            name=handler_name,
            filter_func=filter_func,
        )

        logger.debug(
            f"Subscribed handler '{handler_name}' to "
            f"{getattr(event_type, '__name__', 'all events')} "
            f"with priority {priority_value}"
        )

    async def publish(self, event: PipelineEvent) -> list[Any]:
        """Publish event and collect return values from handlers.

        Parameters
        ----------
        event : PipelineEvent
            Event to publish

        Returns
        -------
        list[Any]
            List of return values from handlers (in priority order)
        """
        await self._ensure_lock()
        if self._lock is None:  # Type guard for mypy
            raise RuntimeError("Lock not initialized")

        async with self._lock:
            applicable_handlers = self._get_applicable_handlers(event)

            results = []
            for prioritized in applicable_handlers:
                handler = prioritized.handler

                try:
                    result = await handler(event)
                    results.append(result)

                    logger.debug(
                        f"Handler '{prioritized.name}' processed {type(event).__name__}: {result}"
                    )
                except Exception as e:
                    logger.error(
                        f"Handler '{prioritized.name}' failed processing "
                        f"{type(event).__name__}: {e}"
                    )
                    # Re-raise for control flow handlers
                    raise

            return results

    def unsubscribe(self, handler_name: str) -> None:
        """Unsubscribe handler by name.

        Parameters
        ----------
        handler_name : str
            Name of handler to remove
        """
        self._remove_handler(handler_name)
        logger.debug(f"Unsubscribed handler '{handler_name}'")
