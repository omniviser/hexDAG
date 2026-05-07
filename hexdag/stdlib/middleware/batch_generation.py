"""Batch generation middleware with concurrency control.

Wraps any LLM port and provides:

1. **Concurrency limiting** — all ``aresponse()`` calls are gated by an
   ``asyncio.Semaphore``, preventing thundering-herd API overload when
   many nodes in a wave call the LLM concurrently.

2. **Explicit batch** — ``aresponse_batch()`` fires multiple requests via
   ``asyncio.gather()`` (like LangChain's ``batch()``), returning a
   :class:`~hexdag.kernel.ports.llm.BatchResult` with per-item status.

Example YAML::

    spec:
      ports:
        llm:
          adapter: hexdag.stdlib.adapters.openai.OpenAIAdapter
          middleware:
            - hexdag.stdlib.middleware.batch_generation.BatchGeneration
"""

from __future__ import annotations

import asyncio
from typing import Any

from hexdag.kernel.logging import get_logger
from hexdag.kernel.ports.llm import (
    BatchItemResult,
    BatchItemStatus,
    BatchResult,
    MessageList,
    SupportsUsageTracking,
)

logger = get_logger(__name__)

_DEFAULT_MAX_CONCURRENCY = 10


class BatchGeneration:
    """Middleware that provides batch generation and concurrency control.

    All LLM calls are gated by an ``asyncio.Semaphore`` so that at most
    ``max_concurrency`` requests fly in parallel.  The ``aresponse_batch()``
    method fires multiple requests via ``asyncio.gather()`` — like
    LangChain's ``batch()``.

    Parameters
    ----------
    inner : Any
        The port/adapter to wrap.
    max_concurrency : int
        Maximum parallel ``aresponse()`` calls (default 10).
    """

    def __init__(
        self,
        inner: Any,
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        self._inner = inner
        self._max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._active_calls = 0
        self._total_calls = 0

    @property
    def active_calls(self) -> int:
        """Number of currently in-flight calls."""
        return self._active_calls

    @property
    def total_calls(self) -> int:
        """Total number of calls processed."""
        return self._total_calls

    # -- Batch method (SupportsBatchGeneration) --

    async def aresponse_batch(
        self,
        message_lists: list[MessageList],
    ) -> BatchResult:
        """Generate responses for multiple message lists concurrently.

        Fires all requests via ``asyncio.gather()``, gated by the semaphore.
        Individual item failures are captured as ``FAILED`` items — one
        failure does not abort the entire batch.

        Parameters
        ----------
        message_lists : list[MessageList]
            Independent conversations to process in batch.

        Returns
        -------
        BatchResult
            Aggregated results with per-item status and usage.
        """

        async def _call_one(index: int, messages: MessageList) -> BatchItemResult:
            async with self._semaphore:
                self._active_calls += 1
                self._total_calls += 1
                try:
                    content = await self._inner.aresponse(messages)
                    usage = None
                    if isinstance(self._inner, SupportsUsageTracking):
                        usage = self._inner.get_last_usage()
                    return BatchItemResult(
                        index=index,
                        content=content,
                        status=BatchItemStatus.COMPLETED,
                        usage=usage,
                    )
                except Exception as exc:
                    logger.warning("Batch item {} failed: {}", index, exc)
                    return BatchItemResult(
                        index=index,
                        content=None,
                        status=BatchItemStatus.FAILED,
                        error=str(exc),
                    )
                finally:
                    self._active_calls -= 1

        items = list(
            await asyncio.gather(
                *[_call_one(i, msgs) for i, msgs in enumerate(message_lists)],
            )
        )

        total_usage = BatchResult.aggregate_usage([item.usage for item in items])

        return BatchResult(
            items=items,
            total_usage=total_usage,
            provider="gather",
        )

    # -- LLM protocol methods (semaphore-gated) --

    async def aresponse(self, *args: Any, **kwargs: Any) -> Any:
        """Concurrency-limited aresponse."""
        async with self._semaphore:
            self._active_calls += 1
            self._total_calls += 1
            try:
                return await self._inner.aresponse(*args, **kwargs)
            finally:
                self._active_calls -= 1

    async def aresponse_with_tools(self, *args: Any, **kwargs: Any) -> Any:
        """Concurrency-limited aresponse_with_tools."""
        async with self._semaphore:
            self._active_calls += 1
            self._total_calls += 1
            try:
                return await self._inner.aresponse_with_tools(*args, **kwargs)
            finally:
                self._active_calls -= 1

    async def aresponse_structured(self, *args: Any, **kwargs: Any) -> Any:
        """Concurrency-limited aresponse_structured."""
        async with self._semaphore:
            self._active_calls += 1
            self._total_calls += 1
            try:
                return await self._inner.aresponse_structured(*args, **kwargs)
            finally:
                self._active_calls -= 1

    # -- ToolRouter protocol methods --

    async def acall_tool(self, *args: Any, **kwargs: Any) -> Any:
        """Concurrency-limited acall_tool."""
        async with self._semaphore:
            self._active_calls += 1
            self._total_calls += 1
            try:
                return await self._inner.acall_tool(*args, **kwargs)
            finally:
                self._active_calls -= 1

    # -- Passthrough --

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the inner port."""
        return getattr(self._inner, name)
