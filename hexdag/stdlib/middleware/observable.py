"""Observability middleware for LLM ports.

Wraps any ``SupportsGeneration`` adapter and emits ``LLMPortCall`` events
for every method call.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from hexdag.kernel.context import get_current_node_name, get_observer_manager
from hexdag.kernel.ports.llm import (
    LLMPortCall,
    LLMResponse,
    MessageList,
    SupportsGeneration,
    SupportsStructuredOutput,
    SupportsUsageTracking,
    ToolChoice,
)
from hexdag.kernel.utils.node_timer import Timer

if TYPE_CHECKING:
    from pydantic import BaseModel


def _extract_usage(llm: Any, response_usage: Any | None = None) -> dict[str, int] | None:
    """Extract token usage from a response or the LLM adapter."""
    if response_usage:
        return {
            "input_tokens": response_usage.input_tokens,
            "output_tokens": response_usage.output_tokens,
            "total_tokens": response_usage.total_tokens,
        }
    if isinstance(llm, SupportsUsageTracking) and (u := llm.get_last_usage()):
        return {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "total_tokens": u.total_tokens,
        }
    return None


class ObservableLLM(SupportsGeneration, SupportsStructuredOutput):
    """Observability middleware for LLM ports.

    Wraps an LLM adapter and emits ``LLMPortCall`` events for every call.
    Explicitly implements protocol interfaces so ``isinstance`` checks work.
    """

    def __init__(self, inner: Any) -> None:
        """Initialize with the inner LLM adapter to observe."""
        self._inner = inner

    async def aresponse(self, messages: MessageList) -> str | None:
        """Forward aresponse and emit an LLMPortCall event."""
        node_name = get_current_node_name() or "unknown"
        timer = Timer()
        result = await self._inner.aresponse(messages)
        if mgr := get_observer_manager():
            await mgr.notify(
                LLMPortCall(
                    port_type="llm",
                    method="aresponse",
                    node_name=node_name,
                    duration_ms=timer.duration_ms,
                    usage=_extract_usage(self._inner),
                    model=getattr(self._inner, "model", None),
                    messages=[{"role": m.role, "content": m.content} for m in messages],
                    response=result or "",
                )
            )
        return result  # type: ignore[no-any-return]

    async def aresponse_with_tools(
        self,
        messages: MessageList,
        tools: list[dict[str, Any]],
        tool_choice: ToolChoice | dict[str, Any] = "auto",
        **kwargs: Any,
    ) -> LLMResponse:
        """Forward aresponse_with_tools and emit an LLMPortCall event."""
        node_name = get_current_node_name() or "unknown"
        timer = Timer()
        result = await self._inner.aresponse_with_tools(messages, tools, tool_choice, **kwargs)
        if mgr := get_observer_manager():
            tool_calls_data = None
            if result.tool_calls:
                tool_calls_data = [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in result.tool_calls
                ]
            await mgr.notify(
                LLMPortCall(
                    port_type="llm",
                    method="aresponse_with_tools",
                    node_name=node_name,
                    duration_ms=timer.duration_ms,
                    usage=_extract_usage(self._inner, result.usage),
                    model=getattr(self._inner, "model", None),
                    messages=[{"role": m.role, "content": m.content} for m in messages],
                    response=result.content or "",
                    tool_calls=tool_calls_data,
                )
            )
        return result  # type: ignore[no-any-return]

    async def aresponse_structured(
        self,
        messages: MessageList,
        output_schema: dict[str, Any] | type[BaseModel],
    ) -> dict[str, Any]:
        """Forward aresponse_structured and emit an LLMPortCall event."""
        node_name = get_current_node_name() or "unknown"
        timer = Timer()
        result = await self._inner.aresponse_structured(messages, output_schema)
        if mgr := get_observer_manager():
            await mgr.notify(
                LLMPortCall(
                    port_type="llm",
                    method="aresponse_structured",
                    node_name=node_name,
                    duration_ms=timer.duration_ms,
                    usage=_extract_usage(self._inner),
                    model=getattr(self._inner, "model", None),
                    messages=[{"role": m.role, "content": m.content} for m in messages],
                    response=json.dumps(result),
                )
            )
        return result  # type: ignore[no-any-return]

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the inner adapter."""
        return getattr(self._inner, name)
