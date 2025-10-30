"""Port wrappers that provide observability and control.

These wrappers intercept port method calls to:
1. **Observability** - Emit events (LLMPromptSent, LLMResponseReceived, etc.)
2. **Control** - Enable policy-based control (rate limiting, caching, retry, fallback)
3. **Metrics** - Track duration and performance
4. **Extensibility** - Provide hooks for custom behavior

This allows centralized control over infrastructure interactions without
polluting business logic in nodes and macros.
"""

import time
from typing import Any

from hexdag.core.context import get_current_node_name, get_observer_manager
from hexdag.core.logging import get_logger
from hexdag.core.orchestration.events import (
    LLMPromptSent,
    LLMResponseReceived,
    ToolCalled,
    ToolCompleted,
)
from hexdag.core.ports.llm import LLM, LLMResponse, MessageList

logger = get_logger(__name__)


class ObservableLLMWrapper:
    """Wraps an LLM port for observability and policy-based control.

    This wrapper provides:
    - Automatic event emission for all LLM calls
    - Policy evaluation before/after calls (future: rate limiting, caching)
    - Duration tracking and performance metrics
    - Transparent forwarding to underlying LLM
    """

    def __init__(self, llm: LLM):
        """Initialize wrapper with underlying LLM port.

        Parameters
        ----------
        llm : LLM
            The underlying LLM port to wrap
        """
        self._llm = llm

    async def aresponse(self, messages: MessageList, **kwargs: Any) -> str | None:
        """Call LLM with observability and policy control.

        This method:
        1. Emits LLMPromptSent event (for observability)
        2. Calls underlying LLM
        3. Emits LLMResponseReceived event with duration
        4. (Future: Policy evaluation for rate limiting, caching, retry)

        Parameters
        ----------
        messages : MessageList
            Messages to send to LLM
        **kwargs : Any
            Additional LLM parameters

        Returns
        -------
        str | None
            LLM response text
        """
        node_name = get_current_node_name() or "unknown"

        # Emit prompt sent event (OBSERVABILITY)
        if observer_mgr := get_observer_manager():
            await observer_mgr.notify(
                LLMPromptSent(
                    node_name=node_name,
                    messages=[{"role": m.role, "content": m.content} for m in messages],
                )
            )

        # TODO(hexdag-team): Add policy evaluation here (CONTROL) #noqa: TD003
        # if policy_mgr := get_policy_manager():
        #     policy_response = await policy_mgr.evaluate(LLMCallPolicy(...))
        #     if policy_response.signal == PolicySignal.SKIP:
        #         return policy_response.data  # Cached response
        #     elif policy_response.signal == PolicySignal.FAIL:
        #         raise RateLimitError("LLM rate limit exceeded")

        # Call underlying LLM
        start_time = time.perf_counter()
        response = await self._llm.aresponse(messages, **kwargs)
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Emit response received event (OBSERVABILITY)
        if observer_mgr := get_observer_manager():
            await observer_mgr.notify(
                LLMResponseReceived(
                    node_name=node_name, response=response or "", duration_ms=duration_ms
                )
            )

        return response

    async def aresponse_with_tools(
        self,
        messages: MessageList,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
        **kwargs: Any,
    ) -> LLMResponse:
        """Call LLM with tools and emit events.

        Parameters
        ----------
        messages : MessageList
            Messages to send to LLM
        tools : list[dict[str, Any]]
            Tool definitions
        tool_choice : str | dict[str, Any]
            Tool choice strategy
        **kwargs : Any
            Additional LLM parameters

        Returns
        -------
        LLMResponse
            Structured response with tool calls
        """
        node_name = get_current_node_name() or "unknown"

        # Emit prompt sent event
        if observer_mgr := get_observer_manager():
            await observer_mgr.notify(
                LLMPromptSent(
                    node_name=node_name,
                    messages=[{"role": m.role, "content": m.content} for m in messages],
                )
            )

        # Call underlying LLM
        start_time = time.perf_counter()
        response = await self._llm.aresponse_with_tools(messages, tools, tool_choice, **kwargs)
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Emit response received event
        if observer_mgr := get_observer_manager():
            await observer_mgr.notify(
                LLMResponseReceived(
                    node_name=node_name, response=response.content or "", duration_ms=duration_ms
                )
            )

        return response

    def __getattr__(self, name: str) -> Any:
        """Forward all other attribute access to underlying LLM."""
        return getattr(self._llm, name)


class ObservableToolRouterWrapper:
    """Wraps a ToolRouter port for observability and policy-based control.

    This wrapper provides:
    - Automatic event emission for all tool calls (ToolCalled, ToolCompleted)
    - Policy evaluation before/after calls (future: auth, rate limiting)
    - Duration tracking and performance metrics
    - Error handling and logging
    """

    def __init__(self, tool_router: Any):
        """Initialize wrapper with underlying tool router.

        Parameters
        ----------
        tool_router : Any
            The underlying tool router to wrap
        """
        self._tool_router = tool_router

    async def acall_tool(self, tool_name: str, params: dict[str, Any]) -> Any:
        """Call tool with observability and policy control.

        This method:
        1. Emits ToolCalled event (for observability)
        2. Calls underlying tool
        3. Emits ToolCompleted event with duration and result
        4. (Future: Policy evaluation for auth, rate limiting)

        Parameters
        ----------
        tool_name : str
            Name of the tool to call
        params : dict[str, Any]
            Tool parameters

        Returns
        -------
        Any
            Tool execution result
        """
        node_name = get_current_node_name() or "unknown"

        # Emit tool called event (OBSERVABILITY)
        if observer_mgr := get_observer_manager():
            await observer_mgr.notify(
                ToolCalled(node_name=node_name, tool_name=tool_name, params=params)
            )

        # TODO(hexdag-team): Add policy evaluation here (CONTROL) #noqa: TD003
        # if policy_mgr := get_policy_manager():
        #     policy_response = await policy_mgr.evaluate(ToolCallPolicy(...))
        #     if policy_response.signal == PolicySignal.FAIL:
        #         raise AuthorizationError(f"Tool '{tool_name}' not authorized")

        # Call underlying tool
        start_time = time.perf_counter()
        try:
            result = await self._tool_router.acall_tool(tool_name, params)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Emit tool completed event (OBSERVABILITY)
            if observer_mgr := get_observer_manager():
                await observer_mgr.notify(
                    ToolCompleted(
                        node_name=node_name,
                        tool_name=tool_name,
                        result=result,
                        duration_ms=duration_ms,
                    )
                )

            return result

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Tool '{tool_name}' failed in {duration_ms:.2f}ms: {e}")

            # Still emit completed event with error info
            if observer_mgr := get_observer_manager():
                await observer_mgr.notify(
                    ToolCompleted(
                        node_name=node_name,
                        tool_name=tool_name,
                        result={"error": str(e)},
                        duration_ms=duration_ms,
                    )
                )

            raise

    def __getattr__(self, name: str) -> Any:
        """Forward all other attribute access to underlying tool router."""
        return getattr(self._tool_router, name)


def wrap_llm_port(llm: Any) -> Any:
    """Wrap LLM port with event emission if it implements the LLM protocol.

    Parameters
    ----------
    llm : Any
        The LLM port to potentially wrap

    Returns
    -------
    Any
        Wrapped LLM if it has aresponse method, otherwise original object
    """
    if hasattr(llm, "aresponse"):
        return ObservableLLMWrapper(llm)
    return llm


def wrap_tool_router_port(tool_router: Any) -> Any:
    """Wrap ToolRouter port with event emission.

    Parameters
    ----------
    tool_router : Any
        The tool router to potentially wrap

    Returns
    -------
    Any
        Wrapped tool router if it has acall_tool method, otherwise original object
    """
    if hasattr(tool_router, "acall_tool"):
        return ObservableToolRouterWrapper(tool_router)
    return tool_router


def wrap_ports_with_observability(ports: dict[str, Any]) -> dict[str, Any]:
    """Wrap all ports with event-emitting wrappers.

    This wraps:
    - LLM ports: Emit LLMPromptSent/LLMResponseReceived events
    - ToolRouter ports: Emit ToolCalled/ToolCompleted events
    - (Future: Database, Memory, etc.)

    Parameters
    ----------
    ports : dict[str, Any]
        Dictionary of port name to port adapter

    Returns
    -------
    dict[str, Any]
        Dictionary with wrapped ports
    """
    wrapped = {}
    for name, port in ports.items():
        if name == "llm":
            wrapped[name] = wrap_llm_port(port)
        elif name == "tool_router":
            wrapped[name] = wrap_tool_router_port(port)
        else:
            wrapped[name] = port
    return wrapped
