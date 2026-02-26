"""Port wrappers that provide observability and control.

These wrappers intercept port method calls to:
1. **Observability** - Emit events (LLMGeneration, LLMFunctionCalling, ToolRouterEvent, etc.)
2. **Control** - Enable policy-based control (rate limiting, caching, retry, fallback)
3. **Metrics** - Track duration and performance
4. **Extensibility** - Provide hooks for custom behavior

This allows centralized control over infrastructure interactions without
polluting business logic in nodes and macros.
"""

from typing import Any

from hexdag.kernel.context import get_current_node_name, get_observer_manager
from hexdag.kernel.logging import get_logger
from hexdag.kernel.ports.llm import (
    LLM,
    LLMFunctionCalling,
    LLMGeneration,
    LLMResponse,
    MessageList,
    SupportsGeneration,
    SupportsUsageTracking,
    ToolChoice,
)
from hexdag.kernel.ports.tool_router import ToolRouter, ToolRouterEvent
from hexdag.kernel.utils.node_timer import Timer

logger = get_logger(__name__)


def _extract_usage_dict(llm: Any, response_usage: Any | None = None) -> dict[str, int] | None:
    """Extract usage dict from LLMResponse.usage or SupportsUsageTracking."""
    if response_usage:
        return {
            "input_tokens": response_usage.input_tokens,
            "output_tokens": response_usage.output_tokens,
            "total_tokens": response_usage.total_tokens,
        }
    if isinstance(llm, SupportsUsageTracking) and (last_usage := llm.get_last_usage()):
        return {
            "input_tokens": last_usage.input_tokens,
            "output_tokens": last_usage.output_tokens,
            "total_tokens": last_usage.total_tokens,
        }
    return None


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

        # TODO(hexdag-team): Add policy evaluation here (CONTROL) #noqa: TD003

        # Call underlying LLM (capability validated at mount time by orchestrator)
        llm_timer = Timer()
        response = await self._llm.aresponse(messages, **kwargs)  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
        duration_ms = llm_timer.duration_ms

        usage_dict = _extract_usage_dict(self._llm)

        # Emit single LLMGeneration event (OBSERVABILITY)
        if observer_mgr := get_observer_manager():
            await observer_mgr.notify(
                LLMGeneration(
                    node_name=node_name,
                    duration_ms=duration_ms,
                    usage=usage_dict,
                    model=getattr(self._llm, "model", None),
                    messages=[{"role": m.role, "content": m.content} for m in messages],
                    response=response or "",
                )
            )

        return response  # type: ignore[no-any-return]

    async def aresponse_with_tools(
        self,
        messages: MessageList,
        tools: list[dict[str, Any]],
        tool_choice: ToolChoice | dict[str, Any] = "auto",
        **kwargs: Any,
    ) -> LLMResponse:
        """Call LLM with tools and emit events.

        Parameters
        ----------
        messages : MessageList
            Messages to send to LLM
        tools : list[dict[str, Any]]
            Tool definitions
        tool_choice : ToolChoice | dict[str, Any]
            Tool choice strategy
        **kwargs : Any
            Additional LLM parameters

        Returns
        -------
        LLMResponse
            Structured response with tool calls
        """
        node_name = get_current_node_name() or "unknown"

        # Call underlying LLM (capability validated at mount time by orchestrator)
        llm_timer = Timer()
        response = await self._llm.aresponse_with_tools(messages, tools, tool_choice, **kwargs)  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
        duration_ms = llm_timer.duration_ms

        usage_dict = _extract_usage_dict(self._llm, response.usage)

        # Emit single LLMFunctionCalling event (OBSERVABILITY)
        if observer_mgr := get_observer_manager():
            tool_calls_data = None
            if response.tool_calls:
                tool_calls_data = [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in response.tool_calls
                ]
            await observer_mgr.notify(
                LLMFunctionCalling(
                    node_name=node_name,
                    duration_ms=duration_ms,
                    usage=usage_dict,
                    model=getattr(self._llm, "model", None),
                    messages=[{"role": m.role, "content": m.content} for m in messages],
                    response=response.content or "",
                    tool_calls=tool_calls_data,
                )
            )

        return response  # type: ignore[no-any-return]

    def __getattr__(self, name: str) -> Any:
        """Forward all other attribute access to underlying LLM."""
        return getattr(self._llm, name)


class ObservableToolRouterWrapper:
    """Wraps a ToolRouter port for observability and policy-based control.

    This wrapper provides:
    - Automatic event emission for all tool calls (ToolRouterEvent)
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

        # TODO(hexdag-team): Add policy evaluation here (CONTROL) #noqa: TD003

        # Call underlying tool
        tool_timer = Timer()
        try:
            result = await self._tool_router.acall_tool(tool_name, params)

            # Emit single ToolRouterEvent (OBSERVABILITY)
            if observer_mgr := get_observer_manager():
                await observer_mgr.notify(
                    ToolRouterEvent(
                        node_name=node_name,
                        tool_name=tool_name,
                        params=params,
                        result=result,
                        duration_ms=tool_timer.duration_ms,
                    )
                )

            return result

        except Exception as e:
            logger.error("Tool '{}' failed in {}ms: {}", tool_name, tool_timer.duration_str, e)

            # Still emit event with error info
            if observer_mgr := get_observer_manager():
                await observer_mgr.notify(
                    ToolRouterEvent(
                        node_name=node_name,
                        tool_name=tool_name,
                        params=params,
                        result={"error": str(e)},
                        duration_ms=tool_timer.duration_ms,
                    )
                )

            raise

    def __getattr__(self, name: str) -> Any:
        """Forward all other attribute access to underlying tool router."""
        return getattr(self._tool_router, name)


def wrap_llm_port(llm: Any) -> Any:
    """Wrap LLM port with event emission if it implements SupportsGeneration.

    Parameters
    ----------
    llm : Any
        The LLM port to potentially wrap

    Returns
    -------
    Any
        Wrapped LLM if it supports generation, otherwise original object
    """
    if isinstance(llm, SupportsGeneration):
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
        Wrapped tool router if it is a ToolRouter, otherwise original object
    """
    if isinstance(tool_router, ToolRouter):
        return ObservableToolRouterWrapper(tool_router)
    return tool_router


def wrap_ports_with_observability(ports: dict[str, Any]) -> dict[str, Any]:
    """Wrap all ports with event-emitting wrappers.

    Uses ``isinstance`` protocol checks to detect wrappable ports,
    regardless of their dictionary key.

    Currently wraps:
    - LLM ports (``SupportsGeneration``): Emit LLMGeneration/LLMFunctionCalling
    - ToolRouter ports: Emit ToolRouterEvent

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
        if isinstance(port, SupportsGeneration):
            wrapped[name] = wrap_llm_port(port)
        elif isinstance(port, ToolRouter):
            wrapped[name] = wrap_tool_router_port(port)
        else:
            wrapped[name] = port
    return wrapped
