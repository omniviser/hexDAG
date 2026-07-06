"""Round-robin load balancer across multiple LLM adapter instances.

Distributes LLM calls across N adapter instances with automatic failover.
Works with any adapter that implements ``SupportsGeneration`` (and optionally
``SupportsStructuredOutput`` / ``SupportsFunctionCalling``).

Example
-------
Python usage::

    from hexdag.stdlib.adapters.google import VertexAIAdapter
    from hexdag.stdlib.middleware import RoundRobin

    adapters = [
        VertexAIAdapter(credentials_json=sa1, project_id=pid1),
        VertexAIAdapter(credentials_json=sa2, project_id=pid2),
        VertexAIAdapter(credentials_json=sa3, project_id=pid3),
    ]
    llm = RoundRobin(adapters)

    # Calls are distributed across the three adapters with failover
    result = await llm.aresponse(messages)

Declarative YAML pool (built automatically by the orchestrator factory)::

    spec:
      ports:
        llm:
          adapters:
            - adapter: llm:vertex
              config: {credentials_json: "${GCP_AI1_SA_JSON}", project_id: "${GCP_AI1_PROJECT_ID}"}
            - adapter: llm:vertex
              config: {credentials_json: "${GCP_AI2_SA_JSON}", project_id: "${GCP_AI2_PROJECT_ID}"}
          strategy: round_robin   # or: failover
"""

import itertools
from typing import Any, Literal

from pydantic import BaseModel

from hexdag.kernel.logging import get_logger
from hexdag.kernel.ports.llm import (
    LLMResponse,
    MessageList,
    SupportsFunctionCalling,
    SupportsGeneration,
    SupportsStructuredOutput,
)

logger = get_logger(__name__)


class RoundRobin(
    SupportsGeneration,
    SupportsStructuredOutput,
    SupportsFunctionCalling,
):
    """Round-robin load balancer across multiple LLM adapter instances.

    Distributes calls across N adapters. If the selected adapter fails,
    the next adapter in the rotation is tried (up to ``len(adapters)``
    attempts). Each adapter may have its own internal retry logic, so
    this outer failover only fires after the adapter itself gives up.

    The pool advertises ``SupportsGeneration``/``SupportsStructuredOutput``/
    ``SupportsFunctionCalling`` regardless of its members, so members must
    have homogeneous capabilities: a member missing a called capability
    raises ``TypeError`` (and automatic ``StructuredOutputFallback`` port
    wrapping is skipped for pools).

    Parameters
    ----------
    adapters : list[SupportsGeneration]
        LLM adapter instances to balance across. Must contain at least
        one adapter.
    failover : bool
        When ``True`` (default), try the next adapter on failure.
        When ``False``, fail immediately.
    strategy : {"round_robin", "failover"}
        ``round_robin`` (default) rotates the starting adapter on every
        call. ``failover`` always starts at the first adapter, spilling
        over to the others only on failure (deterministic primary).
    """

    VALID_STRATEGIES = frozenset({"round_robin", "failover"})

    def __init__(
        self,
        adapters: list[SupportsGeneration],
        failover: bool = True,
        strategy: Literal["round_robin", "failover"] = "round_robin",
    ) -> None:
        """Initialize the round-robin balancer with a list of adapters."""
        if not adapters:
            raise ValueError("RoundRobin requires at least one adapter")
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy {strategy!r}; expected one of {sorted(self.VALID_STRATEGIES)}"
            )
        self._adapters = adapters
        self._failover = failover
        self._strategy = strategy
        self._counter = itertools.count()

        logger.info(
            "RoundRobin initialized with %d adapter(s), failover=%s, strategy=%s",
            len(self._adapters),
            self._failover,
            self._strategy,
        )

    def _next_index(self) -> int:
        """Return the starting adapter index for the next call."""
        if self._strategy == "failover":
            return 0
        return next(self._counter) % len(self._adapters)

    async def _call_with_failover(
        self,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call the named method on the next adapter, failing over on error.

        Tries up to ``len(self._adapters)`` adapters when failover is enabled.
        """
        start_idx = self._next_index()
        max_attempts = len(self._adapters) if self._failover else 1
        last_error: Exception | None = None

        for offset in range(max_attempts):
            idx = (start_idx + offset) % len(self._adapters)
            adapter = self._adapters[idx]
            logger.info(
                "RoundRobin dispatch: slot=%d method=%s",
                idx,
                method_name,
            )
            method = getattr(adapter, method_name, None)
            if method is None:
                raise TypeError(
                    f"Pool member {type(adapter).__name__} (slot {idx}) does not "
                    f"support {method_name}; pool members must have homogeneous "
                    f"capabilities"
                )
            try:
                result = await method(*args, **kwargs)
                if method_name == "aresponse" and result is None:
                    logger.warning(
                        "Adapter %d returned None for %s, trying next",
                        idx,
                        method_name,
                    )
                    last_error = RuntimeError("aresponse returned None")
                    continue
                return result
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Adapter %d failed for %s: %s, trying next",
                    idx,
                    method_name,
                    exc,
                )

        logger.error(
            "All %d adapters failed for %s",
            len(self._adapters),
            method_name,
        )
        if method_name == "aresponse":
            return None
        raise last_error  # type: ignore[misc]

    # ------------------------------------------------------------------
    # SupportsGeneration
    # ------------------------------------------------------------------

    async def aresponse(self, messages: MessageList) -> str | None:
        """Generate a text response via the next adapter in rotation."""
        result: str | None = await self._call_with_failover("aresponse", messages)
        return result

    # ------------------------------------------------------------------
    # SupportsStructuredOutput
    # ------------------------------------------------------------------

    async def aresponse_structured(
        self,
        messages: MessageList,
        output_schema: dict[str, Any] | type[BaseModel],
    ) -> dict[str, Any]:
        """Generate a schema-conforming response via the next adapter in rotation."""
        result: dict[str, Any] = await self._call_with_failover(
            "aresponse_structured",
            messages,
            output_schema,
        )
        return result

    # ------------------------------------------------------------------
    # SupportsFunctionCalling
    # ------------------------------------------------------------------

    async def aresponse_with_tools(
        self,
        messages: MessageList,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
    ) -> LLMResponse:
        """Generate a tool-calling response via the next adapter in rotation."""
        result: LLMResponse = await self._call_with_failover(
            "aresponse_with_tools",
            messages,
            tools,
            tool_choice=tool_choice,
        )
        return result

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """Close all underlying adapters."""
        for adapter in self._adapters:
            close = getattr(adapter, "aclose", None)
            if close is not None:
                await close()

    async def clear_cache(self) -> None:
        """Clear cache on all underlying adapters (if supported)."""
        for adapter in self._adapters:
            clear = getattr(adapter, "clear_cache", None)
            if clear is not None:
                await clear()
