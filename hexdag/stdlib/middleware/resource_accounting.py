"""Resource accounting middleware — enforces per-pipeline resource limits.

An **observer** that listens to ``PortCallEvent`` subtypes, accumulates
resource usage in a ``ResourceUsage`` model, checks against configured
``ResourceLimits``, and emits ``ResourceWarning`` / ``ResourceLimitExceeded``
events when thresholds are crossed.

Register with the observer manager::

    from hexdag.stdlib.middleware.resource_accounting import ResourceAccountingObserver
    from hexdag.kernel.domain.resource_accounting import ResourceLimits

    accounting = ResourceAccountingObserver(
        limits=ResourceLimits(max_total_tokens=100_000, max_llm_calls=50),
        pipeline_name="my-pipeline",
    )
    observer_manager.register(accounting, event_types=(PortCallEvent,))

Example YAML (pipeline-level limits via ``spec.resource_limits``)::

    spec:
      resource_limits:
        max_total_tokens: 100000
        max_llm_calls: 50
        warning_threshold: 0.8
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hexdag.kernel.domain.resource_accounting import (
    ResourceLimits,
    ResourceUsage,
)
from hexdag.kernel.exceptions import HexDAGError
from hexdag.kernel.logging import get_logger
from hexdag.kernel.orchestration.events.events import (
    Event,
    PortCallEvent,
    ResourceLimitExceeded,
    ResourceWarning,
)
from hexdag.kernel.ports.llm import LLMPortCall
from hexdag.kernel.tool_router import ToolRouterCall

if TYPE_CHECKING:
    from hexdag.kernel.ports.observer_manager import ObserverManager

logger = get_logger(__name__)


class ResourceLimitExceededError(HexDAGError):
    """Raised when a resource limit is exceeded and enforcement is enabled."""


class ResourceAccountingObserver:
    """Observer that tracks resource usage and enforces limits.

    Parameters
    ----------
    limits : ResourceLimits
        Configured resource limits.
    pipeline_name : str
        Pipeline name for event attribution.
    observer_manager : ObserverManager | None
        If provided, emits ``ResourceWarning`` and ``ResourceLimitExceeded``
        events.  If None, only logs warnings.
    enforce : bool
        If True, raises ``ResourceLimitExceededError`` when a limit is
        exceeded.  If False (default), only emits events and logs.
    """

    def __init__(
        self,
        limits: ResourceLimits,
        pipeline_name: str = "",
        observer_manager: ObserverManager | None = None,
        enforce: bool = False,
    ) -> None:
        """Initialize the observer with resource limits and optional enforcement."""
        self._limits = limits
        self._pipeline_name = pipeline_name
        self._observer_manager = observer_manager
        self._enforce = enforce
        self._usage = ResourceUsage()
        # Track which resources have already emitted a warning (avoid spam)
        self._warned: set[str] = set()

    @property
    def usage(self) -> ResourceUsage:
        """Current cumulative resource usage."""
        return self._usage

    @property
    def limits(self) -> ResourceLimits:
        """Configured resource limits."""
        return self._limits

    async def handle(self, event: Event) -> None:
        """Handle a port call event by updating usage and checking limits."""
        if not isinstance(event, PortCallEvent):
            return

        self._update_usage(event)
        await self._check_limits()

    def _update_usage(self, event: PortCallEvent) -> None:
        """Accumulate usage from a port call event."""
        if isinstance(event, LLMPortCall):
            usage = event.usage or {}
            self._usage.add_llm_call(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                duration_ms=event.duration_ms,
            )
        elif isinstance(event, ToolRouterCall):
            self._usage.add_tool_call(duration_ms=event.duration_ms)
        else:
            # Generic port call — just track duration
            self._usage.total_duration_ms += event.duration_ms

    async def _check_limits(self) -> None:
        """Check current usage against limits, emit events as needed."""
        checks = self._limits.check(self._usage)

        for check in checks:
            if check.status == "exceeded":
                await self._on_exceeded(check.resource, check.current, check.limit)
            elif check.status == "warning" and check.resource not in self._warned:
                self._warned.add(check.resource)
                await self._on_warning(check.resource, check.current, check.limit, check.ratio)

    async def _on_warning(self, resource: str, current: float, limit: float, ratio: float) -> None:
        """Handle a resource warning threshold being crossed."""
        event = ResourceWarning(
            pipeline_name=self._pipeline_name,
            resource=resource,
            current=current,
            limit=limit,
            ratio=ratio,
        )
        logger.warning("{}", event.log_message())
        if self._observer_manager:
            await self._observer_manager.notify(event)

    async def _on_exceeded(self, resource: str, current: float, limit: float) -> None:
        """Handle a resource limit being exceeded."""
        event = ResourceLimitExceeded(
            pipeline_name=self._pipeline_name,
            resource=resource,
            current=current,
            limit=limit,
        )
        logger.error("{}", event.log_message())
        if self._observer_manager:
            await self._observer_manager.notify(event)
        if self._enforce:
            raise ResourceLimitExceededError(
                f"{resource} exceeded: {current}/{limit} in pipeline '{self._pipeline_name}'"
            )

    def reset(self) -> None:
        """Reset usage counters and warning state."""
        self._usage = ResourceUsage()
        self._warned.clear()

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of current usage and limit status.

        Returns
        -------
        dict[str, Any]
            Keys: ``usage``, ``limits``, ``checks``.
        """
        checks = self._limits.check(self._usage)
        return {
            "usage": {
                "total_tokens": self._usage.total_tokens,
                "input_tokens": self._usage.input_tokens,
                "output_tokens": self._usage.output_tokens,
                "llm_calls": self._usage.llm_calls,
                "tool_calls": self._usage.tool_calls,
                "total_duration_ms": self._usage.total_duration_ms,
            },
            "limits": {
                "max_total_tokens": self._limits.max_total_tokens,
                "max_llm_calls": self._limits.max_llm_calls,
                "max_tool_calls": self._limits.max_tool_calls,
                "max_duration_ms": self._limits.max_duration_ms,
            },
            "checks": [
                {
                    "resource": c.resource,
                    "current": c.current,
                    "limit": c.limit,
                    "ratio": round(c.ratio, 3),
                    "status": c.status,
                }
                for c in checks
            ],
        }
