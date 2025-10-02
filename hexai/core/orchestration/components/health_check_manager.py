"""Health check manager for pre-DAG adapter validation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hexai.core.ports.observer_manager import ObserverManagerPort
else:
    ObserverManagerPort = Any

from hexai.core.application.events import HealthCheckCompleted
from hexai.core.logging import get_logger
from hexai.core.ports.healthcheck import HealthStatus
from hexai.core.protocols import HealthCheckable

logger = get_logger(__name__)

# Constants
MANAGER_PORT_NAMES = frozenset({"observer_manager", "policy_manager"})
LATENCY_PRECISION = 1  # Decimal places for latency display


class HealthCheckManager:
    """Manages health checks on adapters before DAG execution.

    Responsibilities:
    - Check adapter health via ahealth_check() method
    - Emit HealthCheckCompleted events
    - Determine if unhealthy adapters should block execution

    Examples
    --------
    >>> manager = HealthCheckManager(fail_fast=True, warn_only=False)
    >>> health_results = await manager.check_all_adapters(
    ...     ports={"llm": openai, "database": postgres},
    ...     observer_manager=observer,
    ...     pipeline_name="my_pipeline"
    ... )
    """

    def __init__(self, fail_fast: bool = False, warn_only: bool = True):
        """Initialize health check manager.

        Parameters
        ----------
        fail_fast : bool, default=False
            If True, unhealthy adapters block pipeline execution
        warn_only : bool, default=True
            If True, log warnings for unhealthy adapters but don't block
        """
        self.fail_fast = fail_fast
        self.warn_only = warn_only

    async def check_all_adapters(
        self,
        ports: dict[str, Any],
        observer_manager: ObserverManagerPort | None,
        pipeline_name: str,
    ) -> list[HealthStatus]:
        """Run health checks on all adapters that implement ahealth_check().

        Parameters
        ----------
        ports : dict[str, Any]
            All available ports
        observer_manager : ObserverManagerPort | None
            Optional observer for event emission
        pipeline_name : str
            Name of the pipeline

        Returns
        -------
        list[HealthStatus]
            Health status results from all adapters
        """
        health_results = []

        for port_name, adapter in ports.items():
            # Skip non-adapter ports
            if port_name in MANAGER_PORT_NAMES:
                continue

            # Check if adapter implements health check protocol
            if isinstance(adapter, HealthCheckable):
                status = await self._check_single_adapter(port_name, adapter, observer_manager)
                health_results.append(status)

        return health_results

    async def _check_single_adapter(
        self,
        port_name: str,
        adapter: Any,
        observer_manager: ObserverManagerPort | None,
    ) -> HealthStatus:
        """Check health of a single adapter.

        Parameters
        ----------
        port_name : str
            Name of the port
        adapter : Any
            Adapter instance
        observer_manager : ObserverManagerPort | None
            Optional observer for event emission

        Returns
        -------
        HealthStatus
            Health status of the adapter
        """
        try:
            logger.debug(f"Running health check for {port_name}")
            health_check = adapter.ahealth_check
            status: HealthStatus = await health_check()  # pyright: ignore[reportGeneralTypeIssues]
            status.port_name = port_name  # Ensure port name is set

            # Emit event
            if observer_manager:
                event = HealthCheckCompleted(
                    adapter_name=status.adapter_name,
                    port_name=port_name,
                    status=status,
                )
                await observer_manager.notify(event)

            # Log result
            self._log_health_result(port_name, status)

            return status

        except (RuntimeError, ConnectionError, TimeoutError, ValueError) as e:
            # Health check errors - mark adapter as unhealthy
            logger.error(f"Health check failed for {port_name}: {e}", exc_info=True)
            adapter_name = getattr(adapter, "_hexdag_name", port_name)
            return HealthStatus(
                status="unhealthy",
                adapter_name=adapter_name,
                port_name=port_name,
                error=e,
            )

    def _log_health_result(self, port_name: str, status: HealthStatus) -> None:
        """Log health check result.

        Parameters
        ----------
        port_name : str
            Name of the port
        status : HealthStatus
            Health status result
        """
        if status.status == "healthy":
            latency_info = (
                f" ({status.latency_ms:.{LATENCY_PRECISION}f}ms)" if status.latency_ms else ""
            )
            logger.info(f"✅ {port_name} health check: {status.status}{latency_info}")
        else:
            logger.warning(f"⚠️ {port_name} health check: {status.status} - {status.error}")

    def get_unhealthy_adapters(self, health_results: list[HealthStatus]) -> list[HealthStatus]:
        """Filter health results to only unhealthy adapters.

        Parameters
        ----------
        health_results : list[HealthStatus]
            All health check results

        Returns
        -------
        list[HealthStatus]
            Only the unhealthy adapters
        """
        return [h for h in health_results if h.status == "unhealthy"]
