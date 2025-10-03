"""Health check models and types for adapter monitoring.

This module provides shared types for adapter health checking.
Each port can optionally implement an ahealth_check() method that returns HealthStatus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["HealthStatus"]


@dataclass
class HealthStatus:
    """Health check result returned by adapters.

    Adapters can implement an optional async ahealth_check() method
    that returns this status object to indicate their operational state.

    Attributes
    ----------
    status : Literal["healthy", "degraded", "unhealthy"]
        Current health status:
        - "healthy": Adapter is fully operational
        - "degraded": Adapter is operational but with reduced performance
        - "unhealthy": Adapter is not operational
    adapter_name : str
        Name of the adapter being checked
    port_name : str | None
        Name of the port this adapter implements (e.g., "llm", "database")
    details : dict[str, Any]
        Additional context about the health check (e.g., error messages, metrics)
    latency_ms : float | None
        Time taken to perform the health check in milliseconds
    error : Exception | None
        Exception if the health check failed

    Examples
    --------
    Example usage::

        # Healthy adapter
        HealthStatus(
        status="healthy",
        adapter_name="openai",
        port_name="llm",
        latency_ms=45.2
        )

        # Unhealthy adapter with error
        HealthStatus(
        status="unhealthy",
        adapter_name="postgres",
        port_name="database",
        error=ConnectionError("Could not connect to database"),
        details={"host": "localhost", "port": 5432}
        )
    """

    status: Literal["healthy", "degraded", "unhealthy"]
    adapter_name: str
    port_name: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None
    error: Exception | None = None

    def is_healthy(self) -> bool:
        """Check if status is healthy.

        Returns
        -------
        bool
            True if status is "healthy", False otherwise
        """
        return self.status == "healthy"

    def is_operational(self) -> bool:
        """Check if adapter is operational (healthy or degraded).

        Returns
        -------
        bool
            True if status is "healthy" or "degraded", False if "unhealthy"
        """
        return self.status in ("healthy", "degraded")
