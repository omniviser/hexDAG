"""Adapter lifecycle manager for cleanup of adapter resources."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hexdag.core.logging import get_logger

if TYPE_CHECKING:
    from hexdag.core.ports.observer_manager import ObserverManagerPort
else:
    ObserverManagerPort = Any

logger = get_logger(__name__)

# Port names to skip during cleanup
MANAGER_PORT_NAMES = frozenset({"observer_manager"})


class AdapterLifecycleManager:
    """Manages adapter lifecycle including connection cleanup.

    Responsibilities:
    - Close adapter connections (aclose, ashutdown, cleanup methods)
    - Release resources after DAG execution
    - Track which adapters were cleaned up

    Examples
    --------
    Example usage::

        manager = AdapterLifecycleManager()
        result = await manager.cleanup_all_adapters(
            ports={"llm": openai, "database": postgres},
            observer_manager=observer
        )
        # {"cleaned_adapters": ["llm", "database"], "count": 2}
    """

    # Methods to try for cleanup, in order of preference
    CLEANUP_METHODS = ["aclose", "ashutdown", "cleanup"]

    async def cleanup_all_adapters(
        self,
        ports: dict[str, Any],
        observer_manager: ObserverManagerPort | None,
    ) -> dict[str, Any]:
        """Close adapter connections and release resources.

        Parameters
        ----------
        ports : dict[str, Any]
            All available ports
        observer_manager : ObserverManagerPort | None
            Optional observer for event emission

        Returns
        -------
        dict[str, Any]
            Cleanup results with cleaned_adapters list and count

        Examples
        --------
        Example usage::

            result = await manager.cleanup_all_adapters(
                ports={"llm": openai, "database": postgres},
                observer_manager=observer
            )
            # {"cleaned_adapters": ["llm", "database"], "count": 2}
        """
        cleaned_adapters = []

        for port_name, adapter in ports.items():
            # Skip manager ports
            if port_name in MANAGER_PORT_NAMES:
                continue

            # Try each cleanup method in order
            if await self._cleanup_single_adapter(port_name, adapter):
                cleaned_adapters.append(port_name)

        return {"cleaned_adapters": cleaned_adapters, "count": len(cleaned_adapters)}

    async def _cleanup_single_adapter(self, port_name: str, adapter: Any) -> bool:
        """Attempt to clean up a single adapter.

        Parameters
        ----------
        port_name : str
            Name of the port
        adapter : Any
            Adapter instance

        Returns
        -------
        bool
            True if cleanup succeeded, False otherwise
        """
        for method_name in self.CLEANUP_METHODS:
            if hasattr(adapter, method_name) and callable(getattr(adapter, method_name)):
                cleanup_method = getattr(adapter, method_name)
                try:
                    logger.debug(f"Cleaning up adapter '{port_name}' via {method_name}()")
                    await cleanup_method()
                    logger.info(f"✅ Cleaned up adapter: {port_name}")
                    return True  # Only call first matching cleanup method
                except (RuntimeError, ValueError, TypeError, ConnectionError, OSError) as e:
                    # Expected cleanup errors - log but don't crash
                    logger.warning(f"Cleanup failed for {port_name}: {e}")
                    return False

        return False
