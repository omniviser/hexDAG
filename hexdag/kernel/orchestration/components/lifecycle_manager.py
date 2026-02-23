"""Unified lifecycle manager for pipeline setup and cleanup.

This module consolidates pre-DAG and post-DAG hook management into a single
component that handles the complete pipeline lifecycle:

- Health checks on adapters
- Secret injection from KeyVault/SecretStore
- Custom user hooks
- Checkpoint saving
- Secret cleanup (security)
- Adapter cleanup (connections)

The LifecycleManager replaces the separate PreDagHookManager, PostDagHookManager,
HealthCheckManager, SecretManager, and AdapterLifecycleManager components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import MappingProxyType

    from hexdag.kernel.orchestration.models import NodeExecutionContext
    from hexdag.kernel.ports.memory import Memory
    from hexdag.kernel.ports.observer_manager import ObserverManager
    from hexdag.kernel.ports.secret import SecretStore

from hexdag.kernel.context import get_observer_manager, get_port, get_ports
from hexdag.kernel.exceptions import OrchestratorError
from hexdag.kernel.logging import get_logger
from hexdag.kernel.orchestration.components.checkpoint_manager import CheckpointManager
from hexdag.kernel.orchestration.events import HealthCheckCompleted
from hexdag.kernel.orchestration.models import CheckpointState
from hexdag.kernel.ports.healthcheck import HealthStatus
from hexdag.kernel.protocols import HealthCheckable

logger = get_logger(__name__)

# Constants
DEFAULT_SECRET_PREFIX = "secret:"  # nosec B105 - Not a password, it's a key prefix
MANAGER_PORT_NAMES = frozenset({"observer_manager"})
LATENCY_PRECISION = 1
CLEANUP_METHODS = ["aclose", "ashutdown", "cleanup"]

__all__ = [
    "HookConfig",
    "LifecycleManager",
    "PipelineStatus",
    "PostDagHookConfig",
]


class PipelineStatus(StrEnum):
    """Pipeline execution status enumeration."""

    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class HookConfig:
    """Configuration for pre-DAG hooks.

    Attributes
    ----------
    enable_health_checks : bool
        Run health checks on all adapters before pipeline execution
    health_check_fail_fast : bool
        If True, unhealthy adapters block pipeline execution
    health_check_warn_only : bool
        If True, log warnings for unhealthy adapters but don't block
    enable_secret_injection : bool
        Load secrets from SecretStore into Memory before execution
    secret_keys : list[str] | None
        Specific secret keys to load. If None, loads all available secrets.
    secret_prefix : str
        Prefix for secret keys in memory (default: "secret:")
    custom_hooks : list[Callable]
        User-defined pre-DAG hooks. Each receives (ports, context) and returns Any.
    """

    enable_health_checks: bool = True
    health_check_fail_fast: bool = False
    health_check_warn_only: bool = True
    enable_secret_injection: bool = True
    secret_keys: list[str] | None = None
    secret_prefix: str = DEFAULT_SECRET_PREFIX
    custom_hooks: list[Callable] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PostDagHookConfig:
    """Configuration for post-DAG hooks.

    Attributes
    ----------
    enable_adapter_cleanup : bool
        Call adapter.aclose() or adapter.ashutdown() if available
    enable_secret_cleanup : bool
        Remove secrets from Memory after pipeline execution
    enable_checkpoint_save : bool
        Save final checkpoint state
    checkpoint_on_failure : bool
        Save checkpoint even if pipeline fails (useful for debugging)
    custom_hooks : list[Callable]
        User-defined post-DAG hooks
    run_on_success : bool
        Run hooks when pipeline succeeds
    run_on_failure : bool
        Run hooks when pipeline fails
    run_on_cancellation : bool
        Run hooks when pipeline is cancelled
    """

    enable_adapter_cleanup: bool = True
    enable_secret_cleanup: bool = True
    enable_checkpoint_save: bool = False
    checkpoint_on_failure: bool = True
    custom_hooks: list[Callable] = field(default_factory=list)
    run_on_success: bool = True
    run_on_failure: bool = True
    run_on_cancellation: bool = True


class LifecycleManager:
    """Unified manager for pipeline lifecycle: setup, health checks, secrets, cleanup.

    This manager consolidates all pre-DAG and post-DAG operations:

    Pre-execution (pre_execute):
    1. Health checks on all adapters
    2. Secret injection from SecretStore into Memory
    3. Custom user-defined setup hooks

    Post-execution (post_execute):
    1. Checkpoint saving (if enabled)
    2. Custom cleanup hooks
    3. Secret cleanup (CRITICAL - always runs in finally)
    4. Adapter cleanup (CRITICAL - always runs in finally)

    Examples
    --------
    Basic usage::

        manager = LifecycleManager(
            pre_config=HookConfig(enable_health_checks=True),
            post_config=PostDagHookConfig(enable_secret_cleanup=True)
        )

        # Before pipeline execution
        pre_results = await manager.pre_execute(context, "my_pipeline")

        # ... pipeline runs ...

        # After pipeline execution (always call, even on failure)
        post_results = await manager.post_execute(
            context, "my_pipeline", "success", node_results
        )
    """

    def __init__(
        self,
        pre_config: HookConfig | None = None,
        post_config: PostDagHookConfig | None = None,
    ):
        """Initialize lifecycle manager.

        Parameters
        ----------
        pre_config : HookConfig | None
            Configuration for pre-DAG hooks. Uses defaults if None.
        post_config : PostDagHookConfig | None
            Configuration for post-DAG hooks. Uses defaults if None.
        """
        self.pre_config = pre_config or HookConfig()
        self.post_config = post_config or PostDagHookConfig()
        self._loaded_secret_keys: dict[str, list[str]] = {}  # dag_id -> memory_keys

    # ========================================================================
    # Pre-execution
    # ========================================================================

    async def pre_execute(
        self,
        context: NodeExecutionContext,
        pipeline_name: str,
    ) -> dict[str, Any]:
        """Execute all pre-DAG lifecycle tasks.

        Parameters
        ----------
        context : NodeExecutionContext
            Execution context for this pipeline run
        pipeline_name : str
            Name of the pipeline being executed

        Returns
        -------
        dict[str, Any]
            Results from all pre-execution tasks
        """
        results: dict[str, Any] = {}
        ports: MappingProxyType[str, Any] | dict[Any, Any] = get_ports() or {}
        observer_manager = get_observer_manager()

        # 1. Health checks
        if self.pre_config.enable_health_checks:
            logger.info(f"Running health checks for pipeline '{pipeline_name}'")
            health_results = await self._check_all_adapters(
                ports=dict(ports),
                observer_manager=observer_manager,
                pipeline_name=pipeline_name,
            )
            results["health_checks"] = health_results

            # Check for critical failures
            if unhealthy := self._get_unhealthy_adapters(health_results):
                unhealthy_names = [h.adapter_name for h in unhealthy]
                error_msg = f"Unhealthy adapters: {unhealthy_names}"

                if self.pre_config.health_check_fail_fast:
                    logger.error(f"Health check failed - blocking pipeline: {error_msg}")
                    raise OrchestratorError(f"Health check failed: {error_msg}")
                if self.pre_config.health_check_warn_only:
                    logger.warning(f"Health check issues detected: {error_msg}")
                else:
                    logger.info(f"Health check issues: {error_msg}")

        # 2. Secret injection
        if self.pre_config.enable_secret_injection:
            logger.info(f"Loading secrets for pipeline '{pipeline_name}'")
            secret_port = get_port("secret")
            memory = get_port("memory")
            secret_results = await self._load_secrets(
                secret_port=secret_port,
                memory=memory,
                dag_id=context.dag_id,
            )
            results["secrets_loaded"] = secret_results

        # 3. Custom hooks
        for hook in self.pre_config.custom_hooks:
            hook_name = hook.__name__
            logger.info(f"Running custom pre-DAG hook: {hook_name}")
            try:
                hook_result = await hook(ports, context)
                results[hook_name] = hook_result
            except (RuntimeError, ValueError, KeyError, TypeError) as e:
                logger.error(f"Custom hook '{hook_name}' failed: {e}", exc_info=True)
                results[hook_name] = {"error": str(e)}
                raise

        return results

    # ========================================================================
    # Post-execution
    # ========================================================================

    async def post_execute(
        self,
        context: NodeExecutionContext,
        pipeline_name: str,
        pipeline_status: Literal["success", "failed", "cancelled"],
        node_results: dict[str, Any],
        error: BaseException | None = None,
    ) -> dict[str, Any]:
        """Execute all post-DAG lifecycle tasks.

        Parameters
        ----------
        context : NodeExecutionContext
            Execution context
        pipeline_name : str
            Name of the pipeline
        pipeline_status : Literal["success", "failed", "cancelled"]
            Final pipeline status
        node_results : dict[str, Any]
            Results from all executed nodes
        error : BaseException | None
            Exception if pipeline failed

        Returns
        -------
        dict[str, Any]
            Results from all post-execution tasks
        """
        results: dict[str, Any] = {}
        ports: MappingProxyType[str, Any] | dict[Any, Any] = get_ports() or {}
        observer_manager = get_observer_manager()

        should_run = (
            (pipeline_status == "success" and self.post_config.run_on_success)
            or (pipeline_status == "failed" and self.post_config.run_on_failure)
            or (pipeline_status == "cancelled" and self.post_config.run_on_cancellation)
        )

        if not should_run:
            logger.debug(f"Skipping post-DAG hooks for status: {pipeline_status}")
            return {"skipped": True, "reason": f"Not configured for {pipeline_status}"}

        logger.info(f"Running post-DAG hooks for pipeline '{pipeline_name}' ({pipeline_status})")

        try:
            # 1. Save checkpoint (if enabled)
            if self.post_config.enable_checkpoint_save and (
                pipeline_status == "success" or self.post_config.checkpoint_on_failure
            ):
                try:
                    checkpoint_result = await self._save_checkpoint(
                        dict(ports), context, node_results, pipeline_status, observer_manager
                    )
                    results["checkpoint"] = checkpoint_result
                except Exception as e:
                    logger.error(f"Checkpoint save failed: {e}", exc_info=True)
                    results["checkpoint"] = {"error": str(e)}

            # 2. Custom hooks (user-defined)
            for hook in self.post_config.custom_hooks:
                hook_name = hook.__name__
                try:
                    logger.debug(f"Running custom post-DAG hook: {hook_name}")
                    hook_result = await hook(ports, context, node_results, pipeline_status, error)
                    results[hook_name] = hook_result
                except Exception as e:
                    logger.error(f"Custom hook '{hook_name}' failed: {e}", exc_info=True)
                    results[hook_name] = {"error": str(e)}

        finally:
            # CRITICAL CLEANUP: Always run these, even if above hooks fail
            # 3. Secret cleanup (security - do this before adapter cleanup)
            if self.post_config.enable_secret_cleanup:
                try:
                    memory = get_port("memory")
                    secret_cleanup = await self._cleanup_secrets(
                        memory=memory, dag_id=context.dag_id
                    )
                    results["secret_cleanup"] = secret_cleanup
                except Exception as e:
                    logger.error(f"Secret cleanup failed: {e}", exc_info=True)
                    results["secret_cleanup"] = {"error": str(e)}

            # 4. Adapter cleanup (close connections - do this last)
            if self.post_config.enable_adapter_cleanup:
                try:
                    adapter_cleanup = await self._cleanup_all_adapters(
                        ports=dict(ports), observer_manager=observer_manager
                    )
                    results["adapter_cleanup"] = adapter_cleanup
                except Exception as e:
                    logger.error(f"Adapter cleanup failed: {e}", exc_info=True)
                    results["adapter_cleanup"] = {"error": str(e)}

        return results

    # ========================================================================
    # Health Checks (inlined from HealthCheckManager)
    # ========================================================================

    async def _check_all_adapters(
        self,
        ports: dict[str, Any],
        observer_manager: ObserverManager | None,
        pipeline_name: str,
    ) -> list[HealthStatus]:
        """Run health checks on all adapters that implement ahealth_check()."""
        health_results = []

        for port_name, adapter in ports.items():
            if port_name in MANAGER_PORT_NAMES:
                continue

            if isinstance(adapter, HealthCheckable):
                status = await self._check_single_adapter(port_name, adapter, observer_manager)
                health_results.append(status)

        return health_results

    async def _check_single_adapter(
        self,
        port_name: str,
        adapter: Any,
        observer_manager: ObserverManager | None,
    ) -> HealthStatus:
        """Check health of a single adapter."""
        try:
            logger.debug(f"Running health check for {port_name}")
            health_check = adapter.ahealth_check
            status: HealthStatus = await health_check()
            status.port_name = port_name

            if observer_manager:
                event = HealthCheckCompleted(
                    adapter_name=status.adapter_name,
                    port_name=port_name,
                    status=status,
                )
                await observer_manager.notify(event)

            self._log_health_result(port_name, status)
            return status

        except (RuntimeError, ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"Health check failed for {port_name}: {e}", exc_info=True)
            adapter_name = getattr(adapter, "_hexdag_name", port_name)
            return HealthStatus(
                status="unhealthy",
                adapter_name=adapter_name,
                port_name=port_name,
                error=e,
            )

    def _log_health_result(self, port_name: str, status: HealthStatus) -> None:
        """Log health check result."""
        if status.status == "healthy":
            latency_info = (
                f" ({status.latency_ms:.{LATENCY_PRECISION}f}ms)" if status.latency_ms else ""
            )
            logger.info(f"✅ {port_name} health check: {status.status}{latency_info}")
        else:
            logger.warning(f"⚠️ {port_name} health check: {status.status} - {status.error}")

    def _get_unhealthy_adapters(self, health_results: list[HealthStatus]) -> list[HealthStatus]:
        """Filter health results to only unhealthy adapters."""
        return [h for h in health_results if h.status == "unhealthy"]

    # ========================================================================
    # Secret Management (inlined from SecretManager)
    # ========================================================================

    async def _load_secrets(
        self,
        secret_port: SecretStore | None,
        memory: Memory | None,
        dag_id: str,
    ) -> dict[str, str]:
        """Load secrets from SecretStore into Memory."""
        if not secret_port:
            logger.debug("No secret port configured, skipping secret injection")
            return {}

        if not memory:
            logger.warning("Secret port configured but no memory port available")
            return {}

        try:
            mapping = await secret_port.aload_secrets_to_memory(
                memory=memory,
                prefix=self.pre_config.secret_prefix,
                keys=self.pre_config.secret_keys,
            )

            memory_keys = list(mapping.values())
            self._loaded_secret_keys[dag_id] = memory_keys

            logger.info(
                f"Loaded {len(mapping)} secrets into memory with prefix "
                f"'{self.pre_config.secret_prefix}'"
            )
            logger.debug(f"Secret keys loaded: {list(mapping.keys())}")

            return mapping

        except (ValueError, KeyError, RuntimeError) as e:
            logger.error(f"Failed to inject secrets: {e}", exc_info=True)
            raise

    async def _cleanup_secrets(
        self,
        memory: Memory | None,
        dag_id: str,
    ) -> dict[str, Any]:
        """Remove secrets from Memory for security."""
        if not memory:
            logger.debug("No memory port available for secret cleanup")
            return {"cleaned": False, "reason": "No memory port"}

        secret_keys = self._loaded_secret_keys.get(dag_id, [])

        if not secret_keys:
            logger.debug("No secrets were loaded for this pipeline")
            return {"cleaned": True, "keys_removed": 0}

        removed_count = 0
        for secret_key in secret_keys:
            try:
                await memory.aset(secret_key, None)
                removed_count += 1
                logger.debug(f"Removed secret from memory: {secret_key}")
            except (RuntimeError, ValueError, KeyError) as e:
                logger.warning(f"Failed to remove secret '{secret_key}': {e}")

        if dag_id in self._loaded_secret_keys:
            del self._loaded_secret_keys[dag_id]

        logger.info(f"Secret cleanup: Removed {removed_count} secret(s) from memory")
        return {"cleaned": True, "keys_removed": removed_count}

    # ========================================================================
    # Adapter Cleanup (inlined from AdapterLifecycleManager)
    # ========================================================================

    async def _cleanup_all_adapters(
        self,
        ports: dict[str, Any],
        observer_manager: ObserverManager | None,
    ) -> dict[str, Any]:
        """Close adapter connections and release resources."""
        cleaned_adapters = []

        for port_name, adapter in ports.items():
            if port_name in MANAGER_PORT_NAMES:
                continue

            if await self._cleanup_single_adapter(port_name, adapter):
                cleaned_adapters.append(port_name)

        return {"cleaned_adapters": cleaned_adapters, "count": len(cleaned_adapters)}

    async def _cleanup_single_adapter(self, port_name: str, adapter: Any) -> bool:
        """Attempt to clean up a single adapter."""
        for method_name in CLEANUP_METHODS:
            if hasattr(adapter, method_name) and callable(getattr(adapter, method_name)):
                cleanup_method = getattr(adapter, method_name)
                try:
                    logger.debug(f"Cleaning up adapter '{port_name}' via {method_name}()")
                    await cleanup_method()
                    logger.info(f"✅ Cleaned up adapter: {port_name}")
                    return True
                except (RuntimeError, ValueError, TypeError, ConnectionError, OSError) as e:
                    logger.warning(f"Cleanup failed for {port_name}: {e}")
                    return False

        return False

    # ========================================================================
    # Checkpoint (inlined from PostDagHookManager)
    # ========================================================================

    async def _save_checkpoint(
        self,
        ports: dict[str, Any],
        context: NodeExecutionContext,
        node_results: dict[str, Any],
        status: str,
        observer_manager: ObserverManager | None,
    ) -> dict[str, Any]:
        """Save final checkpoint state."""
        memory = get_port("memory")
        if not memory:
            logger.debug("No memory port available for checkpoint save")
            return {"skipped": "No memory port available"}

        checkpoint_mgr = CheckpointManager(storage=memory)

        from datetime import UTC, datetime  # lazy: deferred to avoid datetime import in hot path

        state = CheckpointState(
            run_id=context.dag_id,
            dag_id=context.dag_id,
            graph_snapshot={},
            initial_input=None,
            node_results=node_results,
            completed_node_ids=list(node_results.keys()),
            failed_node_ids=[],
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            metadata={"pipeline_status": status},
        )

        await checkpoint_mgr.save(state)
        logger.info(f"Saved checkpoint for run_id: {context.dag_id}")

        return {"saved": True, "run_id": context.dag_id, "node_count": len(node_results)}
