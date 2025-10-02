"""Pre-DAG and Post-DAG hook management for orchestrator lifecycle.

Hooks provide extensibility points before and after DAG execution for:
- Health checking adapters
- Loading secrets from KeyVault into memory
- Environment validation
- Resource cleanup
- Metrics export
- Notifications
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

    from hexai.core.orchestration.models import NodeExecutionContext
    from hexai.core.ports.observer_manager import ObserverManagerPort

from hexai.core.orchestration.components.adapter_lifecycle_manager import (
    AdapterLifecycleManager,
)
from hexai.core.orchestration.components.health_check_manager import HealthCheckManager
from hexai.core.orchestration.components.secret_manager import SecretManager
from hexai.core.orchestration.hook_context import (
    PipelineStatus,
    PostHookContext,
    PostHookManagerProtocol,
    PreHookContext,
    PreHookManagerProtocol,
)

logger = logging.getLogger(__name__)

# Constants to replace magic values
HEALTH_CHECK_LATENCY_PRECISION = 1  # Decimal places for latency display
DEFAULT_SECRET_PREFIX = "secret:"  # nosec B105 - Not a password, it's a key prefix for memory storage

__all__ = [
    "HookConfig",
    "PostDagHookConfig",
    "PreDagHookManager",
    "PostDagHookManager",
    "PipelineStatus",
    "PreHookContext",
    "PostHookContext",
    "PreHookManagerProtocol",
    "PostHookManagerProtocol",
]


@dataclass
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
        Load secrets from SecretPort into Memory before execution
    secret_keys : list[str] | None
        Specific secret keys to load. If None, loads all available secrets.
    secret_prefix : str
        Prefix for secret keys in memory (default: "secret:")
    custom_hooks : list[Callable]
        User-defined pre-DAG hooks. Each receives (ports, context) and returns Any.

    Examples
    --------
    >>> config = HookConfig(
    ...     enable_health_checks=True,
    ...     health_check_fail_fast=True,
    ...     enable_secret_injection=True,
    ...     secret_keys=["OPENAI_API_KEY", "DB_PASSWORD"]
    ... )
    """

    enable_health_checks: bool = True
    health_check_fail_fast: bool = False
    health_check_warn_only: bool = True
    enable_secret_injection: bool = True
    secret_keys: list[str] | None = None
    secret_prefix: str = DEFAULT_SECRET_PREFIX
    custom_hooks: list[Callable] = field(default_factory=list)


@dataclass
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
    enable_metrics_export : bool
        Export pipeline metrics to configured backends
    metrics_backends : list[str]
        List of metric backend names (e.g., ["prometheus", "datadog"])
    enable_notifications : bool
        Send notifications about pipeline completion
    notification_channels : list[str]
        List of notification channels (e.g., ["slack", "email"])
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
    enable_metrics_export: bool = False
    metrics_backends: list[str] = field(default_factory=list)
    enable_notifications: bool = False
    notification_channels: list[str] = field(default_factory=list)
    custom_hooks: list[Callable] = field(default_factory=list)
    run_on_success: bool = True
    run_on_failure: bool = True
    run_on_cancellation: bool = True


class PreDagHookManager:
    """Manages pre-DAG hook execution before pipeline starts.

    Pre-DAG hooks execute BEFORE the PipelineStarted event and include:
    1. Health checks on all adapters
    2. Secret injection from KeyVault/SecretPort into Memory
    3. Custom user-defined setup hooks

    Examples
    --------
    >>> config = HookConfig(enable_health_checks=True)
    >>> manager = PreDagHookManager(config)
    >>> results = await manager.execute_hooks(
    ...     ports={"llm": openai, "database": postgres},
    ...     context=context,
    ...     observer_manager=observer,
    ...     policy_manager=policy,
    ...     pipeline_name="my_pipeline"
    ... )
    """

    def __init__(self, config: HookConfig | None = None):
        """Initialize pre-DAG hook manager.

        Parameters
        ----------
        config : HookConfig | None
            Hook configuration. If None, uses default configuration.
        """
        self.config = config or HookConfig()

        # Initialize focused sub-managers
        self._health_check_manager = HealthCheckManager(
            fail_fast=self.config.health_check_fail_fast,
            warn_only=self.config.health_check_warn_only,
        )
        self._secret_manager = SecretManager(
            secret_keys=self.config.secret_keys,
            secret_prefix=self.config.secret_prefix,
        )

    def get_secret_manager(self) -> SecretManager:
        """Get the secret manager for post-hook cleanup access.

        Returns
        -------
        SecretManager
            The secret manager instance used for secret lifecycle management
        """
        return self._secret_manager

    async def execute_hooks(
        self,
        context: NodeExecutionContext,
        pipeline_name: str,
    ) -> dict[str, Any]:
        """Execute all pre-DAG hooks in order."""
        from hexai.core.context import (
            get_observer_manager,
            get_port,
            get_ports,
        )
        from hexai.core.orchestration.components import OrchestratorError

        results: dict[str, Any] = {}
        ports = get_ports() or {}
        observer_manager = get_observer_manager()

        # 1. Health checks
        if self.config.enable_health_checks:
            logger.info(f"Running health checks for pipeline '{pipeline_name}'")
            health_results = await self._health_check_manager.check_all_adapters(
                ports=ports, observer_manager=observer_manager, pipeline_name=pipeline_name
            )
            results["health_checks"] = health_results

            # Check for critical failures
            unhealthy = self._health_check_manager.get_unhealthy_adapters(health_results)
            if unhealthy:
                unhealthy_names = [h.adapter_name for h in unhealthy]
                error_msg = f"Unhealthy adapters: {unhealthy_names}"

                if self.config.health_check_fail_fast:
                    logger.error(f"Health check failed - blocking pipeline: {error_msg}")
                    raise OrchestratorError(f"Health check failed: {error_msg}")
                if self.config.health_check_warn_only:
                    logger.warning(f"Health check issues detected: {error_msg}")
                else:
                    logger.info(f"Health check issues: {error_msg}")

        # 2. Secret injection
        if self.config.enable_secret_injection:
            logger.info(f"Loading secrets for pipeline '{pipeline_name}'")
            secret_port = get_port("secret")
            memory = get_port("memory")
            secret_results = await self._secret_manager.load_secrets(
                secret_port=secret_port, memory=memory, dag_id=context.dag_id
            )
            results["secrets_loaded"] = secret_results

        # 3. Custom hooks
        for hook in self.config.custom_hooks:
            hook_name = hook.__name__
            logger.info(f"Running custom pre-DAG hook: {hook_name}")
            try:
                hook_result = await hook(ports, context)
                results[hook_name] = hook_result
            except Exception as e:
                logger.error(f"Custom hook '{hook_name}' failed: {e}", exc_info=True)
                results[hook_name] = {"error": str(e)}
                raise

        return results


class PostDagHookManager:
    """Manages post-DAG hook execution after pipeline completes.

    Post-DAG hooks execute AFTER the pipeline completes (success/failure/cancellation)
    and include:
    1. Checkpoint saving
    2. Metrics export
    3. Notifications
    4. Custom cleanup hooks
    5. Secret cleanup (security)
    6. Adapter cleanup (close connections)

    These hooks run in a finally block to ensure cleanup happens even on failure.

    Examples
    --------
    >>> config = PostDagHookConfig(enable_secret_cleanup=True)
    >>> manager = PostDagHookManager(config)
    >>> results = await manager.execute_hooks(
    ...     ports=ports,
    ...     context=context,
    ...     observer_manager=observer,
    ...     policy_manager=policy,
    ...     pipeline_name="my_pipeline",
    ...     pipeline_status="success",
    ...     node_results=results,
    ...     duration_ms=1500.0
    ... )
    """

    def __init__(
        self,
        config: PostDagHookConfig | None = None,
        pre_hook_manager: PreDagHookManager | None = None,
    ):
        """Initialize post-DAG hook manager.

        Parameters
        ----------
        config : PostDagHookConfig | None
            Hook configuration. If None, uses default configuration.
        pre_hook_manager : PreDagHookManager | None
            Reference to pre-hook manager for accessing secret manager
        """
        self.config = config or PostDagHookConfig()
        self._pre_hook_manager = pre_hook_manager

        # Initialize focused sub-managers
        self._adapter_lifecycle_manager = AdapterLifecycleManager()

    async def execute_hooks(
        self,
        context: NodeExecutionContext,
        pipeline_name: str,
        pipeline_status: Literal["success", "failed", "cancelled"],
        node_results: dict[str, Any],
        error: Exception | None = None,
    ) -> dict[str, Any]:
        """Execute all post-DAG hooks."""
        from hexai.core.context import (
            get_observer_manager,
            get_port,
            get_ports,
        )

        results: dict[str, Any] = {}
        ports = get_ports() or {}
        observer_manager = get_observer_manager()

        # Check if hooks should run based on pipeline status
        should_run = (
            (pipeline_status == "success" and self.config.run_on_success)
            or (pipeline_status == "failed" and self.config.run_on_failure)
            or (pipeline_status == "cancelled" and self.config.run_on_cancellation)
        )

        if not should_run:
            logger.debug(f"Skipping post-DAG hooks for status: {pipeline_status}")
            return {"skipped": True, "reason": f"Not configured for {pipeline_status}"}

        logger.info(f"Running post-DAG hooks for pipeline '{pipeline_name}' ({pipeline_status})")

        # 1. Save checkpoint (if enabled)
        if self.config.enable_checkpoint_save and (
            pipeline_status == "success" or self.config.checkpoint_on_failure
        ):
            try:
                checkpoint_result = await self._save_checkpoint(
                    ports, context, node_results, pipeline_status, observer_manager
                )
                results["checkpoint"] = checkpoint_result
            except Exception as e:
                logger.error(f"Checkpoint save failed: {e}", exc_info=True)
                results["checkpoint"] = {"error": str(e)}

        # 2. Custom hooks (user-defined)
        for hook in self.config.custom_hooks:
            hook_name = hook.__name__
            try:
                logger.debug(f"Running custom post-DAG hook: {hook_name}")
                hook_result = await hook(ports, context, node_results, pipeline_status, error)
                results[hook_name] = hook_result
            except Exception as e:
                logger.error(f"Custom hook '{hook_name}' failed: {e}", exc_info=True)
                results[hook_name] = {"error": str(e)}

        # 3. Secret cleanup (security - do this before adapter cleanup)
        if self.config.enable_secret_cleanup and self._pre_hook_manager:
            try:
                secret_manager = self._pre_hook_manager.get_secret_manager()
                memory = get_port("memory")
                secret_cleanup = await secret_manager.cleanup_secrets(
                    memory=memory, dag_id=context.dag_id
                )
                results["secret_cleanup"] = secret_cleanup
            except Exception as e:
                logger.error(f"Secret cleanup failed: {e}", exc_info=True)
                results["secret_cleanup"] = {"error": str(e)}

        # 4. Adapter cleanup (close connections - do this last)
        if self.config.enable_adapter_cleanup:
            try:
                adapter_cleanup = await self._adapter_lifecycle_manager.cleanup_all_adapters(
                    ports=ports, observer_manager=observer_manager
                )
                results["adapter_cleanup"] = adapter_cleanup
            except Exception as e:
                logger.error(f"Adapter cleanup failed: {e}", exc_info=True)
                results["adapter_cleanup"] = {"error": str(e)}

        return results

    async def _save_checkpoint(
        self,
        ports: dict[str, Any],
        context: NodeExecutionContext,
        node_results: dict[str, Any],
        status: str,
        observer_manager: ObserverManagerPort | None,
    ) -> dict[str, Any]:
        """Save final checkpoint state."""
        from hexai.core.context import get_port
        from hexai.core.orchestration.components import CheckpointManager
        from hexai.core.orchestration.models import CheckpointState

        memory = get_port("memory")
        if not memory:
            logger.debug("No memory port available for checkpoint save")
            return {"skipped": "No memory port available"}

        checkpoint_mgr = CheckpointManager(storage=memory)

        # Create final checkpoint
        from datetime import UTC, datetime

        state = CheckpointState(
            run_id=context.dag_id,
            dag_id=context.dag_id,
            graph_snapshot={},  # Graph snapshot not available here
            initial_input=None,  # Initial input not available here
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
