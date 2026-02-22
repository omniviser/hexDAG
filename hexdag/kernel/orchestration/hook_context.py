"""Context objects, protocols, and constants for hook execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from hexdag.kernel.orchestration.models import NodeExecutionContext
    from hexdag.kernel.ports.observer_manager import ObserverManagerPort


# Constants for pipeline status (replacing string literals)
class PipelineStatus(StrEnum):
    """Pipeline execution status enumeration.

    This replaces string literals for pipeline status with a proper type.

    Attributes
    ----------
    SUCCESS : str
        Pipeline completed successfully
    FAILED : str
        Pipeline failed with an error
    CANCELLED : str
        Pipeline was cancelled (e.g., timeout)
    """

    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class PreHookContext:
    """Context for pre-DAG hook execution.

    Bundles all parameters needed for pre-DAG hooks into a single object,
    reducing parameter lists and improving maintainability.

    Attributes
    ----------
    ports : dict[str, Any]
        All available ports for the pipeline
    context : NodeExecutionContext
        Execution context for this pipeline run
    observer_manager : ObserverManagerPort | None
        Optional observer for event emission
    pipeline_name : str
        Name of the pipeline being executed

    Examples
    --------
    Example usage::

        ctx = PreHookContext(
            ports={"llm": openai, "database": postgres},
            context=execution_context,
            observer_manager=observer,
            pipeline_name="my_pipeline"
        )
        results = await manager.execute_hooks(ctx)
    """

    ports: dict[str, Any]
    context: NodeExecutionContext
    observer_manager: ObserverManagerPort | None
    pipeline_name: str


@dataclass(slots=True)
class PostHookContext:
    """Context for post-DAG hook execution.

    Bundles all parameters needed for post-DAG hooks into a single object,
    reducing parameter lists and improving maintainability.

    Attributes
    ----------
    ports : dict[str, Any]
        All available ports
    context : NodeExecutionContext
        Execution context
    observer_manager : ObserverManagerPort | None
        Optional observer manager
    pipeline_name : str
        Pipeline name
    pipeline_status : PipelineStatus
        Final pipeline status (enum)
    node_results : dict[str, Any]
        Results from all executed nodes
    error : Exception | None
        Exception if pipeline failed

    Examples
    --------
    Example usage::

        ctx = PostHookContext(
            ports=ports,
            context=execution_context,
            observer_manager=observer,
            pipeline_name="my_pipeline",
            pipeline_status=PipelineStatus.SUCCESS,
            node_results={"node1": "result1"},
            error=None
        )
        results = await manager.execute_hooks(ctx)
    """

    ports: dict[str, Any]
    context: NodeExecutionContext
    observer_manager: ObserverManagerPort | None
    pipeline_name: str
    pipeline_status: PipelineStatus
    node_results: dict[str, Any]
    error: Exception | None = None

    @property
    def status_str(self) -> str:
        """Get status as string for backward compatibility.

        Returns
        -------
        str
            Status value ("success", "failed", or "cancelled")
        """
        return self.pipeline_status.value


class PreHookManagerProtocol(Protocol):
    """Protocol for pre-DAG hook managers.

    This protocol defines the interface that all pre-DAG hook managers must implement,
    enabling better testing, dependency injection, and extensibility.

    Examples
    --------
    >>> class CustomPreHookManager:
    ...     async def execute_hooks(
    ...         self,
    ...         ports: dict[str, Any],
    ...         context: NodeExecutionContext,
    ...         observer_manager: ObserverManagerPort | None,
    ...         pipeline_name: str,
    ...     ) -> dict[str, Any]:
    ...         # Custom implementation
    ...         return {}
    """

    async def execute_hooks(
        self,
        ports: dict[str, Any],
        context: NodeExecutionContext,
        observer_manager: ObserverManagerPort | None,
        pipeline_name: str,
    ) -> dict[str, Any]:
        """Execute all pre-DAG hooks.

        Parameters
        ----------
        ports : dict[str, Any]
            All available ports for the pipeline
        context : NodeExecutionContext
            Execution context for this pipeline run
        observer_manager : ObserverManagerPort | None
            Optional observer for event emission
        pipeline_name : str
            Name of the pipeline being executed

        Returns
        -------
        dict[str, Any]
            Results from all hook executions
        """
        ...


class PostHookManagerProtocol(Protocol):
    """Protocol for post-DAG hook managers.

    This protocol defines the interface that all post-DAG hook managers must implement,
    enabling better testing, dependency injection, and extensibility.

    Examples
    --------
    >>> class CustomPostHookManager:
    ...     async def execute_hooks(
    ...         self,
    ...         ports: dict[str, Any],
    ...         context: NodeExecutionContext,
    ...         observer_manager: ObserverManagerPort | None,
    ...         pipeline_name: str,
    ...         pipeline_status: str,
    ...         node_results: dict[str, Any],
    ...         error: Exception | None = None,
    ...     ) -> dict[str, Any]:
    ...         # Custom implementation
    ...         return {}
    """

    async def execute_hooks(
        self,
        ports: dict[str, Any],
        context: NodeExecutionContext,
        observer_manager: ObserverManagerPort | None,
        pipeline_name: str,
        pipeline_status: str,
        node_results: dict[str, Any],
        error: Exception | None = None,
    ) -> dict[str, Any]:
        """Execute all post-DAG hooks.

        Parameters
        ----------
        ports : dict[str, Any]
            All available ports
        context : NodeExecutionContext
            Execution context
        observer_manager : ObserverManagerPort | None
            Optional observer manager
        pipeline_name : str
            Pipeline name
        pipeline_status : str
            Final pipeline status
        node_results : dict[str, Any]
            Results from all executed nodes
        error : Exception | None
            Exception if pipeline failed

        Returns
        -------
        dict[str, Any]
            Results from all hook executions
        """
        ...
