"""Models for orchestration state and checkpoints.

This module contains models for representing orchestrator execution
state, execution context, and human-in-the-loop approval requests.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


@dataclass(frozen=True)
class OrchestratorConfig:
    """Configuration for orchestrator behavior.

    This immutable configuration object centralizes all orchestrator settings,
    making it easier to pass configurations around and test different settings.

    Attributes
    ----------
    max_concurrent_nodes : int, default=10
        Maximum number of nodes to execute concurrently in a wave.
        Controls parallelism and resource usage.
    strict_validation : bool, default=False
        If True, raise errors on validation failures.
        If False, log warnings and continue execution.
    default_node_timeout : float | None, default=None
        Default timeout in seconds for node execution.
        None means no timeout. Can be overridden per-node.

    Examples
    --------
    >>> config = OrchestratorConfig(
    ...     max_concurrent_nodes=5,
    ...     strict_validation=True,
    ...     default_node_timeout=30.0
    ... )
    >>> orchestrator = Orchestrator(config=config)

    >>> # Or use defaults
    >>> config = OrchestratorConfig()
    >>> config.max_concurrent_nodes
    10
    """

    max_concurrent_nodes: int = 10
    strict_validation: bool = False
    default_node_timeout: float | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.max_concurrent_nodes <= 0:
            raise ValueError("max_concurrent_nodes must be positive")

        if self.default_node_timeout is not None and self.default_node_timeout <= 0:
            raise ValueError("default_node_timeout must be positive or None")


class CheckpointState(BaseModel):
    """Complete state for checkpoint/resume.

    Saves everything needed to resume a DAG execution from where it left off,
    including the full graph structure (handles both static and dynamic DAGs).

    Attributes
    ----------
    run_id : str
        Unique identifier for this execution run
    dag_id : str
        Stable identifier for the DAG (e.g., YAML file path, function name)
    graph_snapshot : dict[str, Any]
        Serialized DirectedGraph structure
        Format: {"nodes": {...}, "edges": [...]}
    initial_input : Any
        Initial input data passed to the DAG
    node_results : dict[str, Any]
        Results from completed nodes (node_id -> output)
    completed_node_ids : list[str]
        Ordered list of completed node IDs (preserves execution order)
    failed_node_ids : list[str]
        List of node IDs that failed (for retry/debugging)
    created_at : datetime
        When execution started
    updated_at : datetime
        Last checkpoint save time
    metadata : dict[str, Any]
        Optional metadata (custom fields, tags, etc.)

    Notes
    -----
    To resume execution:
    1. Load CheckpointState by run_id
    2. Deserialize graph_snapshot to DirectedGraph
    3. Filter out completed nodes using completed_node_ids
    4. Resume execution with filtered graph and node_results
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    dag_id: str
    graph_snapshot: dict[str, Any]  # Always save the graph
    initial_input: Any
    node_results: dict[str, Any]  # node_id -> output
    completed_node_ids: list[str]  # Ordered
    failed_node_ids: list[str] = []
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = {}


@dataclass
class NodeExecutionContext:
    """Lightweight context tracking current execution position.

    This flows through the execution pipeline and is NOT persisted in checkpoints.
    It's only for tracking where we are during live execution (which node, wave, attempt).

    Attributes
    ----------
    dag_id : str
        Identifier for the DAG being executed
    node_id : str | None
        Current node being executed (None for DAG-level operations)
    wave_index : int
        Index of the current execution wave (for parallel execution tracking)
    attempt : int
        Attempt number (for retry scenarios)
    metadata : dict[str, Any]
        Additional metadata that can be attached to the context
    """

    dag_id: str
    node_id: str | None = None
    wave_index: int = 0
    attempt: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_node(self, node_id: str, wave_index: int) -> "NodeExecutionContext":
        """Create new context for a specific node execution.

        Args
        ----
            node_id: The ID of the node being executed
            wave_index: The wave index for parallel execution tracking

        Returns
        -------
        NodeExecutionContext
            New context with updated node and wave information
        """
        return NodeExecutionContext(
            dag_id=self.dag_id,
            node_id=node_id,
            wave_index=wave_index,
            attempt=self.attempt,
            metadata=self.metadata.copy(),
        )

    def with_attempt(self, attempt: int) -> "NodeExecutionContext":
        """Create new context with updated attempt number.

        Args
        ----
            attempt: The attempt number (for retry scenarios)

        Returns
        -------
        NodeExecutionContext
            New context with updated attempt number
        """
        return NodeExecutionContext(
            dag_id=self.dag_id,
            node_id=self.node_id,
            wave_index=self.wave_index,
            attempt=attempt,
            metadata=self.metadata.copy(),
        )
