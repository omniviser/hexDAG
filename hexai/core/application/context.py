"""Core models for the application layer.

This module contains data models that are used across the application layer,
particularly for orchestration and execution flow.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionContext:
    """Context that flows through node and event execution.

    Carries metadata through the execution pipeline, providing
    information about the current execution state and allowing
    tracking of execution flow through the DAG.
    """

    dag_id: str
    node_id: str | None = None
    wave_index: int = 0
    attempt: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_node(self, node_id: str, wave_index: int) -> "ExecutionContext":
        """Create new context for a specific node execution.

        Args
        ----
            node_id: The ID of the node being executed
            wave_index: The wave index for parallel execution tracking

        Returns
        -------
        ExecutionContext
            New context with updated node and wave information
        """
        return ExecutionContext(
            dag_id=self.dag_id,
            node_id=node_id,
            wave_index=wave_index,
            attempt=self.attempt,
            metadata=self.metadata.copy(),
        )

    def with_attempt(self, attempt: int) -> "ExecutionContext":
        """Create new context with updated attempt number.

        Args
        ----
            attempt: The attempt number (for retry scenarios)

        Returns
        -------
        ExecutionContext
            New context with updated attempt number
        """
        return ExecutionContext(
            dag_id=self.dag_id,
            node_id=self.node_id,
            wave_index=self.wave_index,
            attempt=attempt,
            metadata=self.metadata.copy(),
        )
