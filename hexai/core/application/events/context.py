"""Execution context that flows through the DAG execution."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class ExecutionContext:
    """Immutable context that flows through node and event execution.

    This context carries metadata through the execution pipeline and can be
    extended by hooks and handlers without modifying the original.
    """

    dag_id: str
    node_id: str | None = None
    wave_index: int = 0
    attempt: int = 1
    metadata: MappingProxyType[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def with_node(self, node_id: str, wave_index: int) -> "ExecutionContext":
        """Create new context for a specific node execution."""
        return ExecutionContext(
            dag_id=self.dag_id,
            node_id=node_id,
            wave_index=wave_index,
            attempt=self.attempt,
            metadata=self.metadata,
        )

    def with_attempt(self, attempt: int) -> "ExecutionContext":
        """Create new context with updated attempt number."""
        return ExecutionContext(
            dag_id=self.dag_id,
            node_id=self.node_id,
            wave_index=self.wave_index,
            attempt=attempt,
            metadata=self.metadata,
        )

    def with_metadata(self, key: str, value: Any) -> "ExecutionContext":
        """Create new context with additional metadata."""
        new_metadata = {**self.metadata, key: value}
        return ExecutionContext(
            dag_id=self.dag_id,
            node_id=self.node_id,
            wave_index=self.wave_index,
            attempt=self.attempt,
            metadata=MappingProxyType(new_metadata),
        )

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key."""
        return self.metadata.get(key, default)
