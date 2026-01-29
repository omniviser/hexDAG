"""Constants for orchestration execution.

This module defines shared constants used between orchestrator and executors
to avoid magic strings and ensure consistency.
"""

# Executor context keys - used to store graph/results in ports context
# These are prefixed with _hexdag_ to avoid collisions with user ports
EXECUTOR_CONTEXT_GRAPH = "_hexdag_graph"
EXECUTOR_CONTEXT_NODE_RESULTS = "_hexdag_node_results"
EXECUTOR_CONTEXT_INITIAL_INPUT = "_hexdag_initial_input"

__all__ = [
    "EXECUTOR_CONTEXT_GRAPH",
    "EXECUTOR_CONTEXT_NODE_RESULTS",
    "EXECUTOR_CONTEXT_INITIAL_INPUT",
]
