"""Orchestration layer for DAG execution.

This package contains all orchestrator-related components including the main
Orchestrator class, configuration, builders, and execution components.

The orchestration layer is responsible for:
- Executing DirectedGraphs in topological order
- Managing concurrent execution with wave-based parallelism
- Providing checkpoint/resume capabilities
- Supporting per-node port configuration
- Coordinating policies and events

Main exports
------------
Orchestrator : The main DAG execution engine
OrchestratorConfig : Configuration dataclass
OrchestratorBuilder : Fluent builder for creating orchestrators
ExecutionState : Model for checkpoint state
ExecutionContext : Context for node execution metadata
PendingApproval : Model for HITL approval requests

Examples
--------
Example usage::

    from hexdag.core.orchestration import Orchestrator, OrchestratorConfig
    config = OrchestratorConfig(max_concurrent_nodes=5)
    orchestrator = Orchestrator(config=config)
"""

from hexdag.core.orchestration.body_executor import BodyExecutor, BodyExecutorError
from hexdag.core.orchestration.components import CheckpointManager
from hexdag.core.orchestration.models import (
    CheckpointState,
    NodeExecutionContext,
    OrchestratorConfig,
)

__all__ = [
    "BodyExecutor",
    "BodyExecutorError",
    "CheckpointManager",
    "CheckpointState",
    "NodeExecutionContext",
    "OrchestratorConfig",
]
