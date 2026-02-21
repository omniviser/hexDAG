"""Components used by the orchestrator.

This package contains reusable components that implement specific
responsibilities in the orchestration pipeline:

Primary Components (Recommended):
- ExecutionCoordinator: Unified policy evaluation, input mapping, event notification
- LifecycleManager: Unified pre/post execution lifecycle management

Execution Components (Keep Unchanged):
- NodeExecutor: Executes individual nodes with validation and timeout
- CheckpointManager: Manages checkpoint save/restore and graph filtering

Deprecated Components (Use Unified Managers Instead):
- PolicyCoordinator: -> Use ExecutionCoordinator
- InputMapper: -> Use ExecutionCoordinator
- HealthCheckManager: -> Use LifecycleManager
- SecretManager: -> Use LifecycleManager
- AdapterLifecycleManager: -> Use LifecycleManager
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

from hexdag.core.exceptions import OrchestratorError
from hexdag.core.orchestration.components.checkpoint_manager import CheckpointManager
from hexdag.core.orchestration.components.execution_coordinator import ExecutionCoordinator
from hexdag.core.orchestration.components.lifecycle_manager import LifecycleManager
from hexdag.core.orchestration.components.node_executor import (
    NodeExecutionError,
    NodeExecutor,
    NodeTimeoutError,
)

# Deprecated classes - lazy loaded via __getattr__
_DEPRECATED_MAPPING: dict[str, tuple[str, str, str]] = {
    "PolicyCoordinator": (
        "hexdag.core.orchestration.components.policy_coordinator",
        "PolicyCoordinator",
        "ExecutionCoordinator",
    ),
    "InputMapper": (
        "hexdag.core.orchestration.components.input_mapper",
        "InputMapper",
        "ExecutionCoordinator",
    ),
    "HealthCheckManager": (
        "hexdag.core.orchestration.components.health_check_manager",
        "HealthCheckManager",
        "LifecycleManager",
    ),
    "SecretManager": (
        "hexdag.core.orchestration.components.secret_manager",
        "SecretManager",
        "LifecycleManager",
    ),
    "AdapterLifecycleManager": (
        "hexdag.core.orchestration.components.adapter_lifecycle_manager",
        "AdapterLifecycleManager",
        "LifecycleManager",
    ),
}


def __getattr__(name: str) -> Any:
    """Provide deprecated imports with warnings."""
    if name in _DEPRECATED_MAPPING:
        module_path, class_name, replacement = _DEPRECATED_MAPPING[name]
        warnings.warn(
            f"{class_name} is deprecated and will be removed in a future version. "
            f"Use {replacement} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Primary components (recommended)
    "ExecutionCoordinator",
    "LifecycleManager",
    # Execution components (keep unchanged)
    "CheckpointManager",
    "NodeExecutionError",
    "NodeExecutor",
    "NodeTimeoutError",
    "OrchestratorError",
]
