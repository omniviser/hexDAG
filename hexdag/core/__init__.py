"""HexDAG Core - Hexagonal Architecture for DAG Orchestration.

The registry-free architecture uses Python's import system directly:
- Components are plain Python classes
- YAML uses full module paths (hexdag.builtin.nodes.LLMNode)
- resolve() imports classes dynamically

Example usage:
    from hexdag.core.resolver import resolve

    # Resolve component by module path
    NodeClass = resolve("hexdag.builtin.nodes.LLMNode")
"""

from hexdag.core.configurable import (
    SecretField,
)
from hexdag.core.context import (
    ExecutionContext,
    clear_execution_context,
    get_observer_manager,
    get_policy_manager,
    get_port,
    get_ports,
    get_run_id,
    set_observer_manager,
    set_policy_manager,
    set_ports,
    set_run_id,
)
from hexdag.core.exceptions import (
    ConfigurationError,
    DependencyError,
    HexDAGError,
    OrchestratorError,
    ResourceNotFoundError,
    TypeMismatchError,
    ValidationError,
)
from hexdag.core.protocols import (
    ComponentWithExecute,
    ConfigurablePort,
    DictConvertible,
    HealthCheckable,
    SchemaProvider,
    has_execute_method,
    is_dict_convertible,
    is_schema_type,
)
from hexdag.core.resolver import resolve, resolve_function

__all__ = [
    # Module Resolution
    "resolve",
    "resolve_function",
    # Secret helper
    "SecretField",
    # Execution Context
    "ExecutionContext",
    "clear_execution_context",
    "get_observer_manager",
    "get_policy_manager",
    "get_port",
    "get_ports",
    "get_run_id",
    "set_observer_manager",
    "set_policy_manager",
    "set_ports",
    "set_run_id",
    # Exceptions
    "HexDAGError",
    "ConfigurationError",
    "ValidationError",
    "ResourceNotFoundError",
    "DependencyError",
    "TypeMismatchError",
    "OrchestratorError",
    # Protocols
    "ComponentWithExecute",
    "ConfigurablePort",
    "DictConvertible",
    "HealthCheckable",
    "SchemaProvider",
    "has_execute_method",
    "is_dict_convertible",
    "is_schema_type",
]
