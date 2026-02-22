"""HexDAG Core - Hexagonal Architecture for DAG Orchestration.

The registry-free architecture uses Python's import system directly:
- Components are plain Python classes
- YAML uses full module paths (hexdag.stdlib.nodes.LLMNode)
- resolve() imports classes dynamically

Example usage:
    from hexdag.kernel.resolver import resolve

    # Resolve component by module path
    NodeClass = resolve("hexdag.stdlib.nodes.LLMNode")
"""

from hexdag.kernel.configurable import (
    SecretField,
)
from hexdag.kernel.context import (
    ExecutionContext,
    clear_execution_context,
    get_observer_manager,
    get_port,
    get_ports,
    get_run_id,
    set_observer_manager,
    set_ports,
    set_run_id,
)
from hexdag.kernel.exceptions import (
    ConfigurationError,
    DependencyError,
    HexDAGError,
    OrchestratorError,
    ResourceNotFoundError,
    TypeMismatchError,
    ValidationError,
    VFSError,
)
from hexdag.kernel.protocols import (
    ComponentWithExecute,
    ConfigurablePort,
    DictConvertible,
    HealthCheckable,
    SchemaProvider,
    has_execute_method,
    is_dict_convertible,
    is_schema_type,
)
from hexdag.kernel.resolver import resolve, resolve_function

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
    "get_port",
    "get_ports",
    "get_run_id",
    "set_observer_manager",
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
    "VFSError",
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
