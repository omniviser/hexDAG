"""HexDAG Core - Hexagonal Architecture for DAG Orchestration.

The new bootstrap-based architecture:
1. Decorators only add metadata (no auto-registration)
2. Components are declared in TOML configuration
3. Call bootstrap_registry() to initialize
4. Registry is immutable after bootstrap (in production)

Example usage:
    from hexdag.core import bootstrap_registry, registry

    bootstrap_registry(dev_mode=True)

    # Now use the registry
    node = registry.get('passthrough', namespace='core')
"""

from hexdag.core.bootstrap import bootstrap_registry, ensure_bootstrapped
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
from hexdag.core.registry import registry
from hexdag.core.registry.models import ComponentType

__all__ = [
    "registry",
    "ComponentType",
    "bootstrap_registry",
    "ensure_bootstrapped",
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
