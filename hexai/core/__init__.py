"""HexDAG Core - Hexagonal Architecture for DAG Orchestration.

The new bootstrap-based architecture:
1. Decorators only add metadata (no auto-registration)
2. Components are declared in component_manifest.yaml
3. Call bootstrap_registry() or init_hexdag() to initialize
4. Registry is immutable after bootstrap (in production)

Example usage:
    from hexai.core import init_hexdag, registry

    # Initialize HexDAG
    init_hexdag(dev_mode=True)

    # Now use the registry
    node = registry.get('passthrough', namespace='core')
"""

from hexai.core.bootstrap import bootstrap_registry, ensure_bootstrapped, init_hexdag
from hexai.core.registry import registry
from hexai.core.registry.models import ComponentType

__all__ = [
    "registry",
    "ComponentType",
    "bootstrap_registry",
    "ensure_bootstrapped",
    "init_hexdag",
]
