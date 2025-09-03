"""HexDAG Core - Hexagonal Architecture for DAG Orchestration.

Core Framework Components Auto-Registration:
1. When 'registry' is imported below, it triggers ComponentRegistry.__init__
2. Registry.__init__ calls _load_core_components()
3. This imports core modules (hexai.core.nodes, etc.)
4. Core module decorators (@node, @adapter, etc.) register components
5. Components become available in 'core' namespace

Users don't need to import core components - they're automatically available:
    from hexai.core.registry import registry
    node = registry.get('passthrough', namespace='core')  # Works!
"""

from hexai.core.registry import ComponentType, registry

# At this point, core components are already loaded and registered
# via the registry initialization process.

__all__ = ["registry", "ComponentType"]
