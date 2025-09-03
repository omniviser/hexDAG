"""HexDAG component registry with unified decorator-based registration.

The registry provides a simple, consistent API for registering both core
and plugin components using decorators.

Examples
--------
>>> # Core component
>>> from hexai.core.registry import node
>>>
>>> @node()
>>> class PassthroughNode:
...     def execute(self, data):
...         return data
>>>
>>> # Plugin component
>>> @node(namespace='my_plugin')
>>> class AnalyzerNode:
...     def execute(self, data):
...         return analyze(data)
>>>
>>> # Access components
>>> from hexai.core.registry import registry
>>> node = registry.get('passthrough_node')
"""

from hexai.core.registry.decorators import adapter, component, memory, node, observer, policy, tool
from hexai.core.registry.registry import ComponentRegistry, registry
from hexai.core.registry.types import ComponentType

__all__ = [
    # Main registry
    "ComponentRegistry",
    "registry",
    # Decorators
    "component",
    "node",
    "tool",
    "adapter",
    "policy",
    "memory",
    "observer",
    # Types
    "ComponentType",
]
