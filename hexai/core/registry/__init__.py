"""HexDAG component registry with unified decorator-based registration.

The registry provides a simple, consistent API for registering both core
and plugin components using decorators. All decorators accept strings for
component types and namespaces, making them easy to use.

System Namespaces
-----------------
- 'core': Protected namespace for hexDAG core components
- 'user': Default namespace for user-defined components
- 'plugin': Standard namespace for plugin components
- Any other string: Custom plugin namespace

Component Types
---------------
- 'node': Processing nodes in the DAG
- 'tool': External tools/functions
- 'adapter': Data adapters
- 'policy': Execution policies
- 'memory': Memory/storage components
- 'observer': Event observers

Examples
--------
>>> # User component (default)
>>> from hexai.core.registry import node
>>>
>>> @node(namespace='user')
>>> class MyProcessor:
...     def execute(self, data):
...         return process(data)
>>>
>>> # Plugin component
>>> @node(namespace='my_plugin')
>>> class AnalyzerNode:
...     def execute(self, data):
...         return analyze(data)
>>>
>>> # Access components
>>> from hexai.core.registry import registry
>>> processor = registry.get('my_processor', namespace='user')

Note: The `introspection` module is optional and not exported by default.
It's primarily used for testing and CLI tooling. Import it directly if needed:
>>> from hexai.core.registry.introspection import extract_port_methods
"""

from hexai.core.registry.decorators import (
    adapter,
    agent_node,
    component,
    function_node,
    llm_node,
    memory,
    node,
    observer,
    policy,
    port,
    tool,
)
from hexai.core.registry.registry import registry

# Note: ComponentType and NodeSubtype enums are internal.
# External users should use strings: 'node', 'tool', etc.

__all__ = [
    # Main registry
    "registry",
    # Decorators (all accept strings)
    "component",
    "node",
    "tool",
    "adapter",
    "policy",
    "memory",
    "observer",
    "port",
    # Node subtype decorators
    "function_node",
    "llm_node",
    "agent_node",
]
