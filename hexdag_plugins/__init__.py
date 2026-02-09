"""External plugins for hexDAG framework.

Plugin Convention
-----------------
All plugins in this namespace follow a standard structure::

    hexdag_plugins/
    ├── <plugin_name>/
    │   ├── __init__.py          # Re-exports from adapters/nodes/tools
    │   ├── adapters/            # Adapter implementations
    │   │   ├── __init__.py
    │   │   └── my_adapter.py
    │   ├── nodes/               # Custom node types
    │   │   ├── __init__.py
    │   │   └── my_node.py
    │   ├── tools/               # Agent tools
    │   │   ├── __init__.py
    │   │   └── my_tool.py
    │   └── ports/               # Custom port protocols (optional)
    │       └── __init__.py

Adapter Requirements
--------------------
Adapters MUST inherit from their port protocol to be auto-discovered:

- LLM adapters: inherit from ``LLM``, ``SupportsGeneration``, etc.
- Memory adapters: inherit from ``Memory``
- Database adapters: inherit from ``DatabasePort`` or ``SQLAdapter``
- Secret adapters: inherit from ``SecretPort``
- Storage adapters: inherit from ``FileStoragePort`` or ``VectorStorePort``
- Tool adapters: inherit from ``ToolRouter``

Example::

    from hexdag.core.ports.llm import LLM

    class MyCustomLLMAdapter(LLM):
        async def aresponse(self, messages):
            ...

Node Requirements
-----------------
Custom nodes should inherit from ``BaseNodeFactory``::

    from hexdag.builtin.nodes.base_node_factory import BaseNodeFactory

    class MyCustomNode(BaseNodeFactory):
        def __call__(self, name: str, **kwargs) -> NodeSpec:
            ...

Available Plugins
-----------------
- ``azure``: Azure cloud adapters (OpenAI, Cosmos, KeyVault, Blob)
- ``storage``: File and vector storage adapters
- ``hexdag_etl``: ETL pipeline nodes
"""
