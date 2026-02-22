"""Unified API layer for hexDAG.

This module provides the shared business logic that both the MCP server
and hexdag-studio REST API consume. By using the same functions, feature
parity between interfaces is guaranteed by design.

Usage
-----
MCP server::

    from hexdag import api
    import json

    @mcp.tool()
    def list_nodes() -> str:
        return json.dumps(api.components.list_nodes(), indent=2)

Studio REST API::

    from hexdag import api

    @router.get("/components/nodes")
    async def get_nodes():
        return api.components.list_nodes()

Available submodules
--------------------
- components: Component discovery (nodes, adapters, tools, macros, tags)
- validation: Pipeline validation
- execution: Pipeline execution
- pipeline: YAML manipulation (init, add/remove/update nodes)
- documentation: Guides and references
- export: Project export
- processes: Process management (pipeline runs, scheduling, entity state)
- vfs: Virtual filesystem (uniform path-based introspection)
"""

from hexdag.api import (
    components,
    documentation,
    execution,
    export,
    pipeline,
    processes,
    validation,
    vfs,
)

__all__ = [
    "components",
    "documentation",
    "execution",
    "export",
    "pipeline",
    "processes",
    "validation",
    "vfs",
]
