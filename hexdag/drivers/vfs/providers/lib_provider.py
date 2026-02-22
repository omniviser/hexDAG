"""VFS provider for /lib/ — component introspection.

Handles: ``/lib/nodes/``, ``/lib/adapters/``, ``/lib/macros/``,
``/lib/tools/``, ``/lib/tags/``

Delegates to existing ``api/components.py`` discovery functions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from hexdag.kernel.domain.vfs import DirEntry, EntryType, StatResult
from hexdag.kernel.exceptions import VFSError


@dataclass(frozen=True)
class _EntityTypeConfig:
    """Per-entity-type configuration for the /lib/ namespace."""

    lookup_key: str  # dict key for individual item lookup
    singular: str  # singular form for entity_type field
    list_fn_name: str  # function name on api.components


_ENTITY_REGISTRY: dict[str, _EntityTypeConfig] = {
    "nodes": _EntityTypeConfig(
        lookup_key="kind",
        singular="node",
        list_fn_name="list_nodes",
    ),
    "adapters": _EntityTypeConfig(
        lookup_key="name",
        singular="adapter",
        list_fn_name="list_adapters",
    ),
    "macros": _EntityTypeConfig(
        lookup_key="name",
        singular="macro",
        list_fn_name="list_macros",
    ),
    "tools": _EntityTypeConfig(
        lookup_key="name",
        singular="tool",
        list_fn_name="list_tools",
    ),
    "tags": _EntityTypeConfig(
        lookup_key="name",
        singular="tag",
        list_fn_name="list_tags",
    ),
}


def _get_config(entity_type: str) -> _EntityTypeConfig:
    """Look up entity type config, raising VFSError if unknown."""
    config = _ENTITY_REGISTRY.get(entity_type)
    if config is None:
        raise VFSError(
            f"/lib/{entity_type}",
            f"unknown entity type: {entity_type}",
        )
    return config


class LibProvider:
    """VFS provider for /lib/ — component introspection.

    All discovery logic stays in ``api/components.py``. This provider
    is a thin VFS adapter over those functions.
    """

    def _list_entities(self, entity_type: str) -> list[dict[str, Any]]:
        """Call the appropriate api.components list function."""
        from hexdag.api import components

        config = _get_config(entity_type)
        result: list[dict[str, Any]] = getattr(components, config.list_fn_name)()
        return result

    def _find_entity(self, entity_type: str, entity_name: str) -> dict[str, Any] | None:
        """Find a single entity by name within an entity type."""
        config = _get_config(entity_type)
        for entity in self._list_entities(entity_type):
            if entity.get(config.lookup_key) == entity_name or entity.get("name") == entity_name:
                return entity
        return None

    async def read(self, relative_path: str) -> str:
        """Read content at a relative path under /lib/.

        Paths
        -----
        - ``nodes/llm_node`` → JSON of that node's info dict
        - ``nodes/llm_node/schema`` → JSON schema for the node
        - ``adapters/OpenAIAdapter`` → JSON of adapter info
        """
        parts = relative_path.strip("/").split("/")
        if not parts or not parts[0]:
            raise VFSError(f"/lib/{relative_path}", "cannot read directory; use readdir")

        entity_type = parts[0]
        config = _get_config(entity_type)

        if len(parts) == 1:
            entities = self._list_entities(entity_type)
            return json.dumps(entities, indent=2)

        entity_name = parts[1]
        entity = self._find_entity(entity_type, entity_name)
        if entity is None:
            raise VFSError(
                f"/lib/{relative_path}",
                f"{config.singular} '{entity_name}' not found",
            )

        if len(parts) == 2:
            return json.dumps(entity, indent=2)

        if len(parts) == 3 and parts[2] == "schema":
            from hexdag.api import components

            schema = components.get_component_schema(config.singular, entity_name)
            return json.dumps(schema, indent=2)

        raise VFSError(f"/lib/{relative_path}", "path not found")

    async def readdir(self, relative_path: str) -> list[DirEntry]:
        """List entries under /lib/.

        Paths
        -----
        - ``""`` → entity types: nodes, adapters, macros, tools, tags
        - ``nodes`` → all nodes
        - ``adapters`` → all adapters
        """
        path = relative_path.strip("/")

        if not path:
            return [
                DirEntry(name=t, entry_type=EntryType.DIRECTORY, path=f"/lib/{t}")
                for t in _ENTITY_REGISTRY
            ]

        config = _ENTITY_REGISTRY.get(path)
        if config is None:
            raise VFSError(f"/lib/{relative_path}", "not a directory")

        entities = self._list_entities(path)
        return [
            DirEntry(
                name=e.get(config.lookup_key, e.get("name", "unknown")),
                entry_type=EntryType.FILE,
                path=f"/lib/{path}/{e.get(config.lookup_key, e.get('name', 'unknown'))}",
            )
            for e in entities
        ]

    async def stat(self, relative_path: str) -> StatResult:
        """Get metadata about a path under /lib/."""
        from hexdag.api import components

        path = relative_path.strip("/")

        if not path:
            return StatResult(
                path="/lib",
                entry_type=EntryType.DIRECTORY,
                description="hexDAG component library",
                child_count=len(_ENTITY_REGISTRY),
                capabilities=["read"],
            )

        parts = path.split("/")
        entity_type = parts[0]
        config = _get_config(entity_type)

        if len(parts) == 1:
            entities = self._list_entities(entity_type)
            return StatResult(
                path=f"/lib/{entity_type}",
                entry_type=EntryType.DIRECTORY,
                description=f"Available {entity_type}",
                child_count=len(entities),
                capabilities=["read"],
            )

        entity_name = parts[1]
        entity = self._find_entity(entity_type, entity_name)
        if entity is None:
            raise VFSError(
                f"/lib/{relative_path}",
                f"{config.singular} '{entity_name}' not found",
            )

        module_path = entity.get("module_path")
        is_builtin = components.is_builtin(module_path) if module_path else False
        tags: dict[str, str] = {}
        if is_builtin:
            tags["is_builtin"] = "true"
        if entity.get("port_type"):
            tags["port_type"] = entity["port_type"]

        return StatResult(
            path=f"/lib/{path}",
            entry_type=EntryType.FILE,
            entity_type=config.singular,
            description=entity.get("description"),
            module_path=module_path,
            capabilities=["read"],
            tags=tags,
        )
