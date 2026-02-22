"""VFS provider for /proc/entities/ — entity state introspection.

Delegates to :class:`~hexdag.stdlib.lib.entity_state.EntityState`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from hexdag.kernel.domain.vfs import DirEntry, EntryType, StatResult
from hexdag.kernel.exceptions import VFSError

if TYPE_CHECKING:
    from hexdag.stdlib.lib.entity_state import EntityState


class ProcEntitiesProvider:
    """VFS provider for /proc/entities/ — entity state introspection."""

    def __init__(self, entity_state: EntityState) -> None:
        self._entity_state = entity_state

    async def read(self, relative_path: str) -> str:
        """Read entity state.

        Paths
        -----
        - ``<type>/<id>`` → JSON of current state
        - ``<type>/<id>/history`` → JSON of transition history
        """
        path = relative_path.strip("/")
        if not path:
            raise VFSError("/proc/entities/", "cannot read directory; use readdir")

        parts = path.split("/")
        if len(parts) < 2:
            raise VFSError(
                f"/proc/entities/{path}",
                "expected format: <entity_type>/<entity_id>",
            )

        entity_type = parts[0]
        entity_id = parts[1]

        if len(parts) == 2:
            state: dict[str, Any] | None = await self._entity_state.aget_state(
                entity_type, entity_id
            )
            if state is None:
                raise VFSError(
                    f"/proc/entities/{path}",
                    f"entity '{entity_type}/{entity_id}' not found",
                )
            return json.dumps(state, indent=2, default=str)

        if len(parts) == 3 and parts[2] == "history":
            history: list[dict[str, Any]] = await self._entity_state.aget_history(
                entity_type, entity_id
            )
            return json.dumps(history, indent=2, default=str)

        raise VFSError(f"/proc/entities/{path}", "path not found")

    def _entity_types(self) -> list[str]:
        """Get registered entity types from internal state."""
        return sorted(self._entity_state._machines.keys())

    def _entity_ids(self, entity_type: str) -> list[str]:
        """Get entity IDs for a given type from internal state."""
        return sorted(eid for (etype, eid) in self._entity_state._states if etype == entity_type)

    async def readdir(self, relative_path: str) -> list[DirEntry]:
        """List entity types or entities.

        Paths
        -----
        - ``""`` → list registered entity types
        - ``<type>`` → list entities of that type
        """
        path = relative_path.strip("/")

        if not path:
            return [
                DirEntry(
                    name=t,
                    entry_type=EntryType.DIRECTORY,
                    path=f"/proc/entities/{t}",
                )
                for t in self._entity_types()
            ]

        entity_type = path.split("/")[0]
        return [
            DirEntry(
                name=eid,
                entry_type=EntryType.FILE,
                path=f"/proc/entities/{entity_type}/{eid}",
            )
            for eid in self._entity_ids(entity_type)
        ]

    async def stat(self, relative_path: str) -> StatResult:
        """Get metadata about an entity."""
        path = relative_path.strip("/")

        if not path:
            return StatResult(
                path="/proc/entities",
                entry_type=EntryType.DIRECTORY,
                description="Business entity state machines",
                child_count=len(self._entity_types()),
                capabilities=["read"],
            )

        parts = path.split("/")
        entity_type = parts[0]

        if len(parts) == 1:
            return StatResult(
                path=f"/proc/entities/{entity_type}",
                entry_type=EntryType.DIRECTORY,
                description=f"Entities of type '{entity_type}'",
                child_count=len(self._entity_ids(entity_type)),
                capabilities=["read"],
            )

        entity_id = parts[1]
        state = await self._entity_state.aget_state(entity_type, entity_id)
        if state is None:
            raise VFSError(
                f"/proc/entities/{path}",
                f"entity '{entity_type}/{entity_id}' not found",
            )

        return StatResult(
            path=f"/proc/entities/{entity_type}/{entity_id}",
            entry_type=EntryType.FILE,
            entity_type="entity",
            description=f"{entity_type} '{entity_id}'",
            status=state.get("state"),
            capabilities=["read"],
            tags={"entity_type": entity_type},
        )
