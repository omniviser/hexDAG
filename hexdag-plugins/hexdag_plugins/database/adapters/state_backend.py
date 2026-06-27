# type: ignore
"""SQLAlchemy state backend — the domain DB as the source of truth.

Implements ``SupportsStateBackend`` against status columns on the
application's own tables.  Wire it to a dual-mode adapter so state writes
join the pipeline run's transaction::

    adapter = SQLAlchemyAdapter(session_factory=app_sessionmaker)
    backend = SQLAlchemyStateBackend(
        adapter,
        tables={
            "load": {"table": "loads", "id_column": "id", "state_column": "status"},
            "escalation": {"table": "escalations"},  # id/status by default
        },
    )
    entity_state = EntityState(state_backend=backend)

Writes are compare-and-swap when ``expected`` is given::

    UPDATE loads SET status = :state WHERE id = :id AND status = :expected

Zero rows updated raises ``StaleStateError`` — a concurrent worker
transitioned the entity first (or the row does not exist).

``SupportsStateBackend`` (the contract) stays in ``hexdag.kernel.ports`` because
stdlib ``EntityState`` consumes it; only this implementation lives in the plugin.
"""

from typing import Any

from hexdag.kernel.exceptions import StaleStateError
from sqlalchemy import text

_DEFAULT_ID_COLUMN = "id"
_DEFAULT_STATE_COLUMN = "status"


class SQLAlchemyStateBackend:
    """``SupportsStateBackend`` over status columns in application tables.

    Args:
        db: Adapter exposing ``get_session()`` (dual-mode: state writes
            share the pipeline run's transaction when one is active).
        tables: Map of entity_type to table spec.  Each spec requires
            ``table`` and accepts ``id_column`` (default ``"id"``) and
            ``state_column`` (default ``"status"``).
    """

    def __init__(self, db: Any, tables: dict[str, dict[str, str]]) -> None:
        self._db = db
        self._tables: dict[str, tuple[str, str, str]] = {}
        for entity_type, spec in tables.items():
            self._tables[entity_type] = (
                spec["table"],
                spec.get("id_column", _DEFAULT_ID_COLUMN),
                spec.get("state_column", _DEFAULT_STATE_COLUMN),
            )

    def _spec(self, entity_type: str) -> tuple[str, str, str]:
        spec = self._tables.get(entity_type)
        if spec is None:
            msg = f"No table mapping for entity type {entity_type!r}"
            raise KeyError(msg)
        return spec

    async def aread_state(self, entity_type: str, entity_id: str) -> str | None:
        """Return the entity's current state from its status column."""
        table, id_col, state_col = self._spec(entity_type)
        sql = f"SELECT {state_col} FROM {table} WHERE {id_col} = :entity_id"  # nosec B608
        async with self._db.get_session() as session:
            result = await session.execute(text(sql), {"entity_id": entity_id})
            row = result.fetchone()
            return None if row is None or row[0] is None else str(row[0])

    async def awrite_state(
        self,
        entity_type: str,
        entity_id: str,
        state: str,
        *,
        expected: str | None = None,
    ) -> None:
        """Write the entity's state; compare-and-swap when ``expected`` set."""
        table, id_col, state_col = self._spec(entity_type)
        sql = f"UPDATE {table} SET {state_col} = :state WHERE {id_col} = :entity_id"  # nosec B608
        params: dict[str, Any] = {"state": state, "entity_id": entity_id}
        if expected is not None:
            sql += f" AND {state_col} = :expected"
            params["expected"] = expected

        async with self._db.get_session() as session:
            result = await session.execute(text(sql), params)
            if result.rowcount == 0:
                msg = (
                    f"{entity_type}/{entity_id}: conditional state write to "
                    f"{state!r} matched no rows"
                    + (f" (expected {expected!r})" if expected is not None else "")
                )
                raise StaleStateError(msg)
