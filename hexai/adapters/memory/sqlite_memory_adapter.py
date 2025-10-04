"""SQLite-backed Memory adapter that bridges DatabasePort to Memory Port.

This adapter allows using SQLite databases as a Memory storage backend,
providing persistent key-value storage with SQL database benefits.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hexai.core.configurable import ConfigurableAdapter
from hexai.core.logging import get_logger
from hexai.core.ports.database import DatabasePort
from hexai.core.registry.decorators import adapter

logger = get_logger(__name__)


@adapter(
    name="sqlite_memory",
    implements_port="memory",
    description="SQLite-backed persistent memory storage using key-value table",
)
class SQLiteMemoryAdapter(ConfigurableAdapter):
    """Memory adapter backed by SQLite database.

    Provides persistent key-value storage using SQLite, bridging the
    DatabasePort and Memory Port. Automatically creates a key-value table
    and provides async get/set operations.

    This adapter is ideal for:
    - Checkpoint persistence
    - Configuration storage
    - Any key-value data that needs SQL database benefits

    Parameters
    ----------
    database : DatabasePort
        SQLite database adapter (typically SQLiteAdapter)
    table_name : str, default="memory_store"
        Name of the key-value table
    auto_init : bool, default=True
        Automatically create table on first use

    Examples
    --------
    Example usage::

        from hexai.adapters.database.sqlite import SQLiteAdapter
        db = SQLiteAdapter(db_path="memory.db")
        memory = SQLiteMemoryAdapter(database=db, table_name="memory_store")
    """

    class Config(BaseModel):
        model_config = ConfigDict(frozen=True)

        """Configuration schema for SQLite Memory adapter."""

        table_name: str = Field(
            default="memory_store",
            description="Name of the key-value table in SQLite database",
        )
        auto_init: bool = Field(
            default=True, description="Automatically create table schema if it doesn't exist"
        )

    # Type hint for mypy to understand self.config has Config fields
    config: Config

    def __init__(
        self, database: DatabasePort, table_name: str = "memory_store", auto_init: bool = True
    ) -> None:
        """Initialize SQLite memory adapter.

        Parameters
        ----------
        database : DatabasePort
            SQLite database adapter
        table_name : str
            Name of the key-value table
        auto_init : bool
            Automatically create table if it doesn't exist

        """
        # Initialize config via ConfigurableAdapter
        ConfigurableAdapter.__init__(self, table_name=table_name, auto_init=auto_init)

        # Validate table name to prevent SQL injection
        self._validate_table_name(self.config.table_name)

        self.database = database
        self._initialized = False

    @staticmethod
    def _validate_table_name(table_name: str) -> None:
        """Validate table name to prevent SQL injection.

        Parameters
        ----------
        table_name : str
            Table name to validate

        Raises
        ------
        ValueError
            If table name contains invalid characters
        """
        import re

        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name):
            raise ValueError(
                f"Invalid table name '{table_name}'. "
                "Table names must start with a letter or underscore "
                "and contain only letters, numbers, and underscores."
            )

    async def _ensure_table(self) -> None:
        """Create key-value table if it doesn't exist."""
        if self._initialized:
            return

        # Table name is validated in __init__, safe to use in f-string
        sql = f"""
        CREATE TABLE IF NOT EXISTS {self.config.table_name} (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """  # nosec B608 - table_name is validated
        await self.database.aexecute_query(sql)
        self._initialized = True
        logger.debug(f"Initialized table '{self.config.table_name}' for memory storage")

    async def aget(self, key: str) -> Any:
        """Retrieve a value from memory.

        Parameters
        ----------
        key : str
            The key to retrieve

        Returns
        -------
        Any
            The stored value, or None if key doesn't exist
        """
        if self.config.auto_init:
            await self._ensure_table()

        # Table name is validated, user data in parameters
        sql = f"SELECT value FROM {self.config.table_name} WHERE key = :key"  # nosec B608
        rows = await self.database.aexecute_query(sql, {"key": key})

        if not rows:
            return None

        return rows[0]["value"]

    async def aset(self, key: str, value: Any) -> None:
        """Store a value in memory.

        Parameters
        ----------
        key : str
            The key to store under
        value : Any
            The value to store (must be serializable to string)
        """
        if self.config.auto_init:
            await self._ensure_table()

        # SQLite doesn't support standard UPSERT, use INSERT OR REPLACE
        # Table name is validated, user data in parameters
        sql = f"""
        INSERT OR REPLACE INTO {self.config.table_name} (key, value, updated_at)
        VALUES (:key, :value, CURRENT_TIMESTAMP)
        """  # nosec B608
        await self.database.aexecute_query(sql, {"key": key, "value": str(value)})
        logger.debug(f"Stored key '{key}' in table '{self.config.table_name}'")

    async def adelete(self, key: str) -> bool:
        """Delete a key from memory.

        Parameters
        ----------
        key : str
            The key to delete

        Returns
        -------
        bool
            True if key existed and was deleted, False otherwise
        """
        if self.config.auto_init:
            await self._ensure_table()

        # Check if key exists first
        exists = await self.aget(key)
        if exists is None:
            return False

        # Table name is validated, user data in parameters
        sql = f"DELETE FROM {self.config.table_name} WHERE key = :key"  # nosec B608
        await self.database.aexecute_query(sql, {"key": key})
        logger.debug(f"Deleted key '{key}' from table '{self.config.table_name}'")
        return True

    async def alist_keys(self, prefix: str | None = None) -> list[str]:
        """List all keys in memory, optionally filtered by prefix.

        Parameters
        ----------
        prefix : str | None
            Optional prefix to filter keys

        Returns
        -------
        list[str]
            List of matching keys
        """
        if self.config.auto_init:
            await self._ensure_table()

        if prefix:
            # Table name is validated, user data in parameters
            sql = f"SELECT key FROM {self.config.table_name} WHERE key LIKE :prefix"  # nosec B608
            rows = await self.database.aexecute_query(sql, {"prefix": f"{prefix}%"})
        else:
            # Table name is validated
            sql = f"SELECT key FROM {self.config.table_name}"  # nosec B608
            rows = await self.database.aexecute_query(sql)

        return [row["key"] for row in rows]

    async def aclear(self) -> None:
        """Clear all keys from memory."""
        if self.config.auto_init:
            await self._ensure_table()

        # Table name is validated
        sql = f"DELETE FROM {self.config.table_name}"  # nosec B608
        await self.database.aexecute_query(sql)
        logger.info(f"Cleared all keys from table '{self.config.table_name}'")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"SQLiteMemoryAdapter(table='{self.config.table_name}')"
