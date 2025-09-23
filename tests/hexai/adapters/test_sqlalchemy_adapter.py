"""Tests for SQLAlchemy adapter with SQLite."""

import pytest
import os

from hexai.adapters.sqlalchemy_adapter import SQLAlchemyAdapter

@pytest.fixture
async def db():
    """Provide test database connection (SQLite)."""
    dsn = "sqlite+aiosqlite:///:memory:"
    adapter = SQLAlchemyAdapter(dsn)
    await adapter.connect()

    # Create test tables
    await adapter.execute_raw(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            active INTEGER DEFAULT 1
        )
        """
    )

    # Re-reflect metadata after creating tables
    async with adapter.engine.connect() as conn:
        await conn.run_sync(adapter._metadata.reflect)

    await adapter.execute_raw(
        """
        INSERT INTO users (name, email, active) VALUES
        ('Alice', 'alice@test.com', 1),
        ('Bob', 'bob@test.com', 0)
        """
    )

    yield adapter
    await adapter.disconnect()

@pytest.mark.asyncio
async def test_get_table_schemas(db):
    """Test schema detection."""
    schemas = await db.get_table_schemas()

    assert any(s.name == "users" for s in schemas)
    users = next(s for s in schemas if s.name == "users")
    assert len(users.columns) == 4
    assert [col.name for col in users.columns] == ["id", "name", "email", "active"]

    id_col = next(c for c in users.columns if c.name == "id")
    assert id_col.primary_key

@pytest.mark.asyncio
async def test_query_with_filters(db):
    """Test filtering queries."""
    rows = []
    async for row in db.query("users", filters={"active": 1}):
        rows.append(row)

    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"
    assert rows[0]["email"] == "alice@test.com"

@pytest.mark.asyncio
async def test_raw_sql_query(db):
    """Test raw SQL execution."""
    rows = []
    async for row in db.query_raw(
        "SELECT * FROM users WHERE email LIKE :pattern", {"pattern": "%@test.com"}
    ):
        rows.append(row)

    assert len(rows) == 2
    assert {row["name"] for row in rows} == {"Alice", "Bob"}

@pytest.mark.asyncio
async def test_get_table_statistics(db):
    """Test table statistics."""
    stats = await db.get_table_statistics("users")
    assert stats["row_count"] == 2
