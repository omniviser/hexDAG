"""Tests for SQLAlchemy adapter."""

import pytest

from hexai.adapters.sqlalchemy_adapter import SQLAlchemyAdapter


@pytest.fixture
@pytest.mark.skip(reason="no way of currently testing this, we need a live db")
async def db():
    """Provide test database connection."""
    dsn = "postgresql+asyncpg://postgres:postgres@localhost/test"
    adapter = SQLAlchemyAdapter(dsn)
    await adapter.connect()

    # Create test tables
    await adapter.query_raw(
        """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            active BOOLEAN DEFAULT true
        )
    """
    )

    await adapter.query_raw(
        """
        INSERT INTO users (name, email, active) VALUES
        ('Alice', 'alice@test.com', true),
        ('Bob', 'bob@test.com', false)
    """
    )

    yield adapter

    # Cleanup
    await adapter.query_raw("DROP TABLE IF EXISTS users")
    await adapter.disconnect()


@pytest.mark.skip(reason="no way of currently testing this, we need a live db")
async def test_get_table_schemas(db):
    """Test schema detection."""
    schemas = await db.get_table_schemas()

    assert len(schemas) == 1
    users = next(s for s in schemas if s.name == "users")
    assert len(users.columns) == 4
    assert [col.name for col in users.columns] == ["id", "name", "email", "active"]

    id_col = next(c for c in users.columns if c.name == "id")
    assert id_col.primary_key
    assert not id_col.nullable


@pytest.mark.skip(reason="no way of currently testing this, we need a live db")
async def test_query_with_filters(db):
    """Test filtering queries."""
    rows = []
    async for row in db.query("users", filters={"active": True}):
        rows.append(row)

    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"
    assert rows[0]["email"] == "alice@test.com"


@pytest.mark.skip(reason="no way of currently testing this, we need a live db")
async def test_raw_sql_query(db):
    """Test raw SQL execution."""
    rows = []
    async for row in db.query_raw(
        "SELECT * FROM users WHERE email LIKE :pattern", {"pattern": "%@test.com"}
    ):
        rows.append(row)

    assert len(rows) == 2
    assert {row["name"] for row in rows} == {"Alice", "Bob"}


@pytest.mark.skip(reason="no way of currently testing this, we need a live db")
async def test_get_table_statistics(db):
    """Test table statistics."""
    stats = await db.get_table_statistics("users")
    assert stats["row_count"] == 2
