"""Unit tests for SQLAlchemy adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import Boolean, Column, Integer, MetaData, String, Table

from hexdag.adapters.database.sqlalchemy.sqlalchemy_adapter import SQLAlchemyAdapter


class AsyncIteratorMock:
    """Mock async iterator for SQLAlchemy results."""

    def __init__(self, items):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            item = self.items[self.index]
            self.index += 1
            return item
        except IndexError as err:
            raise StopAsyncIteration from err


class AsyncContextManagerMock:
    """Mock that properly handles async context manager protocol."""

    def __init__(self, return_value=None):
        self.return_value = return_value or AsyncMock()

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc, tb):
        pass


@pytest.fixture
def mock_table():
    """Create a mock SQLAlchemy table for testing."""
    metadata = MetaData()
    return Table(
        "users",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String, nullable=False),
        Column("email", String, nullable=False),
        Column("active", Boolean, default=True),
    )


@pytest.fixture
async def adapter(mock_table):
    # 1️⃣ Mock the fetchone result
    mock_result = AsyncMock()
    mock_result.fetchone = AsyncMock(return_value=(42,))

    # 2️⃣ Mock the connection
    connection = AsyncMock()
    connection.execute = AsyncMock(return_value=mock_result)

    # 3️⃣ Async context manager for engine.connect()
    connect_cm = AsyncContextManagerMock(connection)

    # 4️⃣ Mock engine
    engine = MagicMock()
    engine.connect = MagicMock(return_value=connect_cm)

    # 5️⃣ Patch create_async_engine so adapter uses our engine
    with patch(
        "hexdag.adapters.database.sqlalchemy.sqlalchemy_adapter.create_async_engine",
        return_value=engine,
    ):
        adapter = SQLAlchemyAdapter("sqlite+aiosqlite:///:memory:")
        adapter.engine = engine  # assign explicitly just in case
        adapter._metadata.tables = {"users": mock_table}
        yield adapter


@pytest.mark.asyncio
async def test_query_with_filters(adapter):
    """Test query building with filters."""
    # Setup
    mock_rows = [
        MagicMock(_mapping={"name": "Alice", "email": "alice@test.com", "active": True}),
    ]

    # Create proper async iterator result
    result = AsyncIteratorMock(mock_rows)

    # Set up connection chain
    connection = AsyncMock()
    connection.stream = AsyncMock(return_value=result)

    # Use MagicMock for connect to avoid coroutine issues
    connect_cm = AsyncContextManagerMock(connection)
    adapter.engine.connect = MagicMock(return_value=connect_cm)

    # Execute
    rows = []
    async for row in adapter.query("users", filters={"active": True}):
        rows.append(row)

    # Assert
    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"
    assert rows[0]["email"] == "alice@test.com"


@pytest.mark.asyncio
async def test_get_table_statistics(adapter):
    # Mock row returned by fetchone
    mock_row = (42,)

    # Mock result of execute method
    mock_result = AsyncMock()
    mock_result.fetchone = AsyncMock(return_value=mock_row)

    # Mock connection that has execute returning the above result
    mock_connection = AsyncMock()
    mock_connection.execute = AsyncMock(return_value=mock_result)

    # Use AsyncContextManagerMock to mock engine.connect:
    mock_connect_cm = AsyncContextManagerMock(mock_connection)

    # Assign this to engine.connect (MagicMock returning the async context manager)
    adapter.engine.connect = MagicMock(return_value=mock_connect_cm)

    # Call the function under test
    stats = await adapter.get_table_statistics("users")

    assert stats is not None
    assert stats["row_count"] == 42
