"""Tests for SQL adapters (MySQL, PostgreSQL)."""

import pytest

from hexdag_plugins.storage.sql import MySQLAdapter, PostgreSQLAdapter


class TestSQLAdapterInterface:
    """Test SQL adapter interface and basic functionality."""

    @pytest.mark.asyncio
    async def test_mysql_adapter_creation(self):
        """Test MySQL adapter can be created with config."""
        adapter = MySQLAdapter(connection_string="mysql+aiomysql://test:test@localhost/test")
        assert adapter is not None
        assert (
            adapter.config.connection_string.get_secret_value()
            == "mysql+aiomysql://test:test@localhost/test"
        )
        assert adapter.config.pool_size == 5
        assert adapter.config.max_overflow == 10

    @pytest.mark.asyncio
    async def test_postgresql_adapter_creation(self):
        """Test PostgreSQL adapter can be created with config."""
        adapter = PostgreSQLAdapter(
            connection_string="postgresql+asyncpg://test:test@localhost/test"
        )
        assert adapter is not None
        assert (
            adapter.config.connection_string.get_secret_value()
            == "postgresql+asyncpg://test:test@localhost/test"
        )
        assert adapter.config.pool_size == 5
        assert adapter.config.max_overflow == 10

    @pytest.mark.asyncio
    async def test_mysql_adapter_custom_pool_config(self):
        """Test MySQL adapter with custom pool configuration."""
        adapter = MySQLAdapter(
            connection_string="mysql+aiomysql://test:test@localhost/test",
            pool_size=10,
            max_overflow=20,
            pool_timeout=60.0,
            pool_recycle=7200,
            pool_pre_ping=False,
        )
        assert adapter.config.pool_size == 10
        assert adapter.config.max_overflow == 20
        assert adapter.config.pool_timeout == 60.0
        assert adapter.config.pool_recycle == 7200
        assert adapter.config.pool_pre_ping is False

    @pytest.mark.asyncio
    async def test_postgresql_adapter_custom_pool_config(self):
        """Test PostgreSQL adapter with custom pool configuration."""
        adapter = PostgreSQLAdapter(
            connection_string="postgresql+asyncpg://test:test@localhost/test",
            pool_size=15,
            max_overflow=25,
            pool_timeout=45.0,
            pool_recycle=1800,
            pool_pre_ping=False,
        )
        assert adapter.config.pool_size == 15
        assert adapter.config.max_overflow == 25
        assert adapter.config.pool_timeout == 45.0
        assert adapter.config.pool_recycle == 1800
        assert adapter.config.pool_pre_ping is False

    @pytest.mark.asyncio
    async def test_adapter_not_setup_error(self):
        """Test that operations fail before setup."""
        adapter = MySQLAdapter(connection_string="mysql+aiomysql://test:test@localhost/test")

        with pytest.raises(RuntimeError, match="Adapter not set up"):
            await adapter.aexecute("SELECT 1")

        with pytest.raises(RuntimeError, match="Adapter not set up"):
            await adapter.afetch_one("SELECT 1")

        with pytest.raises(RuntimeError, match="Adapter not set up"):
            await adapter.afetch_all("SELECT 1")

    @pytest.mark.asyncio
    async def test_health_check_before_setup(self):
        """Test health check before setup returns unhealthy status."""
        adapter = MySQLAdapter(connection_string="mysql+aiomysql://test:test@localhost/test")

        health = await adapter.ahealth_check()
        assert health.status == "unhealthy"
        assert "not initialized" in health.details.get("message", "").lower()


# Integration tests require real database connections
# These are marked with @pytest.mark.integration and skipped by default


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mysql_adapter_integration():
    """Integration test for MySQL adapter with real database.

    Requires MySQL running at localhost with test database.
    Set MYSQL_TEST_URL environment variable or skip this test.
    """
    import os

    connection_string = os.getenv("MYSQL_TEST_URL", "mysql+aiomysql://test:test@localhost/test")

    adapter = MySQLAdapter(connection_string=connection_string)

    try:
        await adapter.asetup()

        # Test health check
        health = await adapter.ahealth_check()
        assert health.is_healthy() is True

        # Test table creation
        await adapter.aexecute(
            """
            CREATE TABLE IF NOT EXISTS test_table (
                id INT PRIMARY KEY AUTO_INCREMENT,
                name VARCHAR(255),
                value INT
            )
        """
        )

        # Test insert
        await adapter.aexecute(
            "INSERT INTO test_table (name, value) VALUES (:name, :value)",
            {"name": "test", "value": 42},
        )

        # Test fetch_one
        result = await adapter.afetch_one(
            "SELECT * FROM test_table WHERE name = :name", {"name": "test"}
        )
        assert result is not None
        assert result["name"] == "test"
        assert result["value"] == 42

        # Test fetch_all
        results = await adapter.afetch_all("SELECT * FROM test_table")
        assert len(results) >= 1

        # Cleanup
        await adapter.aexecute("DROP TABLE test_table")

    finally:
        await adapter.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_postgresql_adapter_integration():
    """Integration test for PostgreSQL adapter with real database.

    Requires PostgreSQL running at localhost with test database.
    Set POSTGRES_TEST_URL environment variable or skip this test.
    """
    import os

    connection_string = os.getenv(
        "POSTGRES_TEST_URL", "postgresql+asyncpg://test:test@localhost/test"
    )

    adapter = PostgreSQLAdapter(connection_string=connection_string)

    try:
        await adapter.asetup()

        # Test health check
        health = await adapter.ahealth_check()
        assert health.is_healthy() is True

        # Test table creation
        await adapter.aexecute(
            """
            CREATE TABLE IF NOT EXISTS test_table (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255),
                value INT
            )
        """
        )

        # Test insert
        await adapter.aexecute(
            "INSERT INTO test_table (name, value) VALUES (:name, :value)",
            {"name": "test", "value": 42},
        )

        # Test fetch_one
        result = await adapter.afetch_one(
            "SELECT * FROM test_table WHERE name = :name", {"name": "test"}
        )
        assert result is not None
        assert result["name"] == "test"
        assert result["value"] == 42

        # Test fetch_all
        results = await adapter.afetch_all("SELECT * FROM test_table")
        assert len(results) >= 1

        # Cleanup
        await adapter.aexecute("DROP TABLE test_table")

    finally:
        await adapter.aclose()
