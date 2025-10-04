"""Mock database port implementation for testing."""

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from hexai.core.configurable import ConfigurableAdapter
from hexai.core.ports.database import DatabasePort
from hexai.core.registry import adapter


@adapter(name="mock_database", implements_port="database", namespace="plugin")
class MockDatabaseAdapter(DatabasePort, ConfigurableAdapter):
    """Mock implementation of DatabasePort for testing and demos."""

    # Configuration schema for TOML generation
    class Config(BaseModel):
        """Configuration schema for Mock Database adapter."""

        enable_sample_data: bool = Field(
            default=True, description="Whether to initialize with sample database schema"
        )
        delay_seconds: float = Field(
            default=0.0, ge=0.0, description="Artificial delay to simulate database query latency"
        )

    # Type hint for mypy to understand self.config has Config fields
    config: Config

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with configuration.

        Args
        ----
            **kwargs: Configuration options (enable_sample_data, delay_seconds)
        """
        # Initialize config (accessible via self.config.field_name)
        ConfigurableAdapter.__init__(self, **kwargs)

        if not self.config.enable_sample_data:
            self._table_schemas: dict[str, dict[str, Any]] = {}
            self._relationships: list[dict[str, Any]] = []
            self._indexes: list[dict[str, Any]] = []
            self._table_statistics: dict[str, dict[str, Any]] = {}
            return

        self._table_schemas = {
            "customers": {
                "table_name": "customers",
                "columns": {
                    "id": "INTEGER PRIMARY KEY",
                    "customer_name": "VARCHAR(255)",
                    "email": "VARCHAR(255)",
                    "segment": "VARCHAR(50)",
                    "created_date": "TIMESTAMP",
                    "status": "VARCHAR(20)",
                },
                "primary_keys": ["id"],
                "foreign_keys": [],
            },
            "orders": {
                "table_name": "orders",
                "columns": {
                    "id": "INTEGER PRIMARY KEY",
                    "customer_id": "INTEGER",
                    "order_date": "DATE",
                    "order_value": "DECIMAL(10,2)",
                    "status": "VARCHAR(20)",
                    "created_at": "TIMESTAMP",
                },
                "primary_keys": ["id"],
                "foreign_keys": [
                    {
                        "column": "customer_id",
                        "references_table": "customers",
                        "references_column": "id",
                    }
                ],
            },
            "products": {
                "table_name": "products",
                "columns": {
                    "id": "INTEGER PRIMARY KEY",
                    "product_name": "VARCHAR(255)",
                    "category": "VARCHAR(100)",
                    "price": "DECIMAL(8,2)",
                    "created_at": "TIMESTAMP",
                },
                "primary_keys": ["id"],
                "foreign_keys": [],
            },
            "order_items": {
                "table_name": "order_items",
                "columns": {
                    "id": "INTEGER PRIMARY KEY",
                    "order_id": "INTEGER",
                    "product_id": "INTEGER",
                    "quantity": "INTEGER",
                    "unit_price": "DECIMAL(8,2)",
                },
                "primary_keys": ["id"],
                "foreign_keys": [
                    {
                        "column": "order_id",
                        "references_table": "orders",
                        "references_column": "id",
                    },
                    {
                        "column": "product_id",
                        "references_table": "products",
                        "references_column": "id",
                    },
                ],
            },
        }

        self._relationships = [
            {
                "from_table": "orders",
                "from_column": "customer_id",
                "to_table": "customers",
                "to_column": "id",
                "relationship_type": "many_to_one",
            },
            {
                "from_table": "order_items",
                "from_column": "order_id",
                "to_table": "orders",
                "to_column": "id",
                "relationship_type": "many_to_one",
            },
            {
                "from_table": "order_items",
                "from_column": "product_id",
                "to_table": "products",
                "to_column": "id",
                "relationship_type": "many_to_one",
            },
        ]

        self._indexes = [
            {
                "index_name": "idx_customers_email",
                "table_name": "customers",
                "columns": ["email"],
                "index_type": "btree",
                "is_unique": True,
            },
            {
                "index_name": "idx_orders_customer_id",
                "table_name": "orders",
                "columns": ["customer_id"],
                "index_type": "btree",
                "is_unique": False,
            },
            {
                "index_name": "idx_orders_date",
                "table_name": "orders",
                "columns": ["order_date"],
                "index_type": "btree",
                "is_unique": False,
            },
            {
                "index_name": "idx_order_items_order_id",
                "table_name": "order_items",
                "columns": ["order_id"],
                "index_type": "btree",
                "is_unique": False,
            },
        ]

        self._table_statistics = {
            "customers": {
                "row_count": 10000,
                "size_bytes": 2048000,
                "last_updated": "2024-01-15T10:30:00Z",
            },
            "orders": {
                "row_count": 50000,
                "size_bytes": 8192000,
                "last_updated": "2024-01-15T11:45:00Z",
            },
            "products": {
                "row_count": 1000,
                "size_bytes": 512000,
                "last_updated": "2024-01-10T09:00:00Z",
            },
            "order_items": {
                "row_count": 150000,
                "size_bytes": 12288000,
                "last_updated": "2024-01-15T11:45:00Z",
            },
        }

    # Required methods from DatabasePort
    async def aget_table_schemas(self) -> dict[str, dict[str, Any]]:
        """Get schema information for all tables."""
        if self.config.delay_seconds > 0:
            await asyncio.sleep(self.config.delay_seconds)
        return self._table_schemas.copy()

    async def aexecute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a SQL query and return results."""
        if self.config.delay_seconds > 0:
            await asyncio.sleep(self.config.delay_seconds)

        # Mock implementation - returns sample data based on query keywords
        if "customers" in query.lower():
            return [
                {"id": 1, "customer_name": "John Doe", "segment": "Premium"},
                {"id": 2, "customer_name": "Jane Smith", "segment": "Standard"},
            ]
        if "orders" in query.lower():
            return [
                {"id": 101, "customer_id": 1, "order_value": 299.99},
                {"id": 102, "customer_id": 2, "order_value": 149.50},
            ]
        return []

    # Optional methods from DatabasePort
    async def aget_relationships(self) -> list[dict[str, Any]]:
        """Get foreign key relationships between tables."""
        if self.config.delay_seconds > 0:
            await asyncio.sleep(self.config.delay_seconds)
        return self._relationships.copy()

    async def aget_indexes(self) -> list[dict[str, Any]]:
        """Get index information for performance optimization."""
        if self.config.delay_seconds > 0:
            await asyncio.sleep(self.config.delay_seconds)
        return self._indexes.copy()

    async def aget_table_statistics(self) -> dict[str, dict[str, Any]]:
        """Get table statistics for query optimization."""
        if self.config.delay_seconds > 0:
            await asyncio.sleep(self.config.delay_seconds)
        return self._table_statistics.copy()
