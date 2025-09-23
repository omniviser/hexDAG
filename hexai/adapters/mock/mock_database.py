"""Mock database port implementation for testing."""

from typing import Any

from hexai.core.ports.database import DatabasePort


class MockDatabaseAdapter(DatabasePort):
    """Mock implementation of DatabasePort for testing and demos."""

    def __init__(self) -> None:
        """Initialize with sample database schema."""
        self._table_schemas: dict[str, dict[str, Any]] = {
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

        self._relationships: list[dict[str, Any]] = [
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

        self._indexes: list[dict[str, Any]] = [
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

        self._table_statistics: dict[str, dict[str, Any]] = {
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

    def get_table_schemas(self) -> dict[str, dict[str, Any]]:
        """Get schema information for all tables.

        Returns
        -------
        dict[str, dict[str, Any]]
            Dictionary mapping table names to their schema information
        """
        return self._table_schemas.copy()

    def get_relationships(self) -> list[dict[str, Any]]:
        """Get foreign key relationships between tables.

        Returns
        -------
        list[dict[str, Any]]
            List of relationship information between tables
        """
        return self._relationships.copy()

    def get_indexes(self) -> list[dict[str, Any]]:
        """Get index information for performance optimization.

        Returns
        -------
        list[dict[str, Any]]
            List of index definitions with performance information
        """
        return self._indexes.copy()

    def get_table_statistics(self) -> dict[str, dict[str, Any]]:
        """Get table statistics for query optimization.

        Returns
        -------
        dict[str, dict[str, Any]]
            Dictionary mapping table names to their statistics
        """
        return self._table_statistics.copy()
