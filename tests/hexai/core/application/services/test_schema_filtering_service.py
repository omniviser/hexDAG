"""Tests for SchemaFilteringService application service."""

import pytest

from hexai.core.application.services.schema_filtering_service import SchemaFilteringService
from hexai.core.domain.services.relationship_service import RelationshipService


class TestSchemaFilteringService:
    """Test cases for SchemaFilteringService."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.relationship_service = RelationshipService()
        self.service = SchemaFilteringService(self.relationship_service)

        # Sample schema data
        self.schemas = {
            "users": {
                "columns": {"id": "int", "name": "varchar", "email": "varchar"},
                "primary_keys": ["id"],
            },
            "orders": {
                "columns": {"id": "int", "user_id": "int", "total": "decimal"},
                "primary_keys": ["id"],
            },
            "products": {
                "columns": {"id": "int", "name": "varchar", "price": "decimal"},
                "primary_keys": ["id"],
            },
            "order_items": {
                "columns": {"order_id": "int", "product_id": "int", "quantity": "int"},
                "primary_keys": ["order_id", "product_id"],
            },
        }

        # Sample relationships
        self.relationships = [
            {"from": "orders.user_id", "to": "users.id"},
            {"from": "order_items.order_id", "to": "orders.id"},
            {"from": "order_items.product_id", "to": "products.id"},
        ]

    def test_filter_by_ontology_string_nodes(self) -> None:
        """Test filtering by ontology with string node names."""
        ontology_nodes = ["users", "products"]
        result = self.service.filter_by_ontology(self.schemas, ontology_nodes, self.relationships)

        # Should include core tables plus intermediate table (orders, order_items)
        assert "users" in result
        assert "products" in result
        assert "orders" in result
        assert "order_items" in result

    def test_filter_by_ontology_dict_nodes(self) -> None:
        """Test filtering by ontology with dictionary node formats."""
        ontology_nodes = [
            {"name": "users", "database_metadata": {"db_table": "users"}},
            {"database_metadata": {"db_table": "products"}},
        ]
        result = self.service.filter_by_ontology(self.schemas, ontology_nodes, self.relationships)

        assert "users" in result
        assert "products" in result

    def test_filter_by_ontology_fuzzy_matching(self) -> None:
        """Test fuzzy matching of table names."""
        ontology_nodes = ["user", "product"]  # Partial matches
        result = self.service.filter_by_ontology(self.schemas, ontology_nodes)

        # Should match users and products via fuzzy matching
        assert "users" in result
        assert "products" in result

    def test_filter_by_ontology_single_table(self) -> None:
        """Test filtering with single table - no intermediate expansion."""
        ontology_nodes = ["users"]
        result = self.service.filter_by_ontology(self.schemas, ontology_nodes)

        # Should only return the single matched table
        assert result == ["users"]

    def test_filter_by_context(self) -> None:
        """Test filtering by ontology context."""
        context = [
            {"db_table": "users"},
            {"db_table": "orders"},
            {"db_table": "nonexistent"},  # Should be ignored
        ]
        result = self.service.filter_by_context(self.schemas, context)

        assert "users" in result
        assert "orders" in result
        assert "nonexistent" not in result
        assert len(result) == 2

    def test_filter_by_context_empty(self) -> None:
        """Test filtering with empty context returns all tables."""
        result = self.service.filter_by_context(self.schemas, [])

        # Should return all tables when no context provided
        assert set(result) == set(self.schemas.keys())

    def test_filter_by_query_table_name_match(self) -> None:
        """Test filtering by query with table name matches."""
        query = "users and orders data"
        result = self.service.filter_by_query(self.schemas, query)

        assert "users" in result
        assert "orders" in result

    def test_filter_by_query_column_name_match(self) -> None:
        """Test filtering by query with column name matches."""
        query = "show me all emails"
        result = self.service.filter_by_query(self.schemas, query)

        # Should match users table which has email column
        assert "users" in result

    def test_filter_by_query_empty(self) -> None:
        """Test filtering with empty query returns all tables."""
        result = self.service.filter_by_query(self.schemas, "")

        assert set(result) == set(self.schemas.keys())

    def test_filter_by_query_no_matches(self) -> None:
        """Test filtering when query matches nothing."""
        query = "nonexistent table or column"
        result = self.service.filter_by_query(self.schemas, query)

        # Should return all tables when no matches found
        assert set(result) == set(self.schemas.keys())

    def test_matches_table_name_exact(self) -> None:
        """Test exact table name matching."""
        assert self.service._matches_table_name("users", "users")
        assert not self.service._matches_table_name("users", "orders")

    def test_matches_table_name_fuzzy(self) -> None:
        """Test fuzzy table name matching."""
        assert self.service._matches_table_name("users", "user")
        assert self.service._matches_table_name("user_accounts", "user")
        assert self.service._matches_table_name("user", "user_accounts")

    def test_matches_table_name_case_insensitive(self) -> None:
        """Test case-insensitive matching."""
        assert self.service._matches_table_name("Users", "users")
        assert self.service._matches_table_name("USERS", "user")
