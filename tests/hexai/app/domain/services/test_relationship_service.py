"""Tests for RelationshipService domain service."""

import pytest

from hexai.core.domain.services.relationship_service import RelationshipService


class TestRelationshipService:
    """Test cases for RelationshipService."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.service = RelationshipService()

        # Sample relationship data
        self.relationships = [
            {"from": "orders.customer_id", "to": "customers.id"},
            {"from": "order_items.order_id", "to": "orders.id"},
            {"from": "order_items.product_id", "to": "products.id"},
            "users.role_id -> roles.id",  # String format
        ]

    def test_build_relationship_graph(self) -> None:
        """Test building bidirectional relationship graph."""
        graph = self.service.build_relationship_graph(self.relationships)

        # Check that graph is bidirectional
        assert "customers" in graph["orders"]
        assert "orders" in graph["customers"]
        assert "orders" in graph["order_items"]
        assert "order_items" in graph["orders"]
        assert "users" in graph["roles"]
        assert "roles" in graph["users"]

    def test_find_path_tables_direct_connection(self) -> None:
        """Test finding path between directly connected tables."""
        graph = self.service.build_relationship_graph(self.relationships)
        path = self.service.find_path_tables("orders", "customers", graph)

        assert "orders" in path
        assert "customers" in path
        assert len(path) == 2

    def test_find_path_tables_indirect_connection(self) -> None:
        """Test finding path through intermediate tables."""
        graph = self.service.build_relationship_graph(self.relationships)
        path = self.service.find_path_tables("customers", "products", graph)

        # Should go: customers -> orders -> order_items -> products
        assert "customers" in path
        assert "orders" in path
        assert "order_items" in path
        assert "products" in path

    def test_find_path_tables_no_connection(self) -> None:
        """Test finding path when no connection exists."""
        graph = self.service.build_relationship_graph(self.relationships)
        path = self.service.find_path_tables("customers", "roles", graph)

        # No connection between customers and roles
        assert len(path) == 0

    def test_get_tables_with_relationships(self) -> None:
        """Test getting tables with intermediate relationships."""
        core_tables = ["customers", "products"]
        result = self.service.get_tables_with_relationships(core_tables, self.relationships)

        # Should include core tables plus intermediates
        assert "customers" in result
        assert "products" in result
        assert "orders" in result
        assert "order_items" in result

    def test_get_connected_tables(self) -> None:
        """Test getting directly connected tables."""
        graph = self.service.build_relationship_graph(self.relationships)
        connected = self.service.get_connected_tables(["orders"], graph)

        assert "orders" in connected
        assert "customers" in connected
        assert "order_items" in connected

    def test_is_connected(self) -> None:
        """Test checking if tables are connected."""
        graph = self.service.build_relationship_graph(self.relationships)

        assert self.service.is_connected("customers", "products", graph)
        assert not self.service.is_connected("customers", "roles", graph)

    def test_get_relationship_distance(self) -> None:
        """Test calculating relationship distance."""
        graph = self.service.build_relationship_graph(self.relationships)

        # Direct connection
        assert self.service.get_relationship_distance("orders", "customers", graph) == 1

        # Through intermediate tables
        distance = self.service.get_relationship_distance("customers", "products", graph)
        assert distance == 3  # customers -> orders -> order_items -> products

        # No connection
        assert self.service.get_relationship_distance("customers", "roles", graph) == -1

    def test_empty_relationships(self) -> None:
        """Test handling empty relationship list."""
        empty_graph = self.service.build_relationship_graph([])
        assert len(empty_graph) == 0

        result = self.service.get_tables_with_relationships(["table1"], [])
        assert result == ["table1"]

    def test_malformed_relationships(self) -> None:
        """Test handling malformed relationship data."""
        malformed = [
            {"from": "incomplete"},  # Missing 'to'
            "malformed_string",      # No arrow
            {"to": "also_incomplete"}, # Missing 'from'
        ]

        graph = self.service.build_relationship_graph(malformed)
        assert len(graph) == 0  # Should handle gracefully
