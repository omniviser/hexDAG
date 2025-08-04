"""Mock ontology port implementation for testing."""

from datetime import datetime
from typing import Any

from hexai.core.ports.ontology import OntologyPort


class MockOntologyPort(OntologyPort):
    """Mock implementation of OntologyPort for testing and demos."""

    def __init__(self) -> None:
        """Initialize with sample ontology data based on new database schema."""
        # Simple integer IDs as requested
        ontology_id = 1
        tenant_id = 1

        # Node IDs
        customer_node_id = 1
        order_node_id = 2
        product_node_id = 3
        otif_node_id = 4

        # Relation IDs
        customer_order_relation_id = 1
        order_product_relation_id = 2

        # Create ontology
        self._ontology = {
            "id": ontology_id,
            "name": "Bart Ontology",
            "is_active": True,
            "created_at": datetime.now(),
            "tenant_id": tenant_id,
            "last_synced_at": datetime.now(),
        }

        # Create ontology nodes
        self._ontology_nodes = [
            {
                "id": customer_node_id,
                "ontology_id": ontology_id,
                "name": "Customer",
                "alias": "Cust",
                "permissed_roles": ["analyst", "manager"],
                "last_synced_at": datetime.now(),
            },
            {
                "id": order_node_id,
                "ontology_id": ontology_id,
                "name": "Order",
                "alias": "Ord",
                "permissed_roles": ["analyst", "manager", "sales"],
                "last_synced_at": datetime.now(),
            },
            {
                "id": product_node_id,
                "ontology_id": ontology_id,
                "name": "Product",
                "alias": "Prod",
                "permissed_roles": ["analyst", "manager", "product_manager"],
                "last_synced_at": datetime.now(),
            },
            {
                "id": otif_node_id,
                "ontology_id": ontology_id,
                "name": "OTIF",
                "alias": "OnTimeInFull",
                "permissed_roles": ["analyst", "manager", "operations"],
                "last_synced_at": datetime.now(),
            },
        ]

        # Create ontology relations
        self._ontology_relations = [
            {
                "id": customer_order_relation_id,
                "ontology_id": ontology_id,
                "parent_id": customer_node_id,
                "child_id": order_node_id,
                "description": "Customer places orders",
                "last_synced_at": datetime.now(),
            },
            {
                "id": order_product_relation_id,
                "ontology_id": ontology_id,
                "parent_id": order_node_id,
                "child_id": product_node_id,
                "description": "Orders contain products",
                "last_synced_at": datetime.now(),
            },
        ]

    def get_ontologies(self) -> list[dict[str, Any]]:
        """Get all ontologies."""
        return [self._ontology]

    def get_ontology_nodes(self, ontology_id: int | None = None) -> list[dict[str, Any]]:
        """Get all ontology nodes, optionally filtered by ontology."""
        if ontology_id is None:
            return self._ontology_nodes.copy()
        return [node for node in self._ontology_nodes if node["ontology_id"] == ontology_id]

    def get_ontology_relations(self, ontology_id: int | None = None) -> list[dict[str, Any]]:
        """Get all ontology relations, optionally filtered by ontology."""
        if ontology_id is None:
            return self._ontology_relations.copy()
        return [rel for rel in self._ontology_relations if rel["ontology_id"] == ontology_id]

    def get_database_metadata(self, ontology_node_id: int | None = None) -> list[dict[str, Any]]:
        """Get database metadata for ontology nodes."""
        metadata = [
            {
                "id": 1,
                "ontology_id": 1,
                "ontology_node_id": 1,  # Customer node
                "type": "Table",
                "sub_type": "Master",
                "db_table": "customers",
                "table_primary_key": "id",
                "formula": "",
                "description": "Customer master data",
                "allowed_aggregations": {"count": True, "distinct": True},
                "columns": [
                    {"name": "id", "type": "INTEGER", "nullable": False},
                    {"name": "customer_name", "type": "VARCHAR(255)", "nullable": False},
                    {"name": "email", "type": "VARCHAR(255)", "nullable": True},
                ],
            },
            {
                "id": 2,
                "ontology_id": 1,
                "ontology_node_id": 2,  # Order node
                "type": "Table",
                "sub_type": "Fact",
                "db_table": "orders",
                "table_primary_key": "id",
                "formula": "",
                "description": "Order transaction data",
                "allowed_aggregations": {"count": True, "sum": True, "avg": True},
                "columns": [
                    {"name": "id", "type": "INTEGER", "nullable": False},
                    {"name": "customer_id", "type": "INTEGER", "nullable": False},
                    {"name": "order_date", "type": "DATE", "nullable": False},
                    {"name": "order_value", "type": "DECIMAL(10,2)", "nullable": False},
                ],
            },
        ]

        if ontology_node_id is not None:
            return [m for m in metadata if m["ontology_node_id"] == ontology_node_id]
        return metadata

    def get_context(self, query: str = "") -> list[dict[str, Any]]:
        """Get ontology context for a user query."""
        if not query:
            return []

        # Simple context matching based on query keywords
        query_lower = str(query).lower()
        relevant_context = []

        # Check if query mentions any of our entities
        for node in self._ontology_nodes:
            node_name = str(node.get("name", "")).lower()
            if node_name in query_lower or query_lower in node_name:
                relevant_context.append(
                    {
                        "id": node.get("id"),
                        "name": node.get("name"),
                        "type": "ontology_node",
                        "relevance": 0.8,
                    }
                )

        return relevant_context
