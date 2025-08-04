"""Ontology port interface for accessing business ontology data."""

from typing import Any, Protocol
from uuid import UUID


class OntologyPort(Protocol):
    """Port interface for accessing business ontology data.

    This port abstracts access to ontology storage systems, allowing the analytics engine to work
    with different ontology backends.
    """

    def get_ontologies(self) -> list[dict[str, Any]]:
        """Get all ontologies.

        Returns
        -------
            List of ontology dictionaries with structure:
            {
                "id": UUID,
                "name": str,
                "is_active": bool,
                "created_at": datetime,
                "tenant_id": UUID,
                "last_synced_at": datetime,
                "nodes": list[dict],
                "relations": list[dict]
            }
        """
        ...

    def get_ontology_nodes(self, ontology_id: UUID | None = None) -> list[dict[str, Any]]:
        """Get all ontology nodes, optionally filtered by ontology.

        Returns
        -------
            List of node dictionaries with structure:
            {
                "id": UUID,
                "ontology_id": UUID,
                "name": str,
                "alias": str,
                "permissed_roles": list[str],
                "last_synced_at": datetime,
                "database_metadata": dict
            }
        """
        ...

    def get_ontology_relations(self, ontology_id: UUID | None = None) -> list[dict[str, Any]]:
        """Get all ontology relations, optionally filtered by ontology.

        Returns
        -------
            List of relation dictionaries with structure:
            {
                "id": UUID,
                "ontology_id": UUID,
                "parent_id": UUID,
                "child_id": UUID,
                "description": str,
                "last_synced_at": datetime,
                "join_metadata": dict
            }
        """
        ...

    def get_database_metadata(self, ontology_node_id: UUID | None = None) -> list[dict[str, Any]]:
        """Get database metadata for ontology nodes.

        Returns
        -------
            List of metadata dictionaries with structure:
            {
                "id": UUID,
                "ontology_id": UUID,
                "ontology_node_id": UUID,
                "type": str,  # Table, PreComputedView, DependantView
                "sub_type": str,  # Master, Fact, Function
                "db_table": str,
                "table_primary_key": str,
                "formula": str,
                "description": str,
                "allowed_aggregations": dict,
                "columns": list[dict]
            }
        """
        ...

    def get_context(self, query: str = "") -> list[dict[str, Any]]:
        """Get relevant context for a user query.

        Note: This is a placeholder for future embedding-based semantic matching.
        Current implementation uses simple string matching for testing purposes.

        Args
        ----
            query: User query to find relevant context for

        Returns
        -------
            List of context dictionaries with relevance scores:
            {
                "id": node_id,
                "name": node_name,
                "type": "ontology_node",
                "relevance": float (0.0-1.0)
            }
        """
        ...
