"""Schema filtering service for database schema operations.

This service contains the business logic for filtering database schemas based on ontology, context,
and query criteria. It operates on data provided by database ports without containing infrastructure
concerns.
"""

from typing import Any

from hexai.app.domain.services.relationship_service import RelationshipService


class SchemaFilteringService:
    """Service for filtering database schemas based on various criteria."""

    def __init__(self, relationship_service: RelationshipService) -> None:
        """Initialize the schema filtering service.

        Args
        ----
        relationship_service: Service for handling table relationships
        """
        self._relationship_service = relationship_service

    def filter_by_ontology(
        self,
        schemas: dict[str, dict[str, Any]],
        ontology_nodes: list[Any],
        relationships: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Filter schema based on ontology nodes.

        Args
        ----
        schemas: Dictionary of table schemas
        ontology_nodes: List of ontology nodes with database metadata
        relationships: Optional list of relationships for finding intermediate tables

        Returns
        -------
        List of relevant table names including intermediate tables
        """
        all_tables = list(schemas.keys())
        core_tables = []

        # Extract core tables from ontology nodes
        for node in ontology_nodes:
            node_name = node if isinstance(node, str) else node.get("name", "")

            # Direct table matching
            for table in all_tables:
                if self._matches_table_name(table, node_name):
                    if table not in core_tables:
                        core_tables.append(table)

            # Database metadata matching
            if isinstance(node, dict) and "database_metadata" in node:
                db_table = node["database_metadata"].get("db_table", "")
                if db_table in all_tables and db_table not in core_tables:
                    core_tables.append(db_table)

        # Expand with intermediate tables if multiple core tables found
        if len(core_tables) > 1 and relationships is not None:
            return self._relationship_service.get_tables_with_relationships(
                core_tables, relationships
            )

        return core_tables

    def filter_by_context(
        self,
        schemas: dict[str, dict[str, Any]],
        ontology_context: list[dict[str, Any]],
    ) -> list[str]:
        """Filter schema based on ontology context.

        Args
        ----
        schemas: Dictionary of table schemas
        ontology_context: List of context items with db_table information

        Returns
        -------
        List of relevant table names
        """
        all_tables = list(schemas.keys())
        relevant_tables = []

        for item in ontology_context:
            if isinstance(item, dict):
                db_table = item.get("db_table", "")
                if db_table in all_tables and db_table not in relevant_tables:
                    relevant_tables.append(db_table)

        return relevant_tables if relevant_tables else all_tables

    def filter_by_query(
        self,
        schemas: dict[str, dict[str, Any]],
        query: str,
    ) -> list[str]:
        """Filter schema based on query keywords.

        Args
        ----
        schemas: Dictionary of table schemas
        query: User query string

        Returns
        -------
        List of relevant table names
        """
        if not query:
            return list(schemas.keys())

        query_lower = query.lower()
        relevant_tables = []

        for table, schema in schemas.items():
            # Check table name
            if table.lower() in query_lower:
                relevant_tables.append(table)
            else:
                # Check column names
                columns = schema.get("columns", {})
                if any(col.lower() in query_lower for col in columns.keys()):
                    relevant_tables.append(table)

        return relevant_tables if relevant_tables else list(schemas.keys())

    def _matches_table_name(self, table: str, node_name: str) -> bool:
        """Check if a table name matches a node name using fuzzy matching.

        Args
        ----
        table: Table name from database
        node_name: Node name from ontology

        Returns
        -------
        True if the names match, False otherwise
        """
        return (
            table.lower() == node_name.lower()
            or node_name.lower() in table.lower()
            or table.lower() in node_name.lower()
        )
