"""Enhanced database adapter with schema filtering and relationship handling."""

from typing import Any

from hexai.core.application.services.schema_filtering_service import SchemaFilteringService
from hexai.core.domain.services.relationship_service import RelationshipService
from hexai.core.ports.database import DatabasePort


class EnhancedDatabaseAdapter(DatabasePort):
    """Enhanced database adapter with advanced schema filtering capabilities."""

    def __init__(
        self,
        base_adapter: DatabasePort,
        schema_filtering_service: SchemaFilteringService | None = None,
        relationship_service: RelationshipService | None = None,
    ) -> None:
        """Initialize with a base database adapter and services.

        Args
        ----
        base_adapter: Base database port implementation
        schema_filtering_service: Service for schema filtering logic
        relationship_service: Service for relationship operations
        """
        self._base = base_adapter
        self._relationship_service = relationship_service or RelationshipService()
        self._schema_filtering_service = schema_filtering_service or SchemaFilteringService(
            self._relationship_service
        )

    # Delegate basic methods to base adapter
    def get_table_schemas(self) -> dict[str, dict[str, Any]]:
        """Get schema information for all tables."""
        return self._base.get_table_schemas()

    def get_relationships(self) -> list[dict[str, Any]]:
        """Get foreign key relationships between tables."""
        return self._base.get_relationships()

    def get_indexes(self) -> list[dict[str, Any]]:
        """Get index information."""
        return self._base.get_indexes()

    def get_table_statistics(self) -> dict[str, dict[str, Any]]:
        """Get table statistics."""
        return self._base.get_table_statistics()

    # Enhanced methods for schema filtering
    def filter_schema_by_ontology(self, ontology_nodes: list[Any]) -> list[str]:
        """Filter schema based on ontology nodes.

        Args
        ----
        ontology_nodes: List of ontology nodes with database metadata

        Returns
        -------
        List of relevant table names including intermediate tables
        """
        schemas = self.get_table_schemas()
        relationships = self.get_relationships()
        return self._schema_filtering_service.filter_by_ontology(
            schemas, ontology_nodes, relationships
        )

    def filter_schema_by_context(self, ontology_context: list[dict[str, Any]]) -> list[str]:
        """Filter schema based on ontology context.

        Args
        ----
        ontology_context: List of context items with db_table information

        Returns
        -------
        List of relevant table names
        """
        schemas = self.get_table_schemas()
        return self._schema_filtering_service.filter_by_context(schemas, ontology_context)

    def filter_schema_by_query(self, query: str) -> list[str]:
        """Filter schema based on query keywords.

        Args
        ----
        query: User query string

        Returns
        -------
        List of relevant table names
        """
        schemas = self.get_table_schemas()
        return self._schema_filtering_service.filter_by_query(schemas, query)

    def get_tables_with_relationships(self, core_tables: list[str]) -> list[str]:
        """Get core tables plus all intermediate tables that connect them.

        Args
        ----
        core_tables: List of core table names

        Returns
        -------
        List of all tables including intermediates
        """
        relationships = self.get_relationships()
        return self._relationship_service.get_tables_with_relationships(core_tables, relationships)

    def build_relationship_graph(self) -> dict[str, set[str]]:
        """Build bidirectional relationship graph from foreign keys.

        Args
        ----
        relationships: List of relationship data from database

        Returns
        -------
        Graph where keys are table names and values are sets of connected tables
        """
        relationships = self.get_relationships()
        return self._relationship_service.build_relationship_graph(relationships)

    def find_path_tables(self, start_table: str, end_table: str) -> set[str]:
        """Find all tables in the shortest path between two tables.

        Args
        ----
        start_table: Starting table name
        end_table: Ending table name

        Returns
        -------
        Set of table names in the path (including start and end)
        """
        graph = self.build_relationship_graph()
        return self._relationship_service.find_path_tables(start_table, end_table, graph)

    def get_table_ddls(self, table_names: list[str]) -> str:
        """Get DDL statements for specified tables.

        Args
        ----
        table_names: List of table names

        Returns
        -------
        Formatted DDL string
        """
        schemas = self.get_table_schemas()
        ddls = []

        for table in table_names:
            if table in schemas:
                schema = schemas[table]
                columns = schema.get("columns", {})
                col_defs = [f"{col} {dtype}" for col, dtype in columns.items()]
                ddls.append(f"CREATE TABLE {table} ({', '.join(col_defs)});")

        return "\n".join(ddls)

    def get_sample_data(self, table_names: list[str], rows_per_table: int = 3) -> str:
        """Get sample data for specified tables.

        Args
        ----
        table_names: List of table names
        rows_per_table: Number of sample rows per table

        Returns
        -------
        Formatted sample data string
        """
        # In a real implementation, this would query actual data
        # For now, return empty string as placeholder
        return ""

    def get_foreign_keys_for_tables(self, table_names: list[str]) -> str:
        """Get foreign key definitions for specified tables.

        Args
        ----
        table_names: List of table names

        Returns
        -------
        Formatted foreign key string
        """
        relationships = self.get_relationships()
        relevant_fks = []

        for rel in relationships:
            if isinstance(rel, dict):
                from_table = rel.get("from_table", "")
                to_table = rel.get("to_table", "")
                if from_table in table_names and to_table in table_names:
                    from_col = rel.get("from_column", "")
                    to_col = rel.get("to_column", "")
                    relevant_fks.append(
                        f"ALTER TABLE {from_table} ADD FOREIGN KEY ({from_col}) "
                        f"REFERENCES {to_table}({to_col});"
                    )

        return "\n".join(relevant_fks)
