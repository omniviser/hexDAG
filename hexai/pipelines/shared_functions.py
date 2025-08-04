"""Shared functions for pipeline implementations - pure orchestration."""

import logging
from typing import Any

from hexai.adapters.enhanced_database import EnhancedDatabaseAdapter
from hexai.adapters.mock import MockDatabaseAdapter
from pydantic import BaseModel, Field


# Input Models
class UserQueryInput(BaseModel):
    """Input model for user query extraction."""

    user_query: str = Field(..., description="User query or question")
    question: str | None = Field(None, description="Alternative field for user query")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DatabaseSchemaInput(BaseModel):
    """Input model for database schema loading."""

    user_query: str = Field(..., description="User query for context")
    question: str | None = Field(None, description="Alternative field for user query")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# Output Models
class UserQueryOutput(BaseModel):
    """Output model for user query extraction."""

    user_query: str = Field(..., description="Extracted user query")
    original_input: dict[str, Any] = Field(..., description="Original input data")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Processing metadata")


class TableSchema(BaseModel):
    """Model for individual table schema."""

    table_name: str = Field(..., description="Name of the table")
    name: str = Field(..., description="Alias for backward compatibility")
    columns: list[str] = Field(default_factory=list, description="List of column names")
    column_types: dict[str, Any] = Field(default_factory=dict, description="Column type mappings")
    primary_keys: list[str] = Field(default_factory=list, description="Primary key columns")
    foreign_keys: list[str] = Field(default_factory=list, description="Foreign key columns")


class Relationship(BaseModel):
    """Model for database relationships."""

    from_table: str = Field(..., description="Source table name")
    from_column: str = Field(..., description="Source column name")
    to_table: str = Field(..., description="Target table name")
    to_column: str = Field(..., description="Target column name")


class DatabaseSchemaOutput(BaseModel):
    """Output model for database schema loading."""

    table_schemas: list[TableSchema] = Field(..., description="List of table schemas")
    tables: dict[str, Any] | list[Any] = Field(..., description="Raw table schemas")
    relationships: list[Relationship] = Field(..., description="Database relationships")
    foreign_keys: list[str] = Field(default_factory=list, description="Relationship strings")
    indexes: dict[str, Any] | list[Any] = Field(
        default_factory=dict, description="Database indexes"
    )
    statistics: dict[str, Any] = Field(default_factory=dict, description="Table statistics")
    summary: str = Field(..., description="Schema summary")
    table_count: int = Field(..., description="Number of tables")
    relationship_count: int = Field(..., description="Number of relationships")
    error: str | None = Field(None, description="Error message if any")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Processing metadata")


logger = logging.getLogger(__name__)


def get_input_data(input_data: dict[str, Any] | str | BaseModel) -> dict[str, Any]:
    """Extract data dictionary from various input formats."""
    if hasattr(input_data, "model_dump"):
        return input_data.model_dump()  # type: ignore[attr-defined]
    return input_data if isinstance(input_data, dict) else {"user_query": str(input_data)}


def _format_table_schema(table_name: str, schema: dict[str, Any]) -> TableSchema:
    """Format a single table schema to standard format."""
    # Handle foreign_keys - convert from dict format to string format if needed
    foreign_keys = schema.get("foreign_keys", [])
    formatted_foreign_keys = []

    for fk in foreign_keys:
        if isinstance(fk, dict):
            # Convert dict format to string format
            column = fk.get("column", "")
            ref_table = fk.get("references_table", "")
            ref_column = fk.get("references_column", "")
            if all([column, ref_table, ref_column]):
                formatted_foreign_keys.append(f"{column} -> {ref_table}.{ref_column}")
        else:
            # Already in string format
            formatted_foreign_keys.append(str(fk))

    return TableSchema(
        table_name=table_name,
        name=table_name,  # Alias for backward compatibility
        columns=list(schema.get("columns", {}).keys()),
        column_types=schema.get("columns", {}),
        primary_keys=schema.get("primary_keys", []),
        foreign_keys=formatted_foreign_keys,
    )


def _process_schema_list(schema: dict[str, Any]) -> TableSchema:
    """Process individual schema from list format."""
    table_name = schema.get("table_name", schema.get("name", ""))
    name = schema.get("name", schema.get("table_name", ""))
    return TableSchema(
        table_name=table_name,
        name=name,
        columns=schema.get("columns", []),
        column_types=schema.get("column_types", {}),
        primary_keys=schema.get("primary_keys", []),
        foreign_keys=schema.get("foreign_keys", []),
    )


def _format_relationships(relationships: list[Any]) -> tuple[list[Relationship], list[str]]:
    """Format relationships into structured and string formats."""
    formatted_relationships = []
    relationship_strings = []

    for rel in relationships:
        if isinstance(rel, dict):
            # Structured relationship format
            formatted_relationships.append(
                Relationship(
                    from_table=rel.get("from_table", ""),
                    from_column=rel.get("from_column", ""),
                    to_table=rel.get("to_table", ""),
                    to_column=rel.get("to_column", ""),
                )
            )
            # String format for backward compatibility
            from_table = rel.get("from_table", "")
            from_col = rel.get("from_column", "")
            to_table = rel.get("to_table", "")
            to_col = rel.get("to_column", "")
            if all([from_table, from_col, to_table, to_col]):
                relationship_strings.append(f"{from_table}.{from_col} -> {to_table}.{to_col}")
        else:
            # Handle string format relationships
            relationship_strings.append(str(rel))

    return formatted_relationships, relationship_strings


async def extract_user_query(
    input_data: UserQueryInput,
    **ports: Any,
) -> UserQueryOutput:
    """Extract and validate user query from input data using Pydantic.

    Args
    ----
        input_data: UserQueryInput with query data
        **ports: Injected ports

    Returns
    -------
        UserQueryOutput with extracted query and metadata
    """
    event_manager = ports.get("event_manager")

    # Use Pydantic model directly
    user_query = input_data.user_query or input_data.question or ""

    if not user_query:
        if event_manager:
            event_manager.add_trace("extract_user_query", "No user query found")
    else:
        if event_manager:
            event_manager.add_trace(
                "extract_user_query", f"Extracting user query: {user_query[:100]}..."
            )

    return UserQueryOutput(
        user_query=user_query,
        original_input=input_data.model_dump(),
        metadata={"node_name": "extract_user_query"},
    )


async def load_database_schema(
    input_data: DatabaseSchemaInput,
    **ports: Any,
) -> DatabaseSchemaOutput:
    """Load database schema using database port with enhanced capabilities.

    Args
    ----
        input_data: DatabaseSchemaInput with query context
        **ports: Injected ports including 'database'

    Returns
    -------
        DatabaseSchemaOutput containing schema information in multiple formats
    """
    event_manager = ports.get("event_manager")

    # Get database port, defaulting to MockDatabaseAdapter for development/testing
    database = ports.get("database")
    if not database:
        if event_manager:
            event_manager.add_trace(
                "load_database_schema", "No database port provided, using MockDatabaseAdapter"
            )
        database = MockDatabaseAdapter()

    # Wrap with enhanced adapter for advanced filtering capabilities
    if not hasattr(database, "filter_schema_by_ontology"):
        database = EnhancedDatabaseAdapter(database)
        if event_manager:
            event_manager.add_trace("load_database_schema", "Enhanced database adapter applied")

    try:
        # Get core schema information
        table_schemas = database.get_table_schemas()
        relationships = database.get_relationships()

        # Get additional metadata for query optimization
        indexes = database.get_indexes()
        statistics = database.get_table_statistics()

        # Convert schemas to list format for pipeline compatibility
        schema_list = []

        # Handle both dict and list formats from different database adapters
        if isinstance(table_schemas, dict):
            # Dict format: table_name -> schema_dict
            for table_name, schema in table_schemas.items():
                schema_list.append(_format_table_schema(table_name, schema))
        elif isinstance(table_schemas, list):
            # List format: already processed schemas
            for schema in table_schemas:
                if isinstance(schema, dict):
                    # Ensure consistent format
                    schema_list.append(_process_schema_list(schema))
        else:
            # Fallback for unexpected formats
            schema_list = []

        # Format relationships for different pipeline needs
        formatted_relationships, relationship_strings = _format_relationships(relationships)

        # Store in event manager for downstream functions
        if event_manager:
            event_manager.set_memory("table_schemas", table_schemas)
            event_manager.set_memory("relationships", formatted_relationships)

        schema_count = len(schema_list)
        relationship_count = len(formatted_relationships)

        if event_manager:
            event_manager.add_trace(
                "load_database_schema",
                f"Loaded {schema_count} tables, {relationship_count} relationships, "
                f"{len(indexes)} indexes, statistics for {len(statistics)} tables",
            )

        return DatabaseSchemaOutput(
            table_schemas=schema_list,
            tables=table_schemas,
            relationships=formatted_relationships,
            foreign_keys=relationship_strings,
            indexes=indexes,
            statistics=statistics,
            summary=(
                f"Database schema with {schema_count} tables and "
                f"{relationship_count} relationships"
            ),
            table_count=schema_count,
            relationship_count=relationship_count,
            error=None,
            metadata={"node_name": "load_database_schema"},
        )

    except Exception as e:
        error_msg = f"Failed to load database schema: {str(e)}"
        if event_manager:
            event_manager.add_trace("load_database_schema", error_msg)
        logger.error(error_msg, exc_info=True)

        # Return minimal schema on error to prevent pipeline failure
        return DatabaseSchemaOutput(
            table_schemas=[],
            tables={},
            relationships=[],
            foreign_keys=[],
            indexes=[],
            statistics={},
            summary="Failed to load database schema",
            table_count=0,
            relationship_count=0,
            error=error_msg,
            metadata={"node_name": "load_database_schema", "error": error_msg},
        )
