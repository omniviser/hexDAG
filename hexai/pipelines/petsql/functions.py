"""PetSQL pipeline functions - Pydantic-first implementation."""

import re
from typing import Any

from hexai.adapters.enhanced_database import EnhancedDatabaseAdapter
from pydantic import BaseModel
import sqlglot
from sqlglot import exp


# Pydantic Models for PetSQL Pipeline
class SelectExamplesInput(BaseModel):
    """Input model for select_examples function."""

    user_query: str
    database_schema: dict[str, Any] | None = None


class SelectExamplesOutput(BaseModel):
    """Output model for select_examples function."""

    examples: list[dict[str, Any]]
    user_query: str
    database_schema: dict[str, Any] | None = None
    metadata: dict[str, Any]


class FilterSchemaInput(BaseModel):
    """Input model for filter_relevant_schema function."""

    user_query: str
    examples: list[dict[str, Any]] | None = None
    ontology_context: list[dict[str, Any]] | None = None
    ontology_nodes: list[dict[str, Any]] | None = None


class FilterSchemaOutput(BaseModel):
    """Output model for filter_relevant_schema function."""

    filtered_schema: dict[str, Any]
    user_query: str
    examples: list[dict[str, Any]] | None = None
    metadata: dict[str, Any]


class SqlValidatorInput(BaseModel):
    """Input model for sql_validator function."""

    sql_query: str
    database_schema: dict[str, Any] | None = None


class SqlValidatorOutput(BaseModel):
    """Output model for sql_validator function."""

    validated_sql: str
    table_names: list[str]
    metadata: dict[str, Any]


class SqlConverterInput(BaseModel):
    """Input model for convert_sqlite_to_postgres function."""

    sql_query: str
    table_names: list[str] | None = None


class SqlConverterOutput(BaseModel):
    """Output model for convert_sqlite_to_postgres function."""

    converted_sql: str
    metadata: dict[str, Any]


def _extract_sql_content(data: dict[str, Any]) -> str:
    """Extract SQL content from various data formats."""
    if "output" in data:
        output = data["output"]
        return output.get("content", "") if isinstance(output, dict) else str(output)
    return data.get("sql", data.get("presql", data.get("final_sql", "")))


async def select_examples(input_data: SelectExamplesInput, **ports: Any) -> SelectExamplesOutput:
    """Select relevant examples using embedding similarity."""
    event_manager = ports.get("event_manager")
    embedding_selector = ports.get("embedding_selector")

    if not embedding_selector:
        if event_manager:
            await event_manager.add_trace(
                "select_examples", "No embedding selector available, returning empty examples"
            )
        return SelectExamplesOutput(
            examples=[],
            user_query=input_data.user_query,
            database_schema=input_data.database_schema,
            metadata={"node_name": "select_examples"},
        )

    examples_tuples = embedding_selector.get_examples({"question": input_data.user_query})
    examples = [
        example[0] for example in examples_tuples
    ]  # Extract just the examples, not the schemas

    if event_manager:
        await event_manager.add_trace("select_examples", f"Selected {len(examples)} examples")

    return SelectExamplesOutput(
        examples=examples,
        user_query=input_data.user_query,
        database_schema=input_data.database_schema,
        metadata={"node_name": "select_examples"},
    )


async def filter_relevant_schema(input_data: FilterSchemaInput, **ports: Any) -> FilterSchemaOutput:
    """Filter database schema using ontology nodes and relationships."""
    event_manager = ports.get("event_manager")
    database = ports.get("database")

    if not database:
        if event_manager:
            await event_manager.add_trace(
                "filter_relevant_schema", "No database adapter available, returning empty schema"
            )
        return FilterSchemaOutput(
            filtered_schema={},
            user_query=input_data.user_query,
            examples=input_data.examples,
            metadata={"node_name": "filter_relevant_schema"},
        )

    # Wrap with enhanced adapter if needed
    if not hasattr(database, "filter_schema_by_ontology"):
        database = EnhancedDatabaseAdapter(database)

    # Use adapter methods for filtering
    ontology_nodes = input_data.ontology_nodes or []
    ontology_context = input_data.ontology_context or []

    if ontology_nodes:
        table_names = database.filter_schema_by_ontology(ontology_nodes)

    elif ontology_context:
        table_names = database.filter_schema_by_context(ontology_context)

    else:
        table_names = database.filter_schema_by_query(input_data.user_query)

    # Get DDLs from adapter
    ddls = database.get_table_ddls(table_names)

    # Prepare table objects for template
    relevant_tables = [{"table_name": name, "columns": ["*"]} for name in table_names]

    # Provide singular variables for template compatibility
    table = relevant_tables[0] if relevant_tables else {"table_name": "", "columns": []}
    col = table.get("columns", [""])[0] if table.get("columns") else ""

    if event_manager:
        await event_manager.set_memory("relevant_tables", table_names)
        await event_manager.add_trace(
            "filter_relevant_schema", f"Filtered schema to {len(table_names)} relevant tables"
        )

    return FilterSchemaOutput(
        filtered_schema={
            "table_names": table_names,
            "ddls": ddls,
            "relevant_tables": relevant_tables,
            "table": table,
            "col": col,
            "ontology_nodes": ontology_nodes,
        },
        user_query=input_data.user_query,
        examples=input_data.examples,
        metadata={"node_name": "filter_relevant_schema"},
    )


async def sql_validator(input_data: SqlValidatorInput, **ports: Any) -> SqlValidatorOutput:
    """Validate SQL query and extract table names."""
    event_manager = ports.get("event_manager")

    sql_content = input_data.sql_query

    if not sql_content:
        if event_manager:
            await event_manager.add_trace("sql_validator", "No SQL content found to validate")
        return SqlValidatorOutput(
            validated_sql=sql_content, table_names=[], metadata={"node_name": "sql_validator"}
        )

    try:
        # Parse SQL using sqlglot
        parsed = sqlglot.parse(sql_content)
        if not parsed:
            raise ValueError("Failed to parse SQL")

        # Extract table names
        table_names = _extract_table_names_from_sql(sql_content)

        if event_manager:
            await event_manager.add_trace(
                "sql_validator", f"SQL validated successfully, found {len(table_names)} tables"
            )

        return SqlValidatorOutput(
            validated_sql=sql_content,
            table_names=table_names,
            metadata={"node_name": "sql_validator"},
        )

    except Exception as e:
        if event_manager:
            await event_manager.add_trace("sql_validator", f"SQL validation failed: {str(e)}")

        return SqlValidatorOutput(
            validated_sql=sql_content, table_names=[], metadata={"node_name": "sql_validator"}
        )


def _extract_table_names_from_sql(sql_content: str) -> list[str]:
    """Extract table names from SQL content using sqlglot."""
    try:
        parsed = sqlglot.parse(sql_content)
        if not parsed:
            return []

        table_names = set()

        # Extract table names from parsed SQL
        for expression in parsed:
            if expression and hasattr(expression, "find_all"):
                # Find all table references
                for table_ref in expression.find_all(exp.Table):
                    table_names.add(table_ref.name)
                # Also check for CTEs
                for cte in expression.find_all(exp.CTE):
                    table_names.add(cte.alias)

        return list(table_names)
    except Exception:
        # Fallback to regex if sqlglot fails
        return _extract_table_names_regex(sql_content)


def _extract_table_names_regex(sql_content: str) -> list[str]:
    """Fallback method to extract table names using regex."""
    # Simple regex patterns for common SQL table references
    patterns = [
        r"FROM\s+(\w+)",
        r"JOIN\s+(\w+)",
        r"UPDATE\s+(\w+)",
        r"INSERT\s+INTO\s+(\w+)",
        r"DELETE\s+FROM\s+(\w+)",
    ]

    table_names = set()
    for pattern in patterns:
        matches = re.findall(pattern, sql_content, re.IGNORECASE)
        table_names.update(matches)

    return list(table_names)


async def convert_sqlite_to_postgres(
    input_data: SqlConverterInput, **ports: Any
) -> SqlConverterOutput:
    """Convert SQLite SQL to PostgreSQL format."""
    event_manager = ports.get("event_manager")

    sql_content = input_data.sql_query

    if not sql_content:
        return SqlConverterOutput(
            converted_sql=sql_content, metadata={"node_name": "convert_sqlite_to_postgres"}
        )

    try:
        # Parse and convert using sqlglot
        parsed = sqlglot.parse(sql_content, read="sqlite")
        if not parsed:
            raise ValueError("Failed to parse SQLite SQL")

        # Convert to PostgreSQL
        converted = sqlglot.transpile(sql_content, read="sqlite", write="postgres")
        converted_sql = converted[0] if converted else sql_content

        # Additional manual conversions for common SQLite to PostgreSQL differences
        # Convert double quotes to single quotes for string literals
        # Pattern to match string literals with double quotes
        # This regex looks for double quotes that are not part of identifier quotes
        converted_sql = re.sub(r'(?<!\w)"([^"]*)"(?=\s|$|[,;])', r"'\1'", converted_sql)

        # Handle LIKE patterns - SQLite uses % for wildcards, PostgreSQL uses same
        # But we need to ensure proper escaping
        converted_sql = re.sub(r'LIKE\s+"([^"]*)"', r"LIKE '\1'", converted_sql)

        if event_manager:
            await event_manager.add_trace(
                "convert_sqlite_to_postgres", "SQL converted successfully"
            )

        return SqlConverterOutput(
            converted_sql=converted_sql, metadata={"node_name": "convert_sqlite_to_postgres"}
        )

    except Exception as e:
        if event_manager:
            await event_manager.add_trace(
                "convert_sqlite_to_postgres", f"SQL conversion failed: {str(e)}"
            )

        return SqlConverterOutput(
            converted_sql=sql_content,  # Return original as fallback
            metadata={"node_name": "convert_sqlite_to_postgres"},
        )
