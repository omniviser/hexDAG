"""Text-to-SQL Pipeline Tool Functions and Router.

This module provides tool functions and a ToolRouter implementation for the Text-to-SQL pipeline.
Tool functions use the DatabasePort interface for database access, allowing for proper dependency
injection and testing.
"""

from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable

from hexai.adapters.mock import MockDatabaseAdapter
from hexai.app.ports.database import DatabasePort


class Text2SQLToolRouter:
    """Tool router for Text-to-SQL pipeline functions.

    This router implements proper dependency injection using the DatabasePort interface. Database
    adapters can be mocked for testing, but the ToolRouter itself is real.
    """

    def __init__(self, database_port: DatabasePort | None = None) -> None:
        """Initialize the tool router with database port.

        Args
        ----
            database_port: Database port implementation.
        """
        self.database_port = database_port or MockDatabaseAdapter()

        # Register tool functions with their dependencies
        self.tools: dict[str, Callable[..., Awaitable[Any]]] = {
            "get_schema": self._get_schema,
            "get_column_enum_values": self._get_column_enum_values,
            "get_join_path": self._get_join_path,
            "resolve_date_range": self._resolve_date_range,
            "preview_data": self._preview_data,
            "end": self._end_with_result,
            "tool_end": self._end_with_result,  # Alias for consistency
            "change_phase": self._change_phase,
            "phase": self._change_phase,  # Alias
        }

    async def aroute(self, tool_name: str, input_data: dict[str, Any] | None = None) -> Any:
        """Route a tool call to the appropriate function.

        Args
        ----
            tool_name: Name of the tool to call
            input_data: Parameters for the tool (optional)

        Returns
        -------
            Result from the tool function
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found. Available: {list(self.tools.keys())}")

        tool_function = self.tools[tool_name]

        try:
            # Call the function with parameters if provided
            if input_data:
                result = await tool_function(**input_data)
            else:
                result = await tool_function()

            return result

        except Exception as e:
            # Re-raise the exception instead of returning error dict
            raise e

    async def call_tool(self, tool_name: str, params: dict[str, Any]) -> Any:
        """Call a tool with parameters (compatibility method for agent nodes)."""
        return await self.aroute(tool_name, params)

    def get_available_tools(self) -> list[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())

    def get_tool_schema(self, tool_name: str) -> dict[str, Any]:
        """Get schema for a specific tool."""
        if tool_name not in self.tools:
            return {}

        # Return basic schema for the tool
        return {
            "name": tool_name,
            "description": f"Text2SQL tool: {tool_name}",
            "parameters": [],  # TODO: Could extract from function signature
            "examples": [f"{tool_name}()"],
        }

    def get_all_tool_schemas(self) -> dict[str, dict[str, Any]]:
        """Get schemas for all available tools."""
        return {name: self.get_tool_schema(name) for name in self.tools.keys()}

    # Tool function implementations using DatabasePort

    async def _get_schema(self) -> dict[str, Any]:
        """Get database schema information from the database port.

        Returns
        -------
            Dictionary containing table information with columns and types.
        """
        table_schemas = self.database_port.get_table_schemas()

        # Transform schema data to a more user-friendly format
        tables = {}
        for table_name, schema in table_schemas.items():
            # Extract column names from schema
            column_names = list(schema["columns"].keys())

            # Find foreign keys for this table
            foreign_keys = {}
            for fk in schema.get("foreign_keys", []):
                fk_column = fk["column"]
                ref_table = fk["references_table"]
                ref_column = fk["references_column"]
                foreign_keys[fk_column] = f"{ref_table}.{ref_column}"

            tables[table_name] = {
                "columns": column_names,
                "primary_key": (
                    schema.get("primary_keys", ["id"])[0] if schema.get("primary_keys") else "id"
                ),
                "foreign_keys": foreign_keys,
                "description": f"{table_name.title()} table",
            }

        return {"tables": tables}

    async def _get_column_enum_values(self, table: str, column: str) -> list[str]:
        """Get possible enum values for a specific column.

        Args
        ----
            table: Table name
            column: Column name

        Returns
        -------
            List of possible values for the column.
        """
        # For demo purposes, provide common enum values based on column names
        # In a real implementation, this would query the database for actual enum values
        enum_mappings = {
            ("orders", "status"): [
                "pending",
                "processing",
                "shipped",
                "delivered",
                "cancelled",
                "returned",
            ],
            ("customers", "status"): ["active", "inactive", "suspended"],
            ("customers", "segment"): ["enterprise", "small_business", "individual"],
            ("products", "category"): ["electronics", "clothing", "books", "home", "sports"],
        }

        return enum_mappings.get((table, column), [])

    async def _get_join_path(self, table1: str, table2: str) -> dict[str, Any]:
        """Get join information between two tables using database relationships.

        Args
        ----
            table1: First table name
            table2: Second table name

        Returns
        -------
            Dictionary with join path information.
        """
        relationships = self.database_port.get_relationships()

        # Find direct relationship between tables
        for rel in relationships:
            if (rel["from_table"] == table1 and rel["to_table"] == table2) or (
                rel["from_table"] == table2 and rel["to_table"] == table1
            ):

                if rel["from_table"] == table1:
                    join_condition = f"{table1}.{rel['from_column']} = {table2}.{rel['to_column']}"
                else:
                    join_condition = f"{table2}.{rel['from_column']} = {table1}.{rel['to_column']}"

                return {
                    "join_condition": join_condition,
                    "relationship": rel["relationship_type"],
                    "description": f"Direct relationship between {table1} and {table2}",
                }

        # No direct relationship found
        return {
            "join_condition": "No direct relationship found",
            "relationship": "none",
            "description": f"No known relationship between {table1} and {table2}",
        }

    async def _resolve_date_range(self, phrase: str) -> dict[str, Any]:
        """Resolve natural language date phrases to specific date ranges.

        Args
        ----
            phrase: Natural language date phrase (e.g., "last 3 months")

        Returns
        -------
            Dictionary with start_date, end_date, and interpretation.
        """
        now = datetime.now()
        phrase_lower = phrase.lower()

        # Date range mappings
        if "last 3 months" in phrase_lower or "past 3 months" in phrase_lower:
            start_date = now - timedelta(days=90)
        elif "last month" in phrase_lower or "past month" in phrase_lower:
            start_date = now - timedelta(days=30)
        elif "last week" in phrase_lower or "past week" in phrase_lower:
            start_date = now - timedelta(days=7)
        elif "last year" in phrase_lower or "past year" in phrase_lower:
            start_date = now - timedelta(days=365)
        elif "today" in phrase_lower:
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif "yesterday" in phrase_lower:
            start_date = (now - timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        else:
            # Default to last 30 days for unknown phrases
            start_date = now - timedelta(days=30)

        return {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": now.strftime("%Y-%m-%d"),
            "phrase_interpreted": phrase,
            "sql_condition": f"date_column >= '{start_date.strftime('%Y-%m-%d')}'",
        }

    async def _preview_data(self, table: str, limit: int = 5) -> dict[str, Any]:
        """Preview sample data from a table.

        Args
        ----
            table: Table name to preview
            limit: Maximum number of rows to return

        Returns
        -------
            Dictionary with sample data and metadata.
        """
        # Get table schema to understand structure
        table_schemas = self.database_port.get_table_schemas()
        table_stats = self.database_port.get_table_statistics()

        if table not in table_schemas:
            return {
                "error": f"Table '{table}' not found",
                "available_tables": list(table_schemas.keys()),
            }

        # Mock sample data based on schema
        # In a real implementation, this would query the actual database
        sample_data = self._generate_sample_data(table, table_schemas[table], limit)

        return {
            "table": table,
            "sample_data": sample_data,
            "rows_shown": len(sample_data),
            "total_available": table_stats.get(table, {}).get("row_count", "unknown"),
            "columns": list(table_schemas[table]["columns"].keys()),
        }

    async def _end_with_result(
        self, sql_query: str, explanation: str, confidence: str
    ) -> dict[str, str]:
        """End the reasoning process with structured SQL result.

        This tool should be called when the agent has completed the Text-to-SQL task
        and wants to return the final structured output.

        Args
        ----
            sql_query: The generated SQL query
            explanation: Explanation of the query logic
            confidence: Confidence level (high/medium/low)

        Returns
        -------
            Structured result matching the pipeline output schema
        """
        return {"sql_query": sql_query, "explanation": explanation, "confidence": confidence}

    async def _change_phase(self, phase: str, **context: Any) -> dict[str, Any]:
        """Change the agent's reasoning phase.

        Args
        ----
            phase: The new phase name to transition to
            **context: Optional context data for the phase transition

        Returns
        -------
            Dictionary with phase change information
        """
        return {"action": "change_phase", "new_phase": phase, "context": context}

    def _generate_sample_data(
        self, table: str, schema: dict[str, Any], limit: int
    ) -> list[dict[str, Any]]:
        """Generate sample data based on table schema.

        Args
        ----
            table: Table name
            schema: Table schema information
            limit: Number of rows to generate

        Returns
        -------
            List of sample data rows
        """
        # Mock data generation based on table type
        if table == "customers":
            return [
                {
                    "id": i,
                    "customer_name": f"Customer {i}",
                    "email": f"customer{i}@example.com",
                    "segment": "enterprise" if i % 3 == 0 else "small_business",
                    "status": "active",
                }
                for i in range(1, limit + 1)
            ]
        elif table == "orders":
            return [
                {
                    "id": i + 100,
                    "customer_id": i,
                    "order_date": "2024-12-01",
                    "order_value": 100.0 + i * 10,
                    "status": "delivered" if i % 2 == 0 else "pending",
                }
                for i in range(1, limit + 1)
            ]
        elif table == "products":
            categories = ["electronics", "clothing", "books"]
            return [
                {
                    "id": i,
                    "product_name": f"Product {i}",
                    "category": categories[i % 3],
                    "price": 50.0 + i * 5,
                }
                for i in range(1, limit + 1)
            ]
        elif table == "order_items":
            return [
                {
                    "id": i,
                    "order_id": i + 100,
                    "product_id": i,
                    "quantity": i % 3 + 1,
                    "unit_price": 25.0 + i * 2,
                }
                for i in range(1, limit + 1)
            ]
        else:
            # Generic sample data for unknown tables
            columns = list(schema["columns"].keys())
            return [{col: f"sample_{col}_{i}" for col in columns} for i in range(1, limit + 1)]
