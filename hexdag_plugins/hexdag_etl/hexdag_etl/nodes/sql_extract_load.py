"""SQL extraction and loading nodes for database operations."""

from typing import Any, Literal

from hexdag.core.domain.dag import NodeSpec

from .base_node_factory import BaseNodeFactory

# Convention: SQL load modes for dropdown menus in Studio UI
SqlLoadMode = Literal["append", "replace", "truncate_insert", "merge"]


class SQLExtractNode(BaseNodeFactory):
    """Extract data from SQL databases.

    Placeholder implementation - to be completed with full SQLAlchemy integration.
    """

    # Studio UI metadata
    _hexdag_icon = "Database"
    _hexdag_color = "#3b82f6"  # blue-500

    def __call__(
        self,
        name: str,
        query: str,
        database: str | None = None,
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create SQL extract node.

        Parameters
        ----------
        name : str
            Node name
        query : str
            SQL query to execute
        database : str, optional
            Database connection reference
        deps : list, optional
            Dependencies
        **kwargs : Any
            Additional parameters

        Returns
        -------
        NodeSpec
            Node specification
        """

        async def wrapped_fn(input_data: dict, **ports: dict) -> dict:
            """Placeholder implementation."""
            return {
                "output": [],
                "metadata": {"query": query, "database": database, "status": "placeholder"},
            }

        wrapped_fn.__name__ = f"sql_extract_{name}"

        return self.create_node_with_mapping(
            name=name,
            wrapped_fn=wrapped_fn,
            input_schema={"input_data": dict, "**ports": dict},
            output_schema={"output": dict, "metadata": dict},
            deps=deps or [],
            **kwargs,
        )


class SQLLoadNode(BaseNodeFactory):
    """Load data into SQL databases.

    Placeholder implementation - to be completed with SQLAlchemy integration.
    """

    # Studio UI metadata
    _hexdag_icon = "DatabaseBackup"
    _hexdag_color = "#22c55e"  # green-500

    def __call__(
        self,
        name: str,
        table: str,
        mode: SqlLoadMode = "append",
        database: str | None = None,
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create SQL load node.

        Parameters
        ----------
        name : str
            Node name
        table : str
            Target table name
        mode : SqlLoadMode
            Load mode: 'append', 'replace', 'truncate_insert', 'merge'
        database : str, optional
            Database connection reference
        deps : list, optional
            Dependencies
        **kwargs : Any
            Additional parameters

        Returns
        -------
        NodeSpec
            Node specification
        """

        async def wrapped_fn(input_data: dict, **ports: dict) -> dict:
            """Placeholder implementation."""
            row_count = len(input_data.get("output", [])) if isinstance(input_data, dict) else 0
            return {"status": "loaded", "table": table, "rows": row_count}

        wrapped_fn.__name__ = f"sql_load_{name}"

        return self.create_node_with_mapping(
            name=name,
            wrapped_fn=wrapped_fn,
            input_schema={"input_data": dict, "**ports": dict},
            output_schema={"status": dict, "table": dict, "rows": dict},
            deps=deps or [],
            **kwargs,
        )
