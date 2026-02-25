"""CSV Database Adapter - Async CSV file reading with schema inference.

This module provides a Database implementation for reading CSV files from a directory,
with automatic type inference and async I/O for non-blocking operations.
"""

import csv
import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import aiofiles

from hexdag.kernel.ports.data_store import SupportsQuery
from hexdag.kernel.ports.database import ColumnSchema, ColumnType, TableSchema
from hexdag.stdlib.adapters.base import HexDAGAdapter

logging.basicConfig(level=logging.INFO)


class CsvAdapter(HexDAGAdapter, SupportsQuery, yaml_alias="csv_adapter", port="database"):
    """
    Adapter class for reading CSV files from a specified directory as database tables.

    Provides schema inference for CSV files and querying capabilities, supporting
    filters, column selection, and row limits.
    """

    def __init__(self, directory: str | Path) -> None:
        """
        Initialize the CSV adapter with a directory containing CSV files.

        Args:
            directory (str | Path): Path to the directory holding CSV files.

        Raises:
            ValueError: If the directory does not exist.
        """
        self.__directory = Path(directory)
        if not self.__directory.exists():
            raise ValueError(f"Directory not found: {directory}")

    @property
    def directory(self) -> Path:
        """Return the base directory as a pathlib.Path object."""
        return self.__directory

    def _infer_type(self, values: list[str]) -> str:
        """
        Infer column data type from a sample list of string values.

        Checks for integers, floats, booleans ('true'/'false'), else defaults to 'text'.

        Args:
            values (list[str]): List of sample values from a CSV column.

        Returns:
            str: Inferred data type ('int', 'float', 'text').
        """
        for v in values:
            if v == "":
                continue
            try:
                int(v)
                continue
            except ValueError:
                pass
            try:
                float(v)
                continue
            except ValueError:
                pass
            if v.lower() in ("true", "false"):
                continue
            return "text"
        if all(v.isdigit() or v == "" for v in values):
            return "int"
        return "float"

    async def aget_table_schemas(self) -> dict[str, dict[str, Any]]:
        """
        Get schema information for all CSV files in the adapter's directory.

        Reads each CSV file to infer column types and builds corresponding
        schema dictionaries.

        Returns:
            Dictionary mapping table names to schema information.
        """
        schemas = {}
        for file_path in self.directory.glob("*.csv"):
            async with aiofiles.open(file_path) as f:
                content = await f.read()
                # Process CSV content in memory
                reader = csv.DictReader(content.splitlines())
                if not reader.fieldnames:
                    logging.warning(f"No headers found in CSV file {file_path}, skipping.")
                    continue
                data = list(reader)

                # Build columns dict
                columns = {}
                primary_keys: list[str] = []
                for name in reader.fieldnames:
                    col_values = [row.get(name, "") for row in data]
                    col_type = self._infer_type(col_values)
                    columns[name] = col_type

                schemas[file_path.stem] = {
                    "table_name": file_path.stem,
                    "columns": columns,
                    "primary_keys": primary_keys,
                    "foreign_keys": [],
                }
        return schemas

    async def aexecute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query on CSV files.

        Note: CSV adapter doesn't support SQL queries. This method is required
        by Database but will raise NotImplementedError.

        Args:
            query: SQL query string (not supported)
            params: Optional query parameters (not supported)

        Raises:
            NotImplementedError: CSV adapter doesn't support SQL queries
        """
        raise NotImplementedError(
            "CSV adapter doesn't support SQL queries. Use query() method instead."
        )

    async def get_table_schemas(self) -> list[TableSchema]:
        """
        Generate table schemas for all CSV files in the adapter's directory.

        Reads each CSV file to infer column types and builds corresponding
        TableSchema and ColumnSchema objects.

        Returns:
            Sequence[TableSchema]: List of inferred table schemas, one per CSV file.
        """
        schemas = []
        for file_path in self.directory.glob("*.csv"):
            async with aiofiles.open(file_path) as f:
                content = await f.read()
                # Process CSV content in memory
                reader = csv.DictReader(content.splitlines())
                if not reader.fieldnames:
                    logging.warning(f"No headers found in CSV file {file_path}, skipping.")
                    continue
                data = list(reader)
                columns = []
                for name in reader.fieldnames:
                    col_values = [row.get(name, "") for row in data]
                    col_type = self._infer_type(col_values)
                    columns.append(
                        ColumnSchema(
                            name=name,
                            type=ColumnType[col_type.upper()],
                            nullable=True,
                            primary_key=False,
                        )
                    )
                schemas.append(
                    TableSchema(
                        name=file_path.stem,
                        columns=columns,
                    )
                )
        return schemas

    def _get_safe_file_path(self, table: str) -> Path:
        """
        Safely resolve the file path for a given table name within the base directory.

        Ensures the resolved path does not escape the base directory to prevent
        path traversal attacks.

        Args:
            table (str): Table name (CSV file name without extension).

        Raises:
            ValueError: If the resolved path is outside the base directory or file doesn't exist.

        Returns:
            Path: Resolved safe file path for the CSV.
        """
        file_path = self.directory / f"{table}.csv"
        resolved_path = file_path.resolve()
        base_dir_resolved = self.directory.resolve()

        if not str(resolved_path).startswith(str(base_dir_resolved)):
            raise ValueError(f"Attempted access outside base directory: {table}")

        if not resolved_path.exists():
            raise ValueError(f"Table (CSV file) not found: {table}")

        return resolved_path

    async def query(
        self,
        table: str,
        filters: dict[str, Any] | None = None,
        columns: list[str] | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Query rows from a CSV file with optional filtering, column selection, and row limit.

        Args:
            table (str): Name of the table (CSV file without '.csv').
            filters (dict[str, Any] | None): Optional column-value filters to apply.
            columns (list[str] | None): Optional list of columns to include in results.
            limit (int | None): Optional maximum number of rows to yield.

        Yields:
            dict[str, Any]: Rows matching filters with requested columns.

        Raises:
            ValueError: If the table file does not exist or path is unsafe.
        """
        file_path = self._get_safe_file_path(table)

        count = 0
        async with aiofiles.open(file_path) as f:
            content = await f.read()
            reader = csv.DictReader(content.splitlines())

            if not reader.fieldnames:
                return

            field_names = columns or reader.fieldnames

            for row in reader:
                if filters and any(
                    (isinstance(v, re.Pattern) and not v.search(str(row.get(k, ""))))
                    or (not isinstance(v, re.Pattern) and str(row.get(k, "")) != str(v))
                    for k, v in filters.items()
                ):
                    continue

                yield {k: row[k] for k in field_names if k in row}
                count += 1
                if limit and count >= limit:
                    break
