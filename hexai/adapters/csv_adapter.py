"""CSV file adapter implementation for hexDAG."""

import csv
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from hexai.ports.database import ColumnSchema, DatabasePort, TableSchema


class CsvAdapter(DatabasePort):
    """Adapter for reading CSV files from a directory as database tables."""

    def __init__(self, directory: str | Path) -> None:
        """
        Initialize CSV adapter with directory path.

        Args:
            directory: Path to directory containing CSV files
        """
        self.directory = Path(directory)
        if not self.directory.exists():
            raise ValueError(f"Directory not found: {directory}")

    async def get_table_schemas(self) -> Sequence[TableSchema]:
        """
        Get schema information for all CSV files in directory.

        Returns:
            Sequence[TableSchema]: List of table schemas
        """
        schemas = []
        for file_path in self.directory.glob("*.csv"):
            with open(file_path, newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue

                columns = [
                    ColumnSchema(
                        name=name,
                        type="text",  # Simple type inference
                        nullable=True,
                        primary_key=False,
                    )
                    for name in reader.fieldnames
                ]

                schemas.append(
                    TableSchema(
                        name=file_path.stem,  # Use filename without extension
                        columns=columns,
                    )
                )

        return schemas

    async def query(
        self,
        table: str,
        filters: dict[str, Any] | None = None,
        columns: list[str] | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Query rows from a CSV file with optional filtering and column selection.

        Args:
            table: Name of the CSV file (without .csv extension)
            filters: Optional column-value pairs to filter by
            columns: Optional list of columns to return (None = all)
            limit: Optional maximum number of rows to return

        Yields:
            dict[str, Any]: Each row as a dictionary

        Raises:
            ValueError: If table (CSV file) doesn't exist
        """
        file_path = self.directory / f"{table}.csv"
        if not file_path.exists():
            raise ValueError(f"Table (CSV file) not found: {table}")

        count = 0
        with open(file_path, newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return

            # Filter columns if specified
            field_names = columns if columns else reader.fieldnames

            for row in reader:
                # Apply filters
                if filters and not all(str(row.get(k)) == str(v) for k, v in filters.items()):
                    continue

                # Return only requested columns
                filtered_row = {k: row[k] for k in field_names if k in row}

                yield filtered_row

                count += 1
                if limit and count >= limit:
                    break
