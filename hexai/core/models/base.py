"""Base Pydantic models for HexDAG nodes."""

from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class DataReference(BaseModel):
    """Base class for references to offloaded data.

    This pattern works for Spark (Delta/Parquet), cloud storage (S3/GCS),
    or any system where data is stored externally instead of in memory.

    Attributes
    ----------
    location : str
        URI of the data (e.g., "s3://bucket/path", "delta://table", "file:///path")
    format : str
        Data format (e.g., "parquet", "delta", "csv", "json")
    schema_info : dict[str, str] | None
        Column names and types
    size_bytes : int | None
        Size of the data in bytes
    row_count : int | None
        Number of rows (for tabular data)
    created_at : str | None
        ISO timestamp of when data was created
    """

    location: str
    format: str
    schema_info: dict[str, str] | None = None  # Renamed to avoid conflict with BaseModel.schema
    size_bytes: int | None = None
    row_count: int | None = None
    created_at: str | None = None

    model_config = ConfigDict(frozen=True)  # Immutable references


class FileData(BaseModel):
    """Reference to file-based data.

    Attributes
    ----------
    path : Path
        Path to the file
    encoding : str
        File encoding (default: utf-8)
    mime_type : str | None
        MIME type of the file
    size_bytes : int | None
        File size in bytes
    """

    path: Path
    encoding: str = "utf-8"
    mime_type: str | None = None
    size_bytes: int | None = None

    model_config = ConfigDict(frozen=True)


class InMemoryData(BaseModel, Generic[T]):
    """Container for small data that stays in memory.

    Use this for configuration, metadata, or small datasets
    that don't need to be offloaded.

    Attributes
    ----------
    data : T
        The actual data (any type)
    metadata : dict[str, Any]
        Optional metadata about the data
    """

    data: T
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class StringData(BaseModel):
    """Simple string data container.

    Attributes
    ----------
    value : str
        The string value
    encoding : str
        String encoding (default: utf-8)
    """

    value: str
    encoding: str = "utf-8"

    model_config = ConfigDict(frozen=True)


class JsonData(BaseModel):
    """JSON data container.

    Attributes
    ----------
    data : dict[str, Any]
        The JSON data as a dictionary
    schema_version : str | None
        Optional schema version identifier
    """

    data: dict[str, Any]
    schema_version: str | None = None

    model_config = ConfigDict(frozen=True)


class ListData(BaseModel, Generic[T]):
    """Container for list data.

    Attributes
    ----------
    items : list[T]
        The list of items
    count : int | None
        Optional cached count (useful for large lists)
    """

    items: list[T]

    @property
    def count(self) -> int:
        """Get the count of items."""
        return len(self.items)

    model_config = ConfigDict(frozen=True)
