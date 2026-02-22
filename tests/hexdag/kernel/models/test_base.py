"""Tests for the base models module.

This module tests the base Pydantic models for HexDAG.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from hexdag.kernel.models.base import (
    DataReference,
    FileData,
    InMemoryData,
    JsonData,
    ListData,
    StringData,
)


class TestDataReference:
    """Tests for DataReference model."""

    def test_basic_creation(self) -> None:
        """Test creating a basic DataReference."""
        ref = DataReference(location="s3://bucket/path", format="parquet")
        assert ref.location == "s3://bucket/path"
        assert ref.format == "parquet"
        assert ref.schema_info is None
        assert ref.size_bytes is None
        assert ref.row_count is None
        assert ref.created_at is None

    def test_full_creation(self) -> None:
        """Test creating a fully specified DataReference."""
        ref = DataReference(
            location="delta://table/name",
            format="delta",
            schema_info={"id": "int", "name": "string"},
            size_bytes=1024000,
            row_count=1000,
            created_at="2024-01-15T10:30:00Z",
        )
        assert ref.location == "delta://table/name"
        assert ref.format == "delta"
        assert ref.schema_info == {"id": "int", "name": "string"}
        assert ref.size_bytes == 1024000
        assert ref.row_count == 1000
        assert ref.created_at == "2024-01-15T10:30:00Z"

    def test_frozen_immutability(self) -> None:
        """Test that DataReference is frozen (immutable)."""
        ref = DataReference(location="s3://bucket/path", format="parquet")
        with pytest.raises(ValidationError):
            ref.location = "new_location"  # type: ignore[misc]

    def test_various_formats(self) -> None:
        """Test various data formats."""
        formats = ["parquet", "delta", "csv", "json", "avro", "orc"]
        for fmt in formats:
            ref = DataReference(location="file:///data", format=fmt)
            assert ref.format == fmt

    def test_various_locations(self) -> None:
        """Test various location URIs."""
        locations = [
            "s3://bucket/key",
            "gs://bucket/path",
            "hdfs://cluster/path",
            "file:///local/path",
            "delta://database.table",
        ]
        for loc in locations:
            ref = DataReference(location=loc, format="parquet")
            assert ref.location == loc


class TestFileData:
    """Tests for FileData model."""

    def test_basic_creation(self) -> None:
        """Test creating a basic FileData."""
        file_data = FileData(path=Path("/tmp/test.txt"))
        assert file_data.path == Path("/tmp/test.txt")
        assert file_data.encoding == "utf-8"
        assert file_data.mime_type is None
        assert file_data.size_bytes is None

    def test_full_creation(self) -> None:
        """Test creating a fully specified FileData."""
        file_data = FileData(
            path=Path("/data/file.json"),
            encoding="utf-16",
            mime_type="application/json",
            size_bytes=2048,
        )
        assert file_data.path == Path("/data/file.json")
        assert file_data.encoding == "utf-16"
        assert file_data.mime_type == "application/json"
        assert file_data.size_bytes == 2048

    def test_frozen_immutability(self) -> None:
        """Test that FileData is frozen (immutable)."""
        file_data = FileData(path=Path("/tmp/test.txt"))
        with pytest.raises(ValidationError):
            file_data.path = Path("/new/path")  # type: ignore[misc]

    def test_various_encodings(self) -> None:
        """Test various file encodings."""
        encodings = ["utf-8", "utf-16", "ascii", "latin-1", "iso-8859-1"]
        for enc in encodings:
            file_data = FileData(path=Path("/tmp/test.txt"), encoding=enc)
            assert file_data.encoding == enc

    def test_various_mime_types(self) -> None:
        """Test various MIME types."""
        mime_types = [
            "text/plain",
            "application/json",
            "application/xml",
            "text/csv",
            "application/octet-stream",
        ]
        for mime in mime_types:
            file_data = FileData(path=Path("/tmp/file"), mime_type=mime)
            assert file_data.mime_type == mime


class TestInMemoryData:
    """Tests for InMemoryData model."""

    def test_basic_creation_string(self) -> None:
        """Test creating InMemoryData with string."""
        data = InMemoryData(data="hello world")
        assert data.data == "hello world"
        assert data.metadata == {}

    def test_basic_creation_dict(self) -> None:
        """Test creating InMemoryData with dict."""
        data = InMemoryData(data={"key": "value"})
        assert data.data == {"key": "value"}

    def test_basic_creation_list(self) -> None:
        """Test creating InMemoryData with list."""
        data = InMemoryData(data=[1, 2, 3])
        assert data.data == [1, 2, 3]

    def test_with_metadata(self) -> None:
        """Test creating InMemoryData with metadata."""
        data = InMemoryData(
            data="test data",
            metadata={"source": "api", "timestamp": "2024-01-15"},
        )
        assert data.data == "test data"
        assert data.metadata["source"] == "api"
        assert data.metadata["timestamp"] == "2024-01-15"

    def test_complex_data(self) -> None:
        """Test creating InMemoryData with complex data."""
        complex_data = {
            "users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            "total": 2,
        }
        data = InMemoryData(data=complex_data)
        assert data.data["total"] == 2
        assert len(data.data["users"]) == 2

    def test_arbitrary_types(self) -> None:
        """Test that arbitrary types are allowed."""

        @dataclass
        class CustomClass:
            value: int

        custom = CustomClass(42)
        data = InMemoryData(data=custom)
        assert data.data.value == 42


class TestStringData:
    """Tests for StringData model."""

    def test_basic_creation(self) -> None:
        """Test creating a basic StringData."""
        string_data = StringData(value="hello world")
        assert string_data.value == "hello world"
        assert string_data.encoding == "utf-8"

    def test_custom_encoding(self) -> None:
        """Test creating StringData with custom encoding."""
        string_data = StringData(value="test", encoding="ascii")
        assert string_data.encoding == "ascii"

    def test_frozen_immutability(self) -> None:
        """Test that StringData is frozen (immutable)."""
        string_data = StringData(value="test")
        with pytest.raises(ValidationError):
            string_data.value = "new value"  # type: ignore[misc]

    def test_empty_string(self) -> None:
        """Test creating StringData with empty string."""
        string_data = StringData(value="")
        assert string_data.value == ""

    def test_multiline_string(self) -> None:
        """Test creating StringData with multiline content."""
        content = """Line 1
Line 2
Line 3"""
        string_data = StringData(value=content)
        assert "Line 1" in string_data.value
        assert string_data.value.count("\n") == 2


class TestJsonData:
    """Tests for JsonData model."""

    def test_basic_creation(self) -> None:
        """Test creating a basic JsonData."""
        json_data = JsonData(data={"key": "value"})
        assert json_data.data == {"key": "value"}
        assert json_data.schema_version is None

    def test_with_schema_version(self) -> None:
        """Test creating JsonData with schema version."""
        json_data = JsonData(data={"id": 1}, schema_version="1.0.0")
        assert json_data.schema_version == "1.0.0"

    def test_frozen_immutability(self) -> None:
        """Test that JsonData is frozen (immutable)."""
        json_data = JsonData(data={"key": "value"})
        with pytest.raises(ValidationError):
            json_data.data = {"new": "data"}  # type: ignore[misc]

    def test_empty_dict(self) -> None:
        """Test creating JsonData with empty dict."""
        json_data = JsonData(data={})
        assert json_data.data == {}

    def test_nested_data(self) -> None:
        """Test creating JsonData with nested structure."""
        nested = {
            "level1": {
                "level2": {
                    "level3": ["a", "b", "c"],
                },
            },
        }
        json_data = JsonData(data=nested)
        assert json_data.data["level1"]["level2"]["level3"] == ["a", "b", "c"]


class TestListData:
    """Tests for ListData model."""

    def test_basic_creation(self) -> None:
        """Test creating a basic ListData."""
        list_data = ListData(items=[1, 2, 3])
        assert list_data.items == [1, 2, 3]

    def test_count_property(self) -> None:
        """Test the count property."""
        list_data = ListData(items=["a", "b", "c", "d", "e"])
        assert list_data.count == 5

    def test_empty_list(self) -> None:
        """Test creating ListData with empty list."""
        list_data = ListData(items=[])
        assert list_data.items == []
        assert list_data.count == 0

    def test_frozen_immutability(self) -> None:
        """Test that ListData is frozen (immutable)."""
        list_data = ListData(items=[1, 2, 3])
        with pytest.raises(ValidationError):
            list_data.items = [4, 5, 6]  # type: ignore[misc]

    def test_string_items(self) -> None:
        """Test ListData with string items."""
        list_data = ListData(items=["apple", "banana", "cherry"])
        assert list_data.count == 3
        assert "banana" in list_data.items

    def test_dict_items(self) -> None:
        """Test ListData with dict items."""
        list_data = ListData(items=[{"id": 1}, {"id": 2}])
        assert list_data.count == 2
        assert list_data.items[0]["id"] == 1

    def test_large_list(self) -> None:
        """Test ListData with large list."""
        items = list(range(10000))
        list_data = ListData(items=items)
        assert list_data.count == 10000
        assert list_data.items[9999] == 9999
