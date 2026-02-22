"""Tests for DatabaseTools lib."""

from __future__ import annotations

from typing import Any

import pytest

from hexdag.kernel.ports.data_store import SupportsQuery
from hexdag.stdlib.lib.database_tools import DatabaseTools
from hexdag.stdlib.lib_base import HexDAGLib

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeQueryStore:
    """Minimal SupportsQuery that records calls and returns canned data."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    async def aexecute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        self.calls.append((query, params))
        return self.rows


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDatabaseToolsIsLib:
    """DatabaseTools is a proper HexDAGLib subclass."""

    def test_isinstance(self) -> None:
        store = FakeQueryStore()
        lib = DatabaseTools(store=store)
        assert isinstance(lib, HexDAGLib)

    def test_store_satisfies_protocol(self) -> None:
        store = FakeQueryStore()
        assert isinstance(store, SupportsQuery)


class TestToolExposure:
    """Verify which methods are exposed as tools."""

    def test_exposed_tools(self) -> None:
        store = FakeQueryStore()
        lib = DatabaseTools(store=store)
        tools = lib.get_tools()
        assert "adatabase_query" in tools
        assert "alist_tables" in tools
        assert "adescribe_table" in tools

    def test_no_extra_tools(self) -> None:
        store = FakeQueryStore()
        lib = DatabaseTools(store=store)
        tools = lib.get_tools()
        assert len(tools) == 3


class TestAdatabaseQuery:
    @pytest.mark.asyncio()
    async def test_query_delegates_to_store(self) -> None:
        rows = [{"id": 1, "name": "alice"}]
        store = FakeQueryStore(rows=rows)
        lib = DatabaseTools(store=store)
        result = await lib.adatabase_query("SELECT * FROM users")
        assert result == rows
        assert len(store.calls) == 1
        assert "SELECT * FROM users" in store.calls[0][0]

    @pytest.mark.asyncio()
    async def test_query_with_params(self) -> None:
        store = FakeQueryStore(rows=[])
        lib = DatabaseTools(store=store)
        await lib.adatabase_query("SELECT * FROM users WHERE id = :id", {"id": 42})
        assert store.calls[0][1] == {"id": 42}

    @pytest.mark.asyncio()
    async def test_query_via_get_tools(self) -> None:
        rows = [{"count": 5}]
        store = FakeQueryStore(rows=rows)
        lib = DatabaseTools(store=store)
        tools = lib.get_tools()
        result = await tools["adatabase_query"]("SELECT count(*) FROM t")
        assert result == rows


class TestAlistTables:
    @pytest.mark.asyncio()
    async def test_list_tables_returns_names(self) -> None:
        store = FakeQueryStore(rows=[{"table_name": "users"}, {"table_name": "orders"}])
        lib = DatabaseTools(store=store)
        result = await lib.alist_tables()
        assert result == ["users", "orders"]

    @pytest.mark.asyncio()
    async def test_list_tables_empty(self) -> None:
        store = FakeQueryStore(rows=[])
        lib = DatabaseTools(store=store)
        result = await lib.alist_tables()
        assert result == []


class TestAdescribeTable:
    @pytest.mark.asyncio()
    async def test_describe_table_returns_columns(self) -> None:
        columns = [
            {
                "column_name": "id",
                "data_type": "integer",
                "is_nullable": "NO",
                "column_default": None,
            },
            {
                "column_name": "name",
                "data_type": "text",
                "is_nullable": "YES",
                "column_default": None,
            },
        ]
        store = FakeQueryStore(rows=columns)
        lib = DatabaseTools(store=store)
        result = await lib.adescribe_table("users")
        assert len(result) == 2
        assert result[0]["column_name"] == "id"

    @pytest.mark.asyncio()
    async def test_describe_table_passes_table_param(self) -> None:
        store = FakeQueryStore(rows=[])
        lib = DatabaseTools(store=store)
        await lib.adescribe_table("orders")
        assert store.calls[0][1] == {"table": "orders"}
