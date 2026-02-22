"""Tests for the unified DataStore capability protocols."""

from __future__ import annotations

from typing import Any

import pytest

from hexdag.kernel.ports.data_store import (
    SupportsKeyValue,
    SupportsQuery,
    SupportsSchema,
    SupportsTransactions,
    SupportsTTL,
)

# ---------------------------------------------------------------------------
# Concrete test doubles that implement the protocols
# ---------------------------------------------------------------------------


class FakeKeyValueStore:
    """Minimal SupportsKeyValue implementation for testing."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    async def aget(self, key: str) -> Any:
        return self._data.get(key)

    async def aset(self, key: str, value: Any) -> None:
        self._data[key] = value

    async def adelete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False

    async def aexists(self, key: str) -> bool:
        return key in self._data

    async def alist_keys(self, prefix: str = "") -> list[str]:
        if prefix:
            return [k for k in self._data if k.startswith(prefix)]
        return list(self._data.keys())


class FakeTTLStore(FakeKeyValueStore):
    """Adds TTL support on top of key-value."""

    def __init__(self) -> None:
        super().__init__()
        self._ttls: dict[str, int] = {}

    async def aset_with_ttl(self, key: str, value: Any, ttl_seconds: int) -> None:
        self._data[key] = value
        self._ttls[key] = ttl_seconds


class FakeQueryStore:
    """Minimal SupportsQuery implementation for testing."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = rows or []

    async def aexecute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return self._rows


class FakeSchemaStore:
    """Minimal SupportsSchema implementation for testing."""

    async def aget_table_schemas(self) -> dict[str, dict[str, Any]]:
        return {"users": {"columns": {"id": "int", "name": "text"}}}

    async def aget_relationships(self) -> list[dict[str, Any]]:
        return []

    async def aget_indexes(self, table: str) -> list[dict[str, Any]]:
        return []

    async def aget_table_statistics(self, table: str) -> dict[str, Any]:
        return {"row_count": 0}


class FakeTransactionalStore:
    """Minimal SupportsTransactions implementation for testing."""

    def __init__(self) -> None:
        self.begun = False
        self.committed = False
        self.rolled_back = False

    async def abegin(self) -> None:
        self.begun = True

    async def acommit(self) -> None:
        self.committed = True

    async def arollback(self) -> None:
        self.rolled_back = True


class NotAStore:
    """Object that implements none of the protocols."""

    async def do_something(self) -> None: ...


# ---------------------------------------------------------------------------
# Protocol isinstance checks (runtime_checkable)
# ---------------------------------------------------------------------------


class TestProtocolChecks:
    """Verify that runtime_checkable isinstance works for each protocol."""

    def test_supports_key_value_positive(self) -> None:
        store = FakeKeyValueStore()
        assert isinstance(store, SupportsKeyValue)

    def test_supports_key_value_negative(self) -> None:
        assert not isinstance(NotAStore(), SupportsKeyValue)

    def test_supports_ttl_positive(self) -> None:
        store = FakeTTLStore()
        assert isinstance(store, SupportsTTL)

    def test_supports_ttl_negative(self) -> None:
        # FakeKeyValueStore lacks aset_with_ttl
        assert not isinstance(FakeKeyValueStore(), SupportsTTL)

    def test_supports_query_positive(self) -> None:
        store = FakeQueryStore()
        assert isinstance(store, SupportsQuery)

    def test_supports_query_negative(self) -> None:
        assert not isinstance(NotAStore(), SupportsQuery)

    def test_supports_schema_positive(self) -> None:
        store = FakeSchemaStore()
        assert isinstance(store, SupportsSchema)

    def test_supports_schema_negative(self) -> None:
        assert not isinstance(NotAStore(), SupportsSchema)

    def test_supports_transactions_positive(self) -> None:
        store = FakeTransactionalStore()
        assert isinstance(store, SupportsTransactions)

    def test_supports_transactions_negative(self) -> None:
        assert not isinstance(NotAStore(), SupportsTransactions)


# ---------------------------------------------------------------------------
# Capability composition — objects implementing multiple protocols
# ---------------------------------------------------------------------------


class FakeFullStore(FakeKeyValueStore, FakeQueryStore, FakeSchemaStore):
    """Store implementing key-value, query, and schema protocols."""

    def __init__(self) -> None:
        FakeKeyValueStore.__init__(self)
        FakeQueryStore.__init__(self, rows=[])


class TestComposition:
    """Verify that a single adapter can satisfy multiple DataStore sub-protocols."""

    def test_full_store_satisfies_key_value(self) -> None:
        store = FakeFullStore()
        assert isinstance(store, SupportsKeyValue)

    def test_full_store_satisfies_query(self) -> None:
        store = FakeFullStore()
        assert isinstance(store, SupportsQuery)

    def test_full_store_satisfies_schema(self) -> None:
        store = FakeFullStore()
        assert isinstance(store, SupportsSchema)


# ---------------------------------------------------------------------------
# Functional tests on FakeKeyValueStore
# ---------------------------------------------------------------------------


class TestKeyValueOperations:
    """Verify SupportsKeyValue operations on FakeKeyValueStore."""

    @pytest.fixture()
    def store(self) -> FakeKeyValueStore:
        return FakeKeyValueStore()

    @pytest.mark.asyncio()
    async def test_get_missing_returns_none(self, store: FakeKeyValueStore) -> None:
        assert await store.aget("missing") is None

    @pytest.mark.asyncio()
    async def test_set_and_get(self, store: FakeKeyValueStore) -> None:
        await store.aset("key1", "value1")
        assert await store.aget("key1") == "value1"

    @pytest.mark.asyncio()
    async def test_delete_existing(self, store: FakeKeyValueStore) -> None:
        await store.aset("key1", "value1")
        assert await store.adelete("key1") is True
        assert await store.aget("key1") is None

    @pytest.mark.asyncio()
    async def test_delete_missing(self, store: FakeKeyValueStore) -> None:
        assert await store.adelete("missing") is False

    @pytest.mark.asyncio()
    async def test_exists_true(self, store: FakeKeyValueStore) -> None:
        await store.aset("key1", "value1")
        assert await store.aexists("key1") is True

    @pytest.mark.asyncio()
    async def test_exists_false(self, store: FakeKeyValueStore) -> None:
        assert await store.aexists("missing") is False

    @pytest.mark.asyncio()
    async def test_list_keys_empty(self, store: FakeKeyValueStore) -> None:
        assert await store.alist_keys() == []

    @pytest.mark.asyncio()
    async def test_list_keys_all(self, store: FakeKeyValueStore) -> None:
        await store.aset("a", 1)
        await store.aset("b", 2)
        keys = await store.alist_keys()
        assert sorted(keys) == ["a", "b"]

    @pytest.mark.asyncio()
    async def test_list_keys_with_prefix(self, store: FakeKeyValueStore) -> None:
        await store.aset("user:1", "alice")
        await store.aset("user:2", "bob")
        await store.aset("order:1", "pizza")
        keys = await store.alist_keys(prefix="user:")
        assert sorted(keys) == ["user:1", "user:2"]


# ---------------------------------------------------------------------------
# TTL operations
# ---------------------------------------------------------------------------


class TestTTLOperations:
    """Verify SupportsTTL operations."""

    @pytest.mark.asyncio()
    async def test_set_with_ttl(self) -> None:
        store = FakeTTLStore()
        await store.aset_with_ttl("temp", "data", ttl_seconds=60)
        assert await store.aget("temp") == "data"
        assert store._ttls["temp"] == 60


# ---------------------------------------------------------------------------
# Transaction operations
# ---------------------------------------------------------------------------


class TestTransactionOperations:
    """Verify SupportsTransactions operations."""

    @pytest.mark.asyncio()
    async def test_transaction_lifecycle(self) -> None:
        store = FakeTransactionalStore()
        await store.abegin()
        assert store.begun is True
        await store.acommit()
        assert store.committed is True

    @pytest.mark.asyncio()
    async def test_transaction_rollback(self) -> None:
        store = FakeTransactionalStore()
        await store.abegin()
        await store.arollback()
        assert store.rolled_back is True


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


class TestExports:
    """Verify all protocols are importable from the ports package."""

    def test_import_from_ports_init(self) -> None:
        from hexdag.kernel.ports import (
            SupportsKeyValue,
            SupportsQuery,
            SupportsSchema,
            SupportsTransactions,
            SupportsTTL,
        )

        assert SupportsKeyValue is not None
        assert SupportsQuery is not None
        assert SupportsSchema is not None
        assert SupportsTTL is not None
        assert SupportsTransactions is not None
