"""Tests for InMemoryCollectionStorage adapter."""

from __future__ import annotations

import pytest

from hexdag.stdlib.adapters.memory.collection_memory import InMemoryCollectionStorage


class TestSaveAndLoad:
    @pytest.mark.asyncio()
    async def test_save_and_load(self) -> None:
        storage = InMemoryCollectionStorage()
        await storage.asave("col", "k1", {"name": "alice"})
        result = await storage.aload("col", "k1")
        assert result == {"name": "alice"}

    @pytest.mark.asyncio()
    async def test_load_missing_returns_none(self) -> None:
        storage = InMemoryCollectionStorage()
        assert await storage.aload("col", "missing") is None

    @pytest.mark.asyncio()
    async def test_load_missing_collection_returns_none(self) -> None:
        storage = InMemoryCollectionStorage()
        assert await storage.aload("nonexistent", "k1") is None

    @pytest.mark.asyncio()
    async def test_upsert_overwrites(self) -> None:
        storage = InMemoryCollectionStorage()
        await storage.asave("col", "k1", {"v": 1})
        await storage.asave("col", "k1", {"v": 2})
        result = await storage.aload("col", "k1")
        assert result == {"v": 2}


class TestQuery:
    @pytest.mark.asyncio()
    async def test_query_all(self) -> None:
        storage = InMemoryCollectionStorage()
        await storage.asave("col", "k1", {"status": "a"})
        await storage.asave("col", "k2", {"status": "b"})
        result = await storage.aquery("col")
        assert len(result) == 2

    @pytest.mark.asyncio()
    async def test_query_with_filter(self) -> None:
        storage = InMemoryCollectionStorage()
        await storage.asave("col", "k1", {"status": "running", "ref": "x"})
        await storage.asave("col", "k2", {"status": "completed", "ref": "x"})
        await storage.asave("col", "k3", {"status": "running", "ref": "y"})
        result = await storage.aquery("col", {"status": "running"})
        assert len(result) == 2

    @pytest.mark.asyncio()
    async def test_query_with_multiple_filters(self) -> None:
        storage = InMemoryCollectionStorage()
        await storage.asave("col", "k1", {"status": "running", "ref": "x"})
        await storage.asave("col", "k2", {"status": "running", "ref": "y"})
        result = await storage.aquery("col", {"status": "running", "ref": "x"})
        assert len(result) == 1
        assert result[0]["ref"] == "x"

    @pytest.mark.asyncio()
    async def test_query_empty_collection(self) -> None:
        storage = InMemoryCollectionStorage()
        assert await storage.aquery("empty") == []

    @pytest.mark.asyncio()
    async def test_query_no_match(self) -> None:
        storage = InMemoryCollectionStorage()
        await storage.asave("col", "k1", {"status": "a"})
        assert await storage.aquery("col", {"status": "z"}) == []


class TestDelete:
    @pytest.mark.asyncio()
    async def test_delete_existing(self) -> None:
        storage = InMemoryCollectionStorage()
        await storage.asave("col", "k1", {"x": 1})
        assert await storage.adelete("col", "k1") is True
        assert await storage.aload("col", "k1") is None

    @pytest.mark.asyncio()
    async def test_delete_missing(self) -> None:
        storage = InMemoryCollectionStorage()
        assert await storage.adelete("col", "missing") is False

    @pytest.mark.asyncio()
    async def test_delete_missing_collection(self) -> None:
        storage = InMemoryCollectionStorage()
        assert await storage.adelete("nonexistent", "k1") is False


class TestCollectionIsolation:
    @pytest.mark.asyncio()
    async def test_collections_are_independent(self) -> None:
        storage = InMemoryCollectionStorage()
        await storage.asave("a", "k1", {"from": "a"})
        await storage.asave("b", "k1", {"from": "b"})
        assert (await storage.aload("a", "k1")) == {"from": "a"}
        assert (await storage.aload("b", "k1")) == {"from": "b"}
        await storage.adelete("a", "k1")
        assert await storage.aload("a", "k1") is None
        assert (await storage.aload("b", "k1")) == {"from": "b"}
