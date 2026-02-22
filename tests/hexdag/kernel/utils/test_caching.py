"""Tests for the caching utility module."""

from hexdag.kernel.utils.caching import KeyedCache, schema_cache_key


class TestKeyedCache:
    """Tests for KeyedCache."""

    def test_get_or_create_caches_value(self):
        cache: KeyedCache[int] = KeyedCache()
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return 42

        result1 = cache.get_or_create("key", factory)
        result2 = cache.get_or_create("key", factory)

        assert result1 == 42
        assert result2 == 42
        assert call_count == 1  # factory called only once

    def test_different_keys_call_factory_separately(self):
        cache: KeyedCache[str] = KeyedCache()
        result1 = cache.get_or_create("a", lambda: "alpha")
        result2 = cache.get_or_create("b", lambda: "beta")

        assert result1 == "alpha"
        assert result2 == "beta"
        assert len(cache) == 2

    def test_get_returns_cached_value(self):
        cache: KeyedCache[int] = KeyedCache()
        cache.put("x", 10)
        assert cache.get("x") == 10

    def test_get_returns_none_for_missing_key(self):
        cache: KeyedCache[int] = KeyedCache()
        assert cache.get("missing") is None

    def test_contains(self):
        cache: KeyedCache[int] = KeyedCache()
        cache.put("key", 1)
        assert "key" in cache
        assert "other" not in cache

    def test_clear(self):
        cache: KeyedCache[int] = KeyedCache()
        cache.put("a", 1)
        cache.put("b", 2)
        assert len(cache) == 2

        cache.clear()
        assert len(cache) == 0
        assert cache.get("a") is None

    def test_get_or_create_returns_same_object(self):
        """Verify identity — cached values are the exact same object."""
        cache: KeyedCache[list] = KeyedCache()
        obj = cache.get_or_create("k", lambda: [1, 2, 3])
        obj2 = cache.get_or_create("k", lambda: [4, 5, 6])
        assert obj is obj2


class TestSchemaCacheKey:
    """Tests for schema_cache_key."""

    def test_identical_dicts_produce_same_key(self):
        schema1 = {"name": str, "age": int}
        schema2 = {"age": int, "name": str}  # different order
        assert schema_cache_key(schema1) == schema_cache_key(schema2)

    def test_different_dicts_produce_different_keys(self):
        schema1 = {"name": str}
        schema2 = {"email": str}
        assert schema_cache_key(schema1) != schema_cache_key(schema2)

    def test_string_type_names(self):
        schema1 = {"name": "str", "count": "int"}
        schema2 = {"name": "str", "count": "int"}
        assert schema_cache_key(schema1) == schema_cache_key(schema2)

    def test_nullable_types(self):
        schema1 = {"name": "str?"}
        schema2 = {"name": "str?"}
        assert schema_cache_key(schema1) == schema_cache_key(schema2)

        schema3 = {"name": "str"}
        assert schema_cache_key(schema1) != schema_cache_key(schema3)
