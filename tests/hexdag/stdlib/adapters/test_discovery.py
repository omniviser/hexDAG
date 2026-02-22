"""Tests for adapter autodiscovery alias registration."""

from __future__ import annotations

from hexdag.stdlib.adapters._discovery import (
    _to_snake_case,
    discover_adapter_aliases,
    get_known_adapter_aliases,
)


class TestToSnakeCase:
    def test_simple(self) -> None:
        assert _to_snake_case("MockLLM") == "mock_llm"

    def test_adapter_suffix(self) -> None:
        assert _to_snake_case("OpenAIAdapter") == "open_ai_adapter"

    def test_multi_word(self) -> None:
        assert _to_snake_case("InMemoryMemory") == "in_memory_memory"

    def test_abbreviation(self) -> None:
        assert _to_snake_case("PgVectorAdapter") == "pg_vector_adapter"


class TestDiscoverAdapterAliases:
    def test_returns_dict(self) -> None:
        aliases = discover_adapter_aliases()
        assert isinstance(aliases, dict)
        assert len(aliases) > 0

    def test_snake_case_aliases(self) -> None:
        aliases = discover_adapter_aliases()
        assert "mock_llm" in aliases
        assert "open_ai_adapter" in aliases
        assert "anthropic_adapter" in aliases
        assert "in_memory_memory" in aliases
        assert "sq_lite_adapter" in aliases
        assert "tool_router" in aliases

    def test_port_qualified_aliases(self) -> None:
        aliases = discover_adapter_aliases()
        assert "llm:mock" in aliases
        assert "llm:openai" in aliases
        assert "llm:anthropic" in aliases
        assert "memory:in_memory" in aliases
        assert "memory:file" in aliases
        assert "database:sqlite" in aliases
        assert "database:pgvector" in aliases
        assert "embedding:mock" in aliases
        assert "secret:local" in aliases
        assert "tool_router:default" in aliases

    def test_camel_case_aliases(self) -> None:
        aliases = discover_adapter_aliases()
        assert "MockLLM" in aliases
        assert "OpenAIAdapter" in aliases
        assert "AnthropicAdapter" in aliases
        assert "InMemoryMemory" in aliases
        assert "ToolRouter" in aliases

    def test_backward_compat_aliases(self) -> None:
        aliases = discover_adapter_aliases()
        assert aliases["UnifiedToolRouter"] == "hexdag.kernel.ports.tool_router.ToolRouter"
        assert aliases["FunctionToolRouter"] == "hexdag.kernel.ports.tool_router.ToolRouter"
        assert aliases["FunctionBasedToolRouter"] == "hexdag.kernel.ports.tool_router.ToolRouter"

    def test_all_aliases_point_to_valid_paths(self) -> None:
        aliases = discover_adapter_aliases()
        for alias, path in aliases.items():
            assert "." in path, f"Alias '{alias}' has invalid path: {path}"

    def test_resolve_adapter_by_snake_case(self) -> None:
        from hexdag.kernel.resolver import resolve
        from hexdag.stdlib.adapters.mock import MockLLM

        assert resolve("mock_llm") is MockLLM

    def test_resolve_adapter_by_port_qualified(self) -> None:
        from hexdag.kernel.resolver import resolve
        from hexdag.stdlib.adapters.mock import MockLLM

        assert resolve("llm:mock") is MockLLM

    def test_resolve_adapter_by_camel_case(self) -> None:
        from hexdag.kernel.resolver import resolve
        from hexdag.stdlib.adapters.mock import MockLLM

        assert resolve("MockLLM") is MockLLM

    def test_resolve_tool_router_by_alias(self) -> None:
        from hexdag.kernel.ports.tool_router import ToolRouter
        from hexdag.kernel.resolver import resolve

        assert resolve("tool_router") is ToolRouter

    def test_full_path_still_works(self) -> None:
        from hexdag.kernel.resolver import resolve
        from hexdag.stdlib.adapters.mock import MockLLM

        assert resolve("hexdag.stdlib.adapters.mock.MockLLM") is MockLLM


class TestGetKnownAdapterAliases:
    def test_returns_frozenset(self) -> None:
        result = get_known_adapter_aliases()
        assert isinstance(result, frozenset)
        assert "mock_llm" in result

    def test_matches_discover_keys(self) -> None:
        aliases = discover_adapter_aliases()
        known = get_known_adapter_aliases()
        assert known == frozenset(aliases.keys())
