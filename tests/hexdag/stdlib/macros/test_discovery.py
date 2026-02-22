"""Tests for macro autodiscovery alias registration."""

from __future__ import annotations

from hexdag.stdlib.macros._discovery import (
    discover_macro_aliases,
    get_known_macro_aliases,
)


class TestDiscoverMacroAliases:
    def test_returns_dict(self) -> None:
        aliases = discover_macro_aliases()
        assert isinstance(aliases, dict)
        assert len(aliases) > 0

    def test_snake_case_aliases(self) -> None:
        aliases = discover_macro_aliases()
        assert "reasoning_agent_macro" in aliases
        assert "conversation_macro" in aliases
        assert "llm_macro" in aliases

    def test_core_qualified_aliases(self) -> None:
        aliases = discover_macro_aliases()
        assert "core:reasoning_agent" in aliases
        assert "core:conversation" in aliases
        assert "core:llm_workflow" in aliases

    def test_camel_case_aliases(self) -> None:
        aliases = discover_macro_aliases()
        assert "ReasoningAgentMacro" in aliases
        assert "ConversationMacro" in aliases
        assert "LLMMacro" in aliases

    def test_all_aliases_point_to_valid_paths(self) -> None:
        aliases = discover_macro_aliases()
        for alias, path in aliases.items():
            assert "." in path, f"Alias '{alias}' has invalid path: {path}"

    def test_resolve_macro_by_core_qualified(self) -> None:
        from hexdag.kernel.resolver import resolve
        from hexdag.stdlib.macros import ReasoningAgentMacro

        assert resolve("core:reasoning_agent") is ReasoningAgentMacro

    def test_resolve_macro_by_camel_case(self) -> None:
        from hexdag.kernel.resolver import resolve
        from hexdag.stdlib.macros import ConversationMacro

        assert resolve("ConversationMacro") is ConversationMacro

    def test_resolve_macro_by_snake_case(self) -> None:
        from hexdag.kernel.resolver import resolve
        from hexdag.stdlib.macros import LLMMacro

        assert resolve("llm_macro") is LLMMacro

    def test_full_path_still_works(self) -> None:
        from hexdag.kernel.resolver import resolve
        from hexdag.stdlib.macros import ReasoningAgentMacro

        assert resolve("hexdag.stdlib.macros.ReasoningAgentMacro") is ReasoningAgentMacro


class TestGetKnownMacroAliases:
    def test_returns_frozenset(self) -> None:
        result = get_known_macro_aliases()
        assert isinstance(result, frozenset)
        assert "core:reasoning_agent" in result

    def test_matches_discover_keys(self) -> None:
        aliases = discover_macro_aliases()
        known = get_known_macro_aliases()
        assert known == frozenset(aliases.keys())
