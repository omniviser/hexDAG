"""Tests for ReasoningAgentMacro -- multi-step reasoning with adaptive tool calling."""

from __future__ import annotations

from typing import Any

from hexdag.kernel.domain.dag import DirectedGraph
from hexdag.stdlib.macros.reasoning_agent import (
    ReasoningAgentConfig,
    ReasoningAgentMacro,
)
from hexdag.stdlib.nodes.tool_utils import ToolCallFormat

# ---------------------------------------------------------------------------
# TestReasoningAgentConfig
# ---------------------------------------------------------------------------


class TestReasoningAgentConfig:
    """Tests for ReasoningAgentConfig validation."""

    def test_basic_config(self) -> None:
        """Config accepts basic fields."""
        config = ReasoningAgentConfig(
            name="test",
            main_prompt="Analyze this",
        )
        assert config.main_prompt == "Analyze this"
        assert config.max_steps == 5
        assert config.allowed_tools == []
        assert config.tool_format == ToolCallFormat.MIXED

    def test_custom_max_steps(self) -> None:
        """Config respects custom max_steps."""
        config = ReasoningAgentConfig(
            name="test",
            main_prompt="Think",
            max_steps=10,
        )
        assert config.max_steps == 10

    def test_custom_tool_format(self) -> None:
        """Config respects custom tool_format."""
        config = ReasoningAgentConfig(
            name="test",
            main_prompt="Think",
            tool_format=ToolCallFormat.JSON,
        )
        assert config.tool_format == ToolCallFormat.JSON

    def test_prompt_template_passthrough(self) -> None:
        """PromptTemplate objects are converted to string."""
        from hexdag.kernel.orchestration.prompt.template import PromptTemplate

        template = PromptTemplate("Analyze: {{topic}}")
        config = ReasoningAgentConfig(
            name="test",
            main_prompt=template,  # type: ignore[arg-type]
        )
        assert config.main_prompt == "Analyze: {{topic}}"

    def test_allowed_tools(self) -> None:
        """Config stores allowed tools list."""
        config = ReasoningAgentConfig(
            name="test",
            main_prompt="Think",
            allowed_tools=["core:search", "core:calculate"],
        )
        assert config.allowed_tools == ["core:search", "core:calculate"]


# ---------------------------------------------------------------------------
# TestReasoningAgentMacroExpansion
# ---------------------------------------------------------------------------


class TestReasoningAgentMacroExpansion:
    """Tests for expand() graph generation."""

    def _make_macro(self, **overrides: Any) -> ReasoningAgentMacro:
        """Create macro with default config."""
        defaults: dict[str, Any] = {
            "name": "test_agent",
            "main_prompt": "Reason about: {{topic}}",
            "max_steps": 3,
        }
        defaults.update(overrides)
        return ReasoningAgentMacro(**defaults)

    def test_expands_to_graph(self) -> None:
        """Expand produces a DirectedGraph."""
        macro = self._make_macro()
        graph = macro.expand(
            instance_name="agent1",
            inputs={"topic": "AI safety"},
            dependencies=[],
        )
        assert isinstance(graph, DirectedGraph)
        assert len(graph) > 0

    def test_graph_has_expected_nodes(self) -> None:
        """Graph contains LLM, adapter, tool executor, result merger nodes per step."""
        macro = self._make_macro(max_steps=2)
        graph = macro.expand(
            instance_name="agent1",
            inputs={},
            dependencies=[],
        )
        node_names = set(graph.keys())

        # Each step should produce: llm, adapter, tool_executor, result_merger
        for step_idx in range(2):
            prefix = f"agent1_step_{step_idx}"
            assert f"{prefix}_llm" in node_names, f"Missing {prefix}_llm"
            assert f"{prefix}_adapter" in node_names, f"Missing {prefix}_adapter"
            assert f"{prefix}_tool_executor" in node_names, f"Missing {prefix}_tool_executor"
            assert f"{prefix}_result_merger" in node_names, f"Missing {prefix}_result_merger"

        # Final consolidation node
        assert "agent1_final" in node_names

    def test_custom_max_steps(self) -> None:
        """Max steps controls number of reasoning steps in graph."""
        macro = self._make_macro(max_steps=1)
        graph = macro.expand(
            instance_name="single",
            inputs={},
            dependencies=[],
        )
        node_names = set(graph.keys())

        # Should have step_0 nodes but not step_1
        assert "single_step_0_llm" in node_names
        assert "single_step_1_llm" not in node_names
        assert "single_final" in node_names

    def test_dependencies_wired_to_first_step(self) -> None:
        """External dependencies are wired to the first step's LLM node."""
        macro = self._make_macro(max_steps=1)
        graph = macro.expand(
            instance_name="agent",
            inputs={},
            dependencies=["upstream_node"],
        )
        # First LLM node should depend on upstream_node
        first_llm = graph["agent_step_0_llm"]
        assert "upstream_node" in first_llm.deps

    def test_steps_chain_sequentially(self) -> None:
        """Each step depends on the previous step's result merger."""
        macro = self._make_macro(max_steps=2)
        graph = macro.expand(
            instance_name="agent",
            inputs={},
            dependencies=[],
        )
        # Step 1's LLM should depend on step 0's result merger
        step1_llm = graph["agent_step_1_llm"]
        assert "agent_step_0_result_merger" in step1_llm.deps

    def test_final_node_depends_on_last_step(self) -> None:
        """Final consolidation depends on the last step's result merger."""
        macro = self._make_macro(max_steps=2)
        graph = macro.expand(
            instance_name="agent",
            inputs={},
            dependencies=[],
        )
        final_node = graph["agent_final"]
        assert "agent_step_1_result_merger" in final_node.deps


# ---------------------------------------------------------------------------
# TestBuildToolSchemas
# ---------------------------------------------------------------------------


class TestBuildToolSchemas:
    """Tests for _build_tool_schemas_for_native."""

    def _make_macro(self) -> ReasoningAgentMacro:
        return ReasoningAgentMacro(name="test", main_prompt="Think")

    def test_returns_empty_for_no_tools(self) -> None:
        """Empty tool list returns empty schemas."""
        macro = self._make_macro()
        schemas = macro._build_tool_schemas_for_native([])
        assert schemas == []

    def test_handles_unresolvable_tools(self) -> None:
        """Unresolvable tool names are skipped with warning."""
        macro = self._make_macro()
        schemas = macro._build_tool_schemas_for_native(["nonexistent.module.tool"])
        assert schemas == []


# ---------------------------------------------------------------------------
# TestBuildToolListForText
# ---------------------------------------------------------------------------


class TestBuildToolListForText:
    """Tests for _build_tool_list_for_text."""

    def _make_macro(self) -> ReasoningAgentMacro:
        return ReasoningAgentMacro(name="test", main_prompt="Think")

    def test_no_tools_message(self) -> None:
        """Empty tool list returns no-tools message."""
        macro = self._make_macro()
        result = macro._build_tool_list_for_text([])
        assert result == "No tools available"

    def test_handles_unresolvable_tools(self) -> None:
        """Unresolvable tools show 'unavailable' description."""
        macro = self._make_macro()
        result = macro._build_tool_list_for_text(["nonexistent.mod.fn"])
        assert "unavailable" in result.lower()
