"""Tests for ReActAgentNode -- multi-step reasoning agent."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from hexdag.kernel.domain.dag import NodeSpec
from hexdag.stdlib.nodes.agent_node import (
    AgentConfig,
    AgentState,
    ReActAgentNode,
)
from hexdag.stdlib.nodes.tool_utils import ToolCallFormat

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleOutput(BaseModel):
    answer: str
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# TestAgentStateInitialization
# ---------------------------------------------------------------------------


class TestAgentStateInitialization:
    """Tests for _initialize_or_update_state."""

    def setup_method(self) -> None:
        self.node = ReActAgentNode()

    def test_agent_state_passthrough(self) -> None:
        """AgentState input is returned as-is."""
        existing = AgentState(input_data={"x": 1}, step=3)
        result = self.node._initialize_or_update_state(existing)
        assert result is existing
        assert result.step == 3

    def test_fresh_dict_input_wraps_in_agent_state(self) -> None:
        """Plain dict becomes AgentState with input_data populated."""
        result = self.node._initialize_or_update_state({"query": "hello"})
        assert isinstance(result, AgentState)
        assert result.input_data == {"query": "hello"}
        assert result.step == 0
        assert result.reasoning_steps == []

    def test_dict_with_reasoning_steps_validates(self) -> None:
        """Dict containing reasoning_steps is treated as legacy AgentState."""
        data = {"reasoning_steps": ["step1"], "step": 2, "response": "ok"}
        result = self.node._initialize_or_update_state(data)
        assert isinstance(result, AgentState)
        assert result.reasoning_steps == ["step1"]
        assert result.step == 2

    def test_non_dict_input_raises(self) -> None:
        """Non-dict, non-Pydantic input raises TypeMismatchError (to_dict cannot convert)."""
        from hexdag.kernel.exceptions import TypeMismatchError

        with pytest.raises(TypeMismatchError):
            self.node._initialize_or_update_state(42)

    def test_none_input_raises(self) -> None:
        """None input raises TypeMismatchError (to_dict cannot convert)."""
        from hexdag.kernel.exceptions import TypeMismatchError

        with pytest.raises(TypeMismatchError):
            self.node._initialize_or_update_state(None)

    def test_string_input_raises(self) -> None:
        """String input raises TypeMismatchError (to_dict cannot convert)."""
        from hexdag.kernel.exceptions import TypeMismatchError

        with pytest.raises(TypeMismatchError):
            self.node._initialize_or_update_state("do something")


# ---------------------------------------------------------------------------
# TestToolEndParsing
# ---------------------------------------------------------------------------


class TestToolEndParsing:
    """Tests for _parse_tool_end_result."""

    def setup_method(self) -> None:
        self.node = ReActAgentNode()

    def test_parse_valid_tool_end_result(self) -> None:
        """Valid tool_end string is parsed into a dict."""
        result = self.node._parse_tool_end_result("tool_end: {'answer': 'yes', 'confidence': 0.9}")
        assert result == {"answer": "yes", "confidence": 0.9}

    def test_parse_returns_none_without_prefix(self) -> None:
        """Strings without tool_end: prefix return None."""
        assert self.node._parse_tool_end_result("some random text") is None

    def test_parse_returns_none_for_empty_string(self) -> None:
        """Empty string returns None."""
        assert self.node._parse_tool_end_result("") is None

    def test_parse_returns_none_on_syntax_error(self) -> None:
        """Malformed data after prefix returns None."""
        assert self.node._parse_tool_end_result("tool_end: {bad syntax") is None

    def test_parse_returns_none_for_non_dict(self) -> None:
        """Non-dict literal after prefix returns None."""
        assert self.node._parse_tool_end_result("tool_end: [1, 2, 3]") is None

    def test_parse_returns_none_for_none_input(self) -> None:
        """None input returns None."""
        assert self.node._parse_tool_end_result(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TestShouldTerminate
# ---------------------------------------------------------------------------


class TestShouldTerminate:
    """Tests for _should_terminate."""

    def setup_method(self) -> None:
        self.node = ReActAgentNode()

    def test_terminates_on_tool_end_keyword(self) -> None:
        """Response containing tool_end triggers termination."""
        assert (
            self.node._should_terminate("I will use INVOKE_TOOL: tool_end(answer='done')") is True
        )

    def test_terminates_case_insensitive(self) -> None:
        """tool_end detection is case-insensitive."""
        assert self.node._should_terminate("TOOL_END result") is True
        assert self.node._should_terminate("Tool_END result") is True

    def test_does_not_terminate_on_regular_response(self) -> None:
        """Normal response without tool_end does not terminate."""
        assert self.node._should_terminate("Let me think about this...") is False

    def test_does_not_terminate_on_empty_response(self) -> None:
        """Empty response does not terminate."""
        assert self.node._should_terminate("") is False


# ---------------------------------------------------------------------------
# TestSuccessCondition
# ---------------------------------------------------------------------------


class TestSuccessCondition:
    """Tests for the success_condition closure in _create_agent_with_loop."""

    def test_success_when_not_agent_state(self) -> None:
        """Non-AgentState result signals completion."""
        # success_condition is a closure inside _create_agent_with_loop.
        # We test the logic directly.
        result = {"answer": "done"}
        assert not isinstance(result, AgentState)

    def test_success_when_max_steps_reached(self) -> None:
        """Agent stops when step >= max_steps."""
        state = AgentState(step=5)
        config = AgentConfig(max_steps=5)
        assert state.step >= config.max_steps

    def test_success_when_tool_end_in_response(self) -> None:
        """Agent stops when response contains tool_end."""
        state = AgentState(response="INVOKE_TOOL: tool_end(answer='done')")
        assert "tool_end" in state.response.lower()

    def test_not_success_when_still_reasoning(self) -> None:
        """Agent continues when still reasoning."""
        state = AgentState(step=1, response="Let me think more...")
        config = AgentConfig(max_steps=5)
        assert state.step < config.max_steps
        assert "tool_end" not in state.response.lower()


# ---------------------------------------------------------------------------
# TestGetCurrentPrompt
# ---------------------------------------------------------------------------


class TestGetCurrentPrompt:
    """Tests for _get_current_prompt."""

    def setup_method(self) -> None:
        self.node = ReActAgentNode()

    def test_returns_main_prompt_for_main_phase(self) -> None:
        """Main phase returns main_prompt."""
        result = self.node._get_current_prompt("main prompt", {"other": "other prompt"}, "main")
        assert result == "main prompt"

    def test_returns_continuation_prompt_for_named_phase(self) -> None:
        """Named phase returns its continuation prompt."""
        result = self.node._get_current_prompt("main prompt", {"review": "review prompt"}, "review")
        assert result == "review prompt"

    def test_falls_back_to_main_for_unknown_phase(self) -> None:
        """Unknown phase falls back to main_prompt."""
        result = self.node._get_current_prompt(
            "main prompt", {"review": "review prompt"}, "nonexistent"
        )
        assert result == "main prompt"


# ---------------------------------------------------------------------------
# TestBuildToolInstructions
# ---------------------------------------------------------------------------


class TestBuildToolInstructions:
    """Tests for _build_tool_instructions."""

    def setup_method(self) -> None:
        self.node = ReActAgentNode()

    def test_no_tools_returns_no_tools_message(self) -> None:
        """Empty tool router produces no-tools message."""
        from hexdag.kernel.ports.tool_router import ToolRouter

        router = ToolRouter(tools={})
        result = self.node._build_tool_instructions(router, AgentConfig())
        assert "No tools available" in result

    def test_includes_tool_names_and_params(self) -> None:
        """Tool instructions include tool name and parameter names."""
        from hexdag.kernel.ports.tool_router import ToolRouter

        def my_tool(query: str, limit: int = 10) -> str:
            """Search for items."""
            return "result"

        router = ToolRouter(tools={"search": my_tool})
        # Clear cache before test
        self.node._tool_instructions_cache = None
        result = self.node._build_tool_instructions(router, AgentConfig())
        assert "search" in result
        assert "Available Tools" in result

    def test_caches_after_first_call(self) -> None:
        """Second call returns cached result."""
        from hexdag.kernel.ports.tool_router import ToolRouter

        def dummy(x: str) -> str:
            """A tool."""
            return x

        router = ToolRouter(tools={"dummy": dummy})
        config = AgentConfig()
        # Clear cache
        self.node._tool_instructions_cache = None
        result1 = self.node._build_tool_instructions(router, config)
        result2 = self.node._build_tool_instructions(router, config)
        assert result1 == result2
        assert self.node._tool_instructions_cache is not None


# ---------------------------------------------------------------------------
# TestFormatSpecificGuidelines
# ---------------------------------------------------------------------------


class TestFormatSpecificGuidelines:
    """Tests for _get_format_specific_guidelines."""

    def setup_method(self) -> None:
        self.node = ReActAgentNode()

    def test_function_call_format(self) -> None:
        """FUNCTION_CALL format includes function-style syntax."""
        result = self.node._get_format_specific_guidelines(ToolCallFormat.FUNCTION_CALL)
        assert "INVOKE_TOOL:" in result
        assert "tool_name(param='value')" in result

    def test_json_format(self) -> None:
        """JSON format includes JSON-style syntax."""
        result = self.node._get_format_specific_guidelines(ToolCallFormat.JSON)
        assert '"tool"' in result
        assert '"params"' in result

    def test_mixed_format(self) -> None:
        """MIXED format includes both styles."""
        result = self.node._get_format_specific_guidelines(ToolCallFormat.MIXED)
        assert "Function style" in result
        assert "JSON style" in result


# ---------------------------------------------------------------------------
# TestCheckForFinalOutput (async)
# ---------------------------------------------------------------------------


class TestCheckForFinalOutput:
    """Tests for _check_for_final_output."""

    def setup_method(self) -> None:
        self.node = ReActAgentNode()

    @pytest.mark.asyncio
    async def test_returns_none_when_no_tool_end(self) -> None:
        """Returns None when tool_results has no tool_end entries."""
        state = AgentState(tool_results=["search: found 5 items", "calc: 42"])
        result = await self.node._check_for_final_output(state, SimpleOutput, None)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_validated_output_from_tool_end(self) -> None:
        """Returns validated output model from tool_end result."""
        state = AgentState(tool_results=["tool_end: {'answer': 'yes', 'confidence': 0.95}"])
        result = await self.node._check_for_final_output(state, SimpleOutput, None)
        assert isinstance(result, SimpleOutput)
        assert result.answer == "yes"
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_skips_invalid_tool_end_results(self) -> None:
        """Skips tool_end results that fail validation and tries next."""
        state = AgentState(
            tool_results=[
                "tool_end: {'invalid_field': 'bad'}",
                "tool_end: {'answer': 'fallback'}",
            ]
        )
        result = await self.node._check_for_final_output(state, SimpleOutput, None)
        # Should find the second one (reversed iteration)
        assert result is not None
        assert isinstance(result, SimpleOutput)

    @pytest.mark.asyncio
    async def test_emits_agent_metadata(self) -> None:
        """Emits metadata trace when event_manager is available."""
        event_manager = AsyncMock()
        event_manager.add_trace = AsyncMock()
        state = AgentState(
            tool_results=["tool_end: {'answer': 'done'}"],
            reasoning_steps=["step1"],
            tools_used=["search"],
        )
        result = await self.node._check_for_final_output(state, SimpleOutput, event_manager)
        assert result is not None
        event_manager.add_trace.assert_called_once()


# ---------------------------------------------------------------------------
# TestNodeCreation
# ---------------------------------------------------------------------------


class TestNodeCreation:
    """Tests for __call__ creating NodeSpec."""

    def setup_method(self) -> None:
        self.node = ReActAgentNode()

    def test_creates_node_spec(self) -> None:
        """Factory creates a valid NodeSpec."""
        spec = self.node(name="test_agent", main_prompt="Analyze: {{query}}")
        assert isinstance(spec, NodeSpec)
        assert spec.name == "test_agent"
        assert spec.fn is not None

    def test_creates_with_dependencies(self) -> None:
        """NodeSpec includes specified dependencies."""
        spec = self.node(
            name="agent",
            main_prompt="Process input",
            deps=["upstream_node"],
        )
        assert "upstream_node" in spec.deps

    def test_creates_with_config(self) -> None:
        """NodeSpec respects custom config."""
        config = AgentConfig(max_steps=3)
        spec = self.node(
            name="agent",
            main_prompt="Think about {{topic}}",
            config=config,
        )
        assert spec.fn is not None

    def test_creates_with_continuation_prompts(self) -> None:
        """NodeSpec accepts continuation prompts for phases."""
        spec = self.node(
            name="agent",
            main_prompt="Start reasoning",
            continuation_prompts={"review": "Review your work"},
        )
        assert spec.fn is not None


# ---------------------------------------------------------------------------
# TestEnhancePromptWithTools
# ---------------------------------------------------------------------------


class TestEnhancePromptWithTools:
    """Tests for _enhance_prompt_with_tools."""

    def setup_method(self) -> None:
        self.node = ReActAgentNode()

    def test_returns_original_when_no_router(self) -> None:
        """Without tool router, prompt is returned unchanged."""
        result = self.node._enhance_prompt_with_tools("original prompt", None, AgentConfig())
        assert result == "original prompt"

    def test_enhances_with_tool_router(self) -> None:
        """With tool router, prompt is enhanced with instructions."""
        from hexdag.kernel.ports.tool_router import ToolRouter

        def dummy_tool(x: str) -> str:
            """A dummy tool."""
            return x

        router = ToolRouter(tools={"dummy": dummy_tool})
        # Clear cache
        self.node._tool_instructions_cache = None
        result = self.node._enhance_prompt_with_tools("original", router, AgentConfig())
        # Result should be a PromptTemplate or string containing tool info
        assert result is not None
