"""Test cases for ReActAgentNode class."""

import pytest
from pydantic import BaseModel

from hexai.adapters.mock.mock_llm import MockLLM
from hexai.adapters.unified_tool_router import UnifiedToolRouter
from hexai.core.application.nodes.agent_node import AgentConfig
from hexai.core.application.nodes.tool_utils import ToolCallFormat
from hexai.core.bootstrap import ensure_bootstrapped
from hexai.core.context import ExecutionContext
from hexai.core.registry import registry

# Ensure registry is bootstrapped for tests
ensure_bootstrapped()


class CustomOutput(BaseModel):
    """Custom output model for testing."""

    result: str
    confidence: float = 0.0


class TestReActAgentNode:
    """Test cases for ReActAgentNode class."""

    @pytest.fixture
    def mock_llm(self):
        """Fixture for mock LLM."""
        return MockLLM()

    @pytest.fixture
    def mock_tool_router(self):
        """Fixture for mock tool router."""
        return UnifiedToolRouter()

    @pytest.fixture
    def agent_node(self):
        """Get ReActAgentNode factory from registry."""
        ensure_bootstrapped()
        return registry.get("agent_node", namespace="core")

    def test_basic_agent_creation(self, agent_node):
        """Test basic agent creation and NodeSpec generation."""
        main_prompt = "Analyze the input: {{input}}"

        node_spec = agent_node("test_agent", main_prompt=main_prompt)

        assert node_spec.name == "test_agent"
        assert node_spec.fn.__name__ == "agent_with_internal_loop"
        assert node_spec.in_model is not None
        assert node_spec.out_model is not None

    def test_agent_with_custom_config(self, agent_node):
        """Test agent with custom configuration."""
        main_prompt = "Analyze the input: {{input}}"
        config = AgentConfig(max_steps=5, tool_call_style=ToolCallFormat.MIXED)

        node_spec = agent_node("custom_agent", main_prompt=main_prompt, config=config)

        assert node_spec.name == "custom_agent"

    def test_agent_with_custom_schemas(self, agent_node):
        """Test agent with custom output schemas."""
        main_prompt = "Analyze the input: {{input}}"

        node_spec = agent_node("schema_agent", main_prompt=main_prompt, output_schema=CustomOutput)

        assert node_spec.name == "schema_agent"

    def test_agent_with_continuation_prompts(self, agent_node):
        """Test agent with continuation prompts."""
        main_prompt = "Start analysis: {{input}}"
        continuation_prompts = {
            "analysis": "Continue analysis: {{reasoning_so_far}}",
            "conclusion": "Draw conclusion: {{reasoning_so_far}}",
        }

        node_spec = agent_node(
            "continuation_agent", main_prompt=main_prompt, continuation_prompts=continuation_prompts
        )

        assert node_spec.name == "continuation_agent"

    def test_agent_with_input_mapping(self, agent_node):
        """Test agent with input mapping."""
        main_prompt = "Analyze: {{user_goal}}"
        input_mapping = {"user_goal": "query.goal"}

        node_spec = agent_node(
            "mapping_agent", main_prompt=main_prompt, input_mapping=input_mapping
        )

        assert node_spec.name == "mapping_agent"
        assert "input_mapping" in node_spec.params

    @pytest.mark.asyncio
    async def test_basic_agent_execution(self, agent_node, mock_llm, mock_tool_router):
        """Test basic agent execution."""
        main_prompt = "Analyze the input: {{input}}"

        node_spec = agent_node("test_execution", main_prompt=main_prompt)

        # Mock the LLM response
        mock_llm.responses = ["I will analyze the input: Test input"]

        input_data = {"input": "Test input"}
        ports = {"llm": mock_llm, "tool_router": mock_tool_router}

        async with ExecutionContext(
            observer_manager=None,
            policy_manager=None,
            run_id="test-run",
            ports=ports,
        ):
            result = await node_spec.fn(input_data)

        # Agent returns state dict
        assert isinstance(result, dict)
        assert "reasoning_steps" in result
        assert "response" in result
        assert "input_data" in result

    @pytest.mark.asyncio
    async def test_agent_with_tool_end(self, agent_node, mock_llm, mock_tool_router):
        """Test agent execution with tool_end call."""
        main_prompt = "Process the input and call tool_end when done: {{input}}"

        node_spec = agent_node("tool_end_test", main_prompt=main_prompt, output_schema=CustomOutput)

        # Mock LLM to call tool_end
        mock_llm.responses = ["INVOKE_TOOL: tool_end(result='analysis complete', confidence=0.95)"]

        input_data = {"input": "Test input"}
        ports = {"llm": mock_llm, "tool_router": mock_tool_router}

        async with ExecutionContext(
            observer_manager=None,
            policy_manager=None,
            run_id="test-run",
            ports=ports,
        ):
            result = await node_spec.fn(input_data)

        # Should return the structured output from tool_end
        assert isinstance(result, CustomOutput)
        assert result.result == "analysis complete"
        assert result.confidence == 0.95
