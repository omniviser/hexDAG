"""Test cases for LLMNode class."""

import pytest
from pydantic import BaseModel

from hexai.adapters.mock.mock_llm import MockLLM
from hexai.core.application.prompt import ChatPromptTemplate, PromptTemplate
from hexai.core.bootstrap import ensure_bootstrapped
from hexai.core.registry import registry

# Ensure registry is bootstrapped for tests
ensure_bootstrapped()


class OutputSchema(BaseModel):
    """Test output schema."""

    result: str
    confidence: float


class TestLLMNode:
    """Test cases for LLMNode class."""

    @pytest.fixture
    def llm_node(self):
        """Get LLMNode factory from registry."""
        ensure_bootstrapped()
        return registry.get("llm_node", namespace="core")

    @pytest.fixture
    def mock_llm(self):
        """Fixture for mock LLM."""
        return MockLLM()

    def test_direct_call_with_string_template_no_schema(self, llm_node):
        """Test direct call with string template (simple mode)."""
        node_spec = llm_node("simple_llm", "Process: {{data}}")

        assert node_spec.name == "simple_llm"
        assert node_spec.fn.__name__ == "llm_wrapper"
        assert node_spec.in_type is not None
        assert node_spec.out_type is str  # No rich features for string template

    def test_string_template_with_output_schema_rejected(self, llm_node):
        """Test that string templates with output schema are rejected."""
        with pytest.raises(ValueError, match="output_schema not supported for string templates"):
            llm_node("invalid", "Process: {{data}}", output_schema=OutputSchema)

    def test_from_template_with_prompt_template_and_schema(self, llm_node):
        """Test from_template with PromptTemplate and output schema (rich mode)."""
        template = PromptTemplate("Analyze: {{data}}")
        node_spec = llm_node.from_template("rich_llm", template, OutputSchema)

        assert node_spec.name == "rich_llm"
        assert node_spec.out_type == OutputSchema  # Rich features enabled

    def test_from_template_with_chat_prompt_template(self, llm_node):
        """Test from_template with ChatPromptTemplate."""
        chat_template = ChatPromptTemplate(human_message="Process: {{input}}")
        node_spec = llm_node.from_template("chat_llm", chat_template)

        assert node_spec.name == "chat_llm"
        assert node_spec.fn.__name__ == "llm_wrapper"

    @pytest.mark.asyncio
    async def test_simple_execution(self, llm_node, mock_llm):
        """Test simple LLM execution without output schema."""
        node_spec = llm_node("test_simple", "Process: {{data}}")

        mock_llm.responses = ["Processed data: test"]
        input_data = {"data": "test"}
        ports = {"llm": mock_llm}

        result = await node_spec.fn(input_data, **ports)

        assert result == "Processed data: test"
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_rich_execution_with_schema(self, llm_node, mock_llm):
        """Test rich LLM execution with output schema."""
        template = PromptTemplate("Analyze: {{data}}")
        node_spec = llm_node.from_template("test_rich", template, OutputSchema)

        # Mock JSON response
        mock_llm.responses = ['{"result": "analysis complete", "confidence": 0.95}']
        input_data = {"data": "test"}
        ports = {"llm": mock_llm}

        result = await node_spec.fn(input_data, **ports)

        assert isinstance(result, OutputSchema)
        assert result.result == "analysis complete"
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_missing_llm_port_error(self, llm_node):
        """Test error when LLM port is missing."""
        node_spec = llm_node("test_error", "Process: {{data}}")

        input_data = {"data": "test"}
        ports = {}  # No LLM port

        with pytest.raises(ValueError, match="LLM port is required"):
            await node_spec.fn(input_data, **ports)

    def test_input_mapping_support(self, llm_node):
        """Test LLM node with input mapping."""
        node_spec = llm_node(
            "mapped_llm",
            "Process: {{content}}",
            input_mapping={"content": "source.text"},
        )

        assert "input_mapping" in node_spec.params
        assert node_spec.params["input_mapping"] == {"content": "source.text"}

    def test_dependencies_handling(self, llm_node):
        """Test LLM node with dependencies."""
        node_spec = llm_node("dependent_llm", "Process: {{data}}", deps=["source_node"])

        assert "source_node" in node_spec.deps
