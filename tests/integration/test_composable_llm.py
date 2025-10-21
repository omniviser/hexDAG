"""Integration tests for composable LLM architecture.

Tests the new architecture: PromptNode → RawLLMNode → ParserNode
"""

import pytest
from pydantic import BaseModel

from hexdag.builtin.adapters.mock.mock_llm import MockLLM
from hexdag.builtin.macros import LLMMacro
from hexdag.builtin.nodes import ParserNode, PromptNode, RawLLMNode
from hexdag.core.domain.dag import DirectedGraph
from hexdag.core.orchestration.orchestrator import Orchestrator
from hexdag.core.orchestration.prompt.template import PromptTemplate


class AnalysisResult(BaseModel):
    """Test output schema."""

    summary: str
    sentiment: str
    confidence: float


@pytest.mark.asyncio
class TestComposableLLM:
    """Test composable LLM architecture."""

    async def test_manual_composition_text_only(self):
        """Test manual composition: PromptNode → RawLLMNode (text output)."""
        # Setup
        mock_llm = MockLLM(responses=["The answer is 42."])

        # Build graph manually
        graph = DirectedGraph()

        # Node 1: Build prompt
        prompt_node = PromptNode()
        prompt_spec = prompt_node(
            name="prompt",
            template="Answer this question: {{question}}",
            output_format="messages",
        )
        graph += prompt_spec

        # Node 2: Call LLM
        llm_node = RawLLMNode()
        llm_spec = llm_node(name="llm", deps=["prompt"])
        graph += llm_spec

        # Execute
        orchestrator = Orchestrator(ports={"llm": mock_llm})
        result = await orchestrator.run(graph, {"question": "What is the meaning of life?"})

        # Verify
        assert "llm" in result
        assert result["llm"].text == "The answer is 42."

    async def test_manual_composition_with_parser(self):
        """Test full manual composition: PromptNode → RawLLMNode → ParserNode."""
        # Setup - mock LLM returns JSON
        mock_response = '{"summary": "Very positive", "sentiment": "positive", "confidence": 0.95}'
        mock_llm = MockLLM(responses=[mock_response])

        # Build graph manually
        graph = DirectedGraph()

        # Node 1: Build prompt
        prompt_node = PromptNode()
        prompt_spec = prompt_node(
            name="prompt",
            template=(
                "Analyze the sentiment of: {{text}}. "
                "Return JSON with summary, sentiment, and confidence."
            ),
            output_format="messages",
        )
        graph += prompt_spec

        # Node 2: Call LLM
        llm_node = RawLLMNode()
        llm_spec = llm_node(name="llm", deps=["prompt"])
        graph += llm_spec

        # Node 3: Parse output
        parser_node = ParserNode()
        parser_spec = parser_node(
            name="parser", output_schema=AnalysisResult, strategy="json", deps=["llm"]
        )
        graph += parser_spec

        # Execute
        orchestrator = Orchestrator(ports={"llm": mock_llm})
        result = await orchestrator.run(graph, {"text": "I love this product!"})

        # Verify
        assert "parser" in result
        parsed_result = result["parser"]
        assert isinstance(parsed_result, AnalysisResult)
        assert parsed_result.summary == "Very positive"
        assert parsed_result.sentiment == "positive"
        assert parsed_result.confidence == 0.95

    async def test_llm_macro_text_only(self):
        """Test LLM macro without parsing."""
        # Setup
        mock_llm = MockLLM(responses=["Quantum computing uses quantum mechanics for computation."])

        # Create macro
        macro = LLMMacro(template="Explain {{topic}} in one sentence")

        # Expand macro into graph
        graph = macro.expand(
            instance_name="explainer", inputs={"topic": "quantum computing"}, dependencies=[]
        )

        # Execute
        orchestrator = Orchestrator(ports={"llm": mock_llm})
        result = await orchestrator.run(graph, {"topic": "quantum computing"})

        # Verify - graph should have 2 nodes (prompt + llm, no parser)
        assert len(graph.nodes) == 2
        assert "explainer_llm" in result
        assert "quantum mechanics" in result["explainer_llm"].text

    async def test_llm_macro_with_parsing(self):
        """Test LLM macro with structured output parsing."""
        # Setup - mock LLM returns JSON
        mock_response = """
        {
            "summary": "Quantum computing is revolutionary",
            "sentiment": "positive",
            "confidence": 0.9
        }
        """
        mock_llm = MockLLM(responses=[mock_response])

        # Create macro with output schema
        macro = LLMMacro(
            template="Analyze {{text}} and return JSON with summary, sentiment, and confidence.",
            output_schema=AnalysisResult,
            parse_strategy="json",
        )

        # Expand macro into graph
        graph = macro.expand(instance_name="analyzer", inputs={"text": "..."}, dependencies=[])

        # Verify graph structure
        assert len(graph.nodes) == 3  # prompt + llm + parser

        # Execute
        orchestrator = Orchestrator(ports={"llm": mock_llm})
        result = await orchestrator.run(graph, {"text": "Quantum computing is amazing!"})

        # Verify
        assert "analyzer_parser" in result
        parsed_result = result["analyzer_parser"]
        assert isinstance(parsed_result, AnalysisResult)
        assert parsed_result.summary == "Quantum computing is revolutionary"
        assert parsed_result.sentiment == "positive"

    async def test_composed_prompts(self):
        """Test using composed prompts with + operator."""
        # Setup
        mock_llm = MockLLM(responses=["Analysis complete with high confidence."])

        # Create composed prompt
        base = PromptTemplate("Analyze {{data}}")
        instructions = PromptTemplate("\n\nProvide detailed analysis with confidence scores.")
        composed = base + instructions

        # Build graph with composed prompt
        graph = DirectedGraph()

        prompt_node = PromptNode()
        prompt_spec = prompt_node(name="prompt", template=composed, output_format="messages")
        graph += prompt_spec

        llm_node = RawLLMNode()
        llm_spec = llm_node(name="llm", deps=["prompt"])
        graph += llm_spec

        # Execute
        orchestrator = Orchestrator(ports={"llm": mock_llm})
        result = await orchestrator.run(graph, {"data": "sales report"})

        # Verify
        assert "llm" in result
        assert "confidence" in result["llm"].text

    async def test_raw_llm_node_with_text_input(self):
        """Test RawLLMNode accepts plain text input."""
        # Setup
        mock_llm = MockLLM(responses=["Hello there!"])

        # Create single RawLLMNode
        graph = DirectedGraph()
        llm_node = RawLLMNode()
        llm_spec = llm_node(name="llm")
        graph += llm_spec

        # Execute with text input (will be converted to messages internally)
        orchestrator = Orchestrator(ports={"llm": mock_llm})
        result = await orchestrator.run(graph, {"text": "Hello!"})

        # Verify
        assert result["llm"].text == "Hello there!"

    async def test_raw_llm_node_with_messages_input(self):
        """Test RawLLMNode accepts messages input directly."""
        # Setup
        mock_llm = MockLLM(responses=["I'm an assistant!"])

        # Create single RawLLMNode
        graph = DirectedGraph()
        llm_node = RawLLMNode()
        llm_spec = llm_node(name="llm")
        graph += llm_spec

        # Execute with messages input
        orchestrator = Orchestrator(ports={"llm": mock_llm})
        result = await orchestrator.run(
            graph,
            {
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Who are you?"},
                ]
            },
        )

        # Verify
        assert result["llm"].text == "I'm an assistant!"
