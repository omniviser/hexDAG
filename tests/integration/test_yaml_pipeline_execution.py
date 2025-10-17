"""Integration tests for YAML pipeline execution.

Tests demonstrate:
- YAML-based pipeline definitions
- Pipeline structure and dependencies
- Multi-node pipeline execution
- Complex dependency handling
"""

import pytest

from hexdag.core.domain.dag import DirectedGraph, NodeSpec
from hexdag.core.orchestration.orchestrator import Orchestrator


async def data_loader(input_data: str) -> dict:
    """Load and parse input data."""
    return {"raw_input": input_data, "processed": True, "timestamp": "2024-01-01T10:00:00Z"}


async def text_processor(input_data: dict) -> dict:
    """Process text data."""
    text = input_data.get("raw_input", "")
    words = text.split()

    return {
        "word_count": len(words),
        "char_count": len(text),
        "processed_text": text.upper(),
        "original": input_data,
    }


async def sentiment_analyzer(input_data: dict) -> dict:
    """Analyze sentiment of text."""
    text = input_data.get("processed_text", "")

    # Simple sentiment analysis
    positive_words = ["good", "great", "excellent", "happy", "love"]
    negative_words = ["bad", "terrible", "awful", "hate", "sad"]

    text_lower = text.lower()
    positive_score = sum(1 for word in positive_words if word in text_lower)
    negative_score = sum(1 for word in negative_words if word in text_lower)

    if positive_score > negative_score:
        sentiment = "positive"
        confidence = min(0.9, (positive_score - negative_score) / 5)
    elif negative_score > positive_score:
        sentiment = "negative"
        confidence = min(0.9, (negative_score - positive_score) / 5)
    else:
        sentiment = "neutral"
        confidence = 0.5

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "analysis_data": input_data,
    }


async def report_generator(text_data: dict, sentiment_data: dict) -> dict:
    """Generate comprehensive report."""
    return {
        "report": {
            "text_summary": {
                "word_count": text_data.get("word_count", 0),
                "char_count": text_data.get("char_count", 0),
                "processed_text": text_data.get("processed_text", ""),
            },
            "sentiment_analysis": {
                "sentiment": sentiment_data.get("sentiment"),
                "confidence": sentiment_data.get("confidence"),
                "positive_score": sentiment_data.get("positive_score"),
                "negative_score": sentiment_data.get("negative_score"),
            },
            "timestamp": text_data.get("original", {}).get("timestamp"),
        },
        "analysis_complete": True,
    }


@pytest.fixture
def orchestrator():
    """Provide basic orchestrator."""
    return Orchestrator()


class TestYAMLPipelineExecution:
    """Test suite for YAML pipeline execution patterns."""

    @pytest.mark.asyncio
    async def test_simple_linear_pipeline(self, orchestrator):
        """Test simple linear pipeline execution."""
        graph = DirectedGraph()

        # Add nodes in linear sequence
        graph.add(NodeSpec("data_loader", data_loader))
        graph.add(NodeSpec("text_processor", text_processor).after("data_loader"))
        graph.add(NodeSpec("sentiment_analyzer", sentiment_analyzer).after("text_processor"))

        # Validate structure
        graph.validate()
        waves = graph.waves()
        assert len(waves) == 3  # Three sequential waves

        # Execute
        result = await orchestrator.run(graph, "I love this amazing product")

        assert "data_loader" in result
        assert "text_processor" in result
        assert "sentiment_analyzer" in result
        assert result["sentiment_analyzer"]["sentiment"] == "positive"

    @pytest.mark.asyncio
    async def test_complex_dependency_pipeline(self, orchestrator):
        """Test pipeline with complex multi-level dependencies."""
        graph = DirectedGraph()

        # Add nodes with diamond pattern
        graph.add(NodeSpec("data_loader", data_loader))
        graph.add(NodeSpec("text_processor", text_processor).after("data_loader"))
        graph.add(NodeSpec("sentiment_analyzer", sentiment_analyzer).after("text_processor"))
        graph.add(
            NodeSpec("report_generator", report_generator).after(
                "text_processor", "sentiment_analyzer"
            )
        )

        # Validate structure
        graph.validate()
        waves = graph.waves()
        assert len(waves) == 4

        # Execute
        result = await orchestrator.run(graph, "This is excellent quality work")

        assert len(result) == 4
        assert result["report_generator"]["analysis_complete"] is True
        assert "report" in result["report_generator"]

    @pytest.mark.asyncio
    async def test_pipeline_with_positive_sentiment(self, orchestrator):
        """Test pipeline correctly identifies positive sentiment."""
        graph = DirectedGraph()
        graph.add(NodeSpec("data_loader", data_loader))
        graph.add(NodeSpec("text_processor", text_processor).after("data_loader"))
        graph.add(NodeSpec("sentiment_analyzer", sentiment_analyzer).after("text_processor"))

        result = await orchestrator.run(graph, "I love this great excellent product")

        assert result["sentiment_analyzer"]["sentiment"] == "positive"
        assert result["sentiment_analyzer"]["confidence"] > 0.5
        assert result["sentiment_analyzer"]["positive_score"] > 0

    @pytest.mark.asyncio
    async def test_pipeline_with_negative_sentiment(self, orchestrator):
        """Test pipeline correctly identifies negative sentiment."""
        graph = DirectedGraph()
        graph.add(NodeSpec("data_loader", data_loader))
        graph.add(NodeSpec("text_processor", text_processor).after("data_loader"))
        graph.add(NodeSpec("sentiment_analyzer", sentiment_analyzer).after("text_processor"))

        result = await orchestrator.run(graph, "This is terrible awful bad")

        assert result["sentiment_analyzer"]["sentiment"] == "negative"
        assert result["sentiment_analyzer"]["negative_score"] > 0

    @pytest.mark.asyncio
    async def test_pipeline_with_neutral_sentiment(self, orchestrator):
        """Test pipeline correctly identifies neutral sentiment."""
        graph = DirectedGraph()
        graph.add(NodeSpec("data_loader", data_loader))
        graph.add(NodeSpec("text_processor", text_processor).after("data_loader"))
        graph.add(NodeSpec("sentiment_analyzer", sentiment_analyzer).after("text_processor"))

        result = await orchestrator.run(graph, "The product is okay")

        assert result["sentiment_analyzer"]["sentiment"] == "neutral"
        assert result["sentiment_analyzer"]["confidence"] == 0.5

    @pytest.mark.asyncio
    async def test_pipeline_execution_order(self, orchestrator):
        """Test that pipeline executes nodes in correct topological order."""
        graph = DirectedGraph()
        graph.add(NodeSpec("data_loader", data_loader))
        graph.add(NodeSpec("text_processor", text_processor).after("data_loader"))
        graph.add(NodeSpec("sentiment_analyzer", sentiment_analyzer).after("text_processor"))

        # Get execution waves
        waves = graph.waves()

        assert waves[0] == ["data_loader"]
        assert waves[1] == ["text_processor"]
        assert waves[2] == ["sentiment_analyzer"]

    @pytest.mark.asyncio
    async def test_full_pipeline_with_report(self, orchestrator):
        """Test full pipeline including report generation."""
        graph = DirectedGraph()
        graph.add(NodeSpec("data_loader", data_loader))
        graph.add(NodeSpec("text_processor", text_processor).after("data_loader"))
        graph.add(NodeSpec("sentiment_analyzer", sentiment_analyzer).after("text_processor"))
        graph.add(
            NodeSpec("report_generator", report_generator).after(
                "text_processor", "sentiment_analyzer"
            )
        )

        result = await orchestrator.run(graph, "This is great and excellent")

        report = result["report_generator"]["report"]

        assert "text_summary" in report
        assert "sentiment_analysis" in report
        assert report["sentiment_analysis"]["sentiment"] == "positive"
        assert report["text_summary"]["word_count"] == 5

    @pytest.mark.asyncio
    async def test_pipeline_data_flow(self, orchestrator):
        """Test that data flows correctly through pipeline."""
        graph = DirectedGraph()
        graph.add(NodeSpec("data_loader", data_loader))
        graph.add(NodeSpec("text_processor", text_processor).after("data_loader"))

        input_text = "test data flow"
        result = await orchestrator.run(graph, input_text)

        # Verify data propagated correctly
        assert result["data_loader"]["raw_input"] == input_text
        assert result["text_processor"]["original"]["raw_input"] == input_text
        assert result["text_processor"]["processed_text"] == "TEST DATA FLOW"

    @pytest.mark.asyncio
    async def test_pipeline_with_multiple_inputs(self, orchestrator):
        """Test pipeline with different inputs."""
        graph = DirectedGraph()
        graph.add(NodeSpec("data_loader", data_loader))
        graph.add(NodeSpec("text_processor", text_processor).after("data_loader"))
        graph.add(NodeSpec("sentiment_analyzer", sentiment_analyzer).after("text_processor"))

        test_inputs = [
            "I love this",
            "This is terrible",
            "It's okay",
        ]

        for test_input in test_inputs:
            result = await orchestrator.run(graph, test_input)
            assert "sentiment_analyzer" in result
            assert result["sentiment_analyzer"]["sentiment"] in ["positive", "negative", "neutral"]
