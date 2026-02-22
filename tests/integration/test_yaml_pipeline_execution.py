"""Integration tests for YAML pipeline execution.

Tests demonstrate:
- YAML-based pipeline definitions
- Pipeline structure and dependencies
- Multi-node pipeline execution
- Complex dependency handling
"""

import pytest

from hexdag.kernel.domain.dag import DirectedGraph, NodeSpec
from hexdag.kernel.orchestration.orchestrator import Orchestrator


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


async def report_generator(inputs: dict) -> dict:
    """Generate comprehensive report."""
    text_data = inputs.get("text_processor", {})
    sentiment_data = inputs.get("sentiment_analyzer", {})
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


class TestYAMLPipelineWithInputMapping:
    """Test YAML pipelines with $input and input_mapping features."""

    @pytest.mark.asyncio
    async def test_input_mapping_preserved_through_yaml_builder(self):
        """Test that input_mapping from YAML is correctly stored in node params."""
        from hexdag.kernel.pipeline_builder.yaml_builder import YamlPipelineBuilder

        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: mapping-preservation-test
spec:
  nodes:
    - kind: function_node
      metadata:
        name: my_node
      spec:
        fn: json.loads
        input_mapping:
          field_a: $input.source_a
          field_b: other_node.output
"""
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)

        # Verify input_mapping is in node params
        node = graph.nodes["my_node"]
        assert "input_mapping" in node.params
        assert node.params["input_mapping"] == {
            "field_a": "$input.source_a",
            "field_b": "other_node.output",
        }

    @pytest.mark.asyncio
    async def test_execution_coordinator_applies_input_mapping(self):
        """Test that ExecutionCoordinator correctly applies $input mappings at runtime."""
        from hexdag.kernel.domain.dag import NodeSpec
        from hexdag.kernel.orchestration.components.execution_coordinator import (
            ExecutionCoordinator,
        )

        coordinator = ExecutionCoordinator()

        # Create a node with input_mapping in params
        node_spec = NodeSpec(
            "consumer",
            lambda x: x,
            deps={"producer"},
            params={
                "input_mapping": {
                    "load_id": "$input.load_id",
                    "carrier_mc": "$input.carrier_mc",
                    "producer_result": "producer.output",
                }
            },
        )

        initial_input = {"load_id": "LOAD123", "carrier_mc": "MC456", "extra": "ignored"}
        node_results = {"producer": {"output": "processed_data", "status": "ok"}}

        # This is what happens during execution
        result = coordinator.prepare_node_input(node_spec, node_results, initial_input)

        # Verify $input.* fields were extracted from initial_input
        assert result["load_id"] == "LOAD123"
        assert result["carrier_mc"] == "MC456"
        # Verify dependency.field was extracted from node results
        assert result["producer_result"] == "processed_data"

    @pytest.mark.asyncio
    async def test_full_yaml_pipeline_with_input_mapping_end_to_end(self, orchestrator):
        """Test full pipeline execution with input_mapping using real functions."""
        from hexdag.kernel.domain.dag import DirectedGraph, NodeSpec

        # Create a simple pipeline manually that tests the input_mapping flow
        async def identity(x):
            """Just return the input."""
            return {"result": x}

        async def consumer(x):
            """Consumer that expects mapped input."""
            return {"consumed": x}

        graph = DirectedGraph()
        graph.add(NodeSpec("producer", identity))
        graph.add(
            NodeSpec(
                "consumer",
                consumer,
                deps={"producer"},
                params={
                    "input_mapping": {
                        "original_id": "$input.request_id",
                        "produced": "producer.result",
                    }
                },
            )
        )

        # Run with dict initial input
        initial_input = {"request_id": "REQ001", "extra": "data"}
        result = await orchestrator.run(graph, initial_input)

        # Verify producer got initial input
        assert "producer" in result
        # Verify consumer got mapped input
        assert "consumer" in result
        # The consumer received the mapped input dict
        assert result["consumer"]["consumed"]["original_id"] == "REQ001"
        assert result["consumer"]["consumed"]["produced"] == {
            "request_id": "REQ001",
            "extra": "data",
        }
