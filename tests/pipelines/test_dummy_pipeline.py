"""Tests for the dummy pipeline."""

from pydantic import ValidationError
import pytest

from hexai.pipelines.dummy import DummyPipeline
from hexai.pipelines.dummy.functions import (
    DummyPipelineInput,
    FeaturesOutput,
    ScoreOutput,
    SummaryOutput,
    ValidatedOutput,
    calculate_score,
    extract_features,
    generate_summary,
    validate_input,
)


class MockEventManager:
    """Mock event manager for testing."""

    def __init__(self):
        """Initialize the mock event manager."""
        self.traces = []
        self.memory = {}

    def add_trace(self, node_name: str, message: str):
        """Add a trace message."""
        self.traces.append(f"{node_name}: {message}")

    def set_memory(self, key: str, value):
        """Set a memory value."""
        self.memory[key] = value

    def get_memory(self, key: str, default=None):
        """Get a memory value."""
        return self.memory.get(key, default)


class TestDummyPipeline:
    """Tests for DummyPipeline."""

    def test_pipeline_attributes(self):
        """Test pipeline attributes."""
        pipeline = DummyPipeline()

        assert pipeline.name == "dummy_pipeline"
        assert (
            pipeline.description
            == "Demonstrates hexAI framework with text quality analysis workflow"
        )

        # Check all functions are registered
        expected_functions = [
            "validate_input",
            "extract_features",
            "calculate_score",
            "generate_summary",
        ]
        for func_name in expected_functions:
            assert func_name in pipeline.builder.registered_functions

    def test_pipeline_config(self):
        """Test pipeline configuration."""
        pipeline = DummyPipeline()
        config = pipeline.get_config()

        assert config is not None
        assert "nodes" in config
        assert len(config["nodes"]) == 7  # 7 nodes in the current pipeline

        # Check node configurations
        nodes = config["nodes"]
        node_ids = [node["id"] for node in nodes]
        expected_nodes = [
            "input_validator",
            "sentiment_analyzer",
            "feature_extractor",
            "quality_advisor",
            "score_calculator",
            "insights_generator",
            "summary_generator",
        ]

        for expected_node in expected_nodes:
            assert expected_node in node_ids

        # Check specific node configurations
        validator_node = next(n for n in nodes if n["id"] == "input_validator")
        assert validator_node["type"] == "function"
        assert validator_node["params"]["fn"] == "validate_input"
        assert validator_node["depends_on"] == []

        llm_node = next(n for n in nodes if n["id"] == "sentiment_analyzer")
        assert llm_node["type"] == "llm"
        assert "prompt_template" in llm_node["params"]

    def test_function_registration(self):
        """Test that all required functions are registered."""
        pipeline = DummyPipeline()

        # Check that all functions are registered
        expected_functions = [
            "validate_input",
            "extract_features",
            "calculate_score",
            "generate_summary",
        ]
        for func_name in expected_functions:
            assert func_name in pipeline.builder.registered_functions
            func = pipeline.builder.registered_functions[func_name]
            assert callable(func)


class TestDummyPipelineFunctions:
    """Tests for individual dummy pipeline functions."""

    @pytest.mark.asyncio
    async def test_validate_input_success(self):
        """Test successful input validation."""
        event_manager = MockEventManager()
        input_data = DummyPipelineInput(
            text="This is a valid text input for testing",
            priority=5,
            metadata={"source": "test"},
        )

        result = await validate_input(input_data, event_manager=event_manager)

        assert isinstance(result, ValidatedOutput)
        assert result.text == "This is a valid text input for testing"
        assert result.priority == 5
        assert result.metadata == {"source": "test"}
        assert result.validated is True
        assert len(event_manager.traces) > 0
        assert any("validate_input" in trace for trace in event_manager.traces)

    @pytest.mark.asyncio
    async def test_validate_input_invalid_data_type(self):
        """Test validation with invalid data type."""
        # This test should now check Pydantic validation errors instead
        with pytest.raises(ValidationError):
            # Use type: ignore to bypass linter for intentionally invalid data
            DummyPipelineInput(text="valid", priority="invalid")  # type: ignore

    @pytest.mark.asyncio
    async def test_validate_input_short_text(self):
        """Test validation with text too short."""
        event_manager = MockEventManager()
        input_data = DummyPipelineInput(text="Hi", priority=5)

        with pytest.raises(ValueError, match="Text must be at least 5 characters long"):
            await validate_input(input_data, event_manager=event_manager)

    @pytest.mark.asyncio
    async def test_validate_input_invalid_priority(self):
        """Test validation with invalid priority."""
        event_manager = MockEventManager()
        input_data = DummyPipelineInput(text="Valid text here", priority=15)

        with pytest.raises(ValueError, match="Priority must be between 1 and 10"):
            await validate_input(input_data, event_manager=event_manager)

    @pytest.mark.asyncio
    async def test_validate_input_defaults(self):
        """Test validation with default values."""
        event_manager = MockEventManager()
        input_data = DummyPipelineInput(text="Valid text here")

        result = await validate_input(input_data, event_manager=event_manager)

        assert isinstance(result, ValidatedOutput)
        assert result.text == "Valid text here"
        assert result.priority == 1  # Default value
        assert result.metadata == {}  # Default value
        assert result.validated is True

    @pytest.mark.asyncio
    async def test_extract_features_success(self):
        """Test successful feature extraction."""
        event_manager = MockEventManager()
        input_data = ValidatedOutput(
            text="This is a great test with 123 numbers and UPPERCASE letters.",
            priority=5,
            metadata={"source": "test"},
            validated=True,
        )

        result = await extract_features(input_data, event_manager=event_manager)

        assert isinstance(result, FeaturesOutput)
        assert result.word_count == 11
        assert result.char_count == 60
        assert result.sentence_count == 1
        assert result.has_numbers is True
        assert result.has_uppercase is True
        assert result.avg_word_length > 0
        assert result.original_data == input_data.model_dump()
        assert len(event_manager.traces) > 0

    @pytest.mark.asyncio
    async def test_extract_features_empty_text(self):
        """Test feature extraction with empty text."""
        event_manager = MockEventManager()
        input_data = ValidatedOutput(
            text="",
            priority=1,
            metadata={},
            validated=True,
        )

        result = await extract_features(input_data, event_manager=event_manager)

        assert isinstance(result, FeaturesOutput)
        assert result.word_count == 0
        assert result.char_count == 0
        assert result.sentence_count == 0
        assert result.has_numbers is False
        assert result.has_uppercase is False
        assert result.avg_word_length == 0

    @pytest.mark.asyncio
    async def test_calculate_score_high_quality(self):
        """Test score calculation for high-quality text."""
        event_manager = MockEventManager()
        input_data = FeaturesOutput(
            word_count=15,  # 25 points
            char_count=80,  # 20 points
            sentence_count=3,  # 20 points
            has_numbers=True,  # 5 points
            has_uppercase=True,  # 5 points
            avg_word_length=5.0,  # 5 points + 20 points for readability
            original_data={},
        )

        result = await calculate_score(input_data, event_manager=event_manager)

        assert isinstance(result, ScoreOutput)
        assert result.quality_score == 100.0  # Perfect score
        assert "word_count" in result.score_breakdown
        assert "char_count" in result.score_breakdown
        assert "sentence_structure" in result.score_breakdown
        assert "content_variety" in result.score_breakdown
        assert "readability" in result.score_breakdown
        assert result.features == input_data.model_dump()
        assert len(event_manager.traces) > 0

    @pytest.mark.asyncio
    async def test_calculate_score_low_quality(self):
        """Test score calculation for low-quality text."""
        event_manager = MockEventManager()
        input_data = FeaturesOutput(
            word_count=2,  # 0 points
            char_count=10,  # 0 points
            sentence_count=0,  # 0 points
            has_numbers=False,  # 0 points
            has_uppercase=False,  # 0 points
            avg_word_length=1.5,  # 0 points + 0 points for readability
            original_data={},
        )

        result = await calculate_score(input_data, event_manager=event_manager)

        assert isinstance(result, ScoreOutput)
        assert result.quality_score == 0.0  # Minimum score
        assert result.score_breakdown["word_count"] == 0
        assert result.score_breakdown["char_count"] == 0
        assert result.score_breakdown["sentence_structure"] == 0
        assert result.score_breakdown["content_variety"] == 0
        assert result.score_breakdown["readability"] == 0

    @pytest.mark.asyncio
    async def test_generate_summary_high_score(self):
        """Test summary generation for high-quality score."""
        event_manager = MockEventManager()
        input_data = ScoreOutput(
            quality_score=85.0,
            score_breakdown={
                "word_count": 25,
                "char_count": 20,
                "sentence_structure": 20,
                "content_variety": 10,
                "readability": 10,
            },
            features={},
        )

        result = await generate_summary(input_data, event_manager=event_manager)

        assert isinstance(result, SummaryOutput)
        assert result.analysis_summary["overall_score"] == 85.0
        assert result.analysis_summary["score_category"] == "Excellent"
        assert len(result.analysis_summary["strengths"]) > 0
        assert len(result.recommendations) >= 0  # May be empty for high scores
        assert result.score_breakdown == input_data.score_breakdown
        assert result.metadata["node_name"] == "generate_summary"
        assert result.metadata["quality_score"] == 85.0
        assert len(event_manager.traces) > 0

    @pytest.mark.asyncio
    async def test_generate_summary_low_score(self):
        """Test summary generation for low-quality score."""
        event_manager = MockEventManager()
        input_data = ScoreOutput(
            quality_score=25.0,
            score_breakdown={
                "word_count": 0,
                "char_count": 5,
                "sentence_structure": 10,
                "content_variety": 5,
                "readability": 5,
            },
            features={},
        )

        result = await generate_summary(input_data, event_manager=event_manager)

        assert isinstance(result, SummaryOutput)
        assert result.analysis_summary["overall_score"] == 25.0
        assert result.analysis_summary["score_category"] == "Poor"
        assert len(result.analysis_summary["weaknesses"]) > 0
        assert len(result.recommendations) > 0  # Should have recommendations for low scores
        assert "Consider adding more content" in result.recommendations[0]

    @pytest.mark.asyncio
    async def test_generate_summary_medium_score(self):
        """Test summary generation for medium-quality score."""
        event_manager = MockEventManager()
        input_data = ScoreOutput(
            quality_score=65.0,
            score_breakdown={
                "word_count": 15,
                "char_count": 5,  # Below 10 threshold, should be a weakness
                "sentence_structure": 20,
                "content_variety": 8,  # Below 10 threshold, should be a weakness
                "readability": 10,
            },
            features={},
        )

        result = await generate_summary(input_data, event_manager=event_manager)

        assert isinstance(result, SummaryOutput)
        assert result.analysis_summary["overall_score"] == 65.0
        assert result.analysis_summary["score_category"] == "Good"
        assert len(result.analysis_summary["strengths"]) > 0
        assert len(result.analysis_summary["weaknesses"]) > 0

    @pytest.mark.asyncio
    async def test_function_pipeline_integration(self):
        """Test that functions work together in pipeline order."""
        event_manager = MockEventManager()

        # Step 1: Validate input
        input_data = DummyPipelineInput(
            text="This is a comprehensive test to verify pipeline integration.",
            priority=7,
            metadata={"test": "integration"},
        )
        validated = await validate_input(input_data, event_manager=event_manager)

        # Step 2: Extract features
        features = await extract_features(validated, event_manager=event_manager)

        # Step 3: Calculate score
        score = await calculate_score(features, event_manager=event_manager)

        # Step 4: Generate summary
        summary = await generate_summary(score, event_manager=event_manager)

        # Verify the pipeline flow
        assert isinstance(validated, ValidatedOutput)
        assert isinstance(features, FeaturesOutput)
        assert isinstance(score, ScoreOutput)
        assert isinstance(summary, SummaryOutput)

        # Verify data flow
        assert validated.text == input_data.text
        assert features.original_data == validated.model_dump()
        assert score.features == features.model_dump()
        assert summary.score_breakdown == score.score_breakdown

        # Verify event manager usage
        assert len(event_manager.traces) >= 4  # At least one trace per function

    @pytest.mark.asyncio
    async def test_event_manager_integration(self):
        """Test that all functions properly use event manager."""
        event_manager = MockEventManager()

        # Test each function with event manager
        input_data = DummyPipelineInput(text="Test event manager integration", priority=5)

        await validate_input(input_data, event_manager=event_manager)

        # Check that traces were added
        assert len(event_manager.traces) > 0
        assert any("validate_input" in trace for trace in event_manager.traces)
        assert any("Validating input data" in trace for trace in event_manager.traces)

    @pytest.mark.asyncio
    async def test_function_without_event_manager(self):
        """Test that functions work without event manager."""
        input_data = DummyPipelineInput(text="Test without event manager", priority=3)

        # Should not raise an error
        result = await validate_input(input_data)

        assert isinstance(result, ValidatedOutput)
        assert result.text == "Test without event manager"
        assert result.priority == 3
