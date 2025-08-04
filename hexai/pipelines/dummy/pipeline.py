"""Dummy pipeline implementation for demonstrating hexAI framework capabilities."""

from ..base import PipelineDefinition
from .functions import calculate_score, extract_features, generate_summary, validate_input


class DummyPipeline(PipelineDefinition):
    """Pipeline for demonstrating hexAI framework capabilities with text analysis."""

    name = "dummy_pipeline"
    description = "Demonstrates hexAI framework with text quality analysis workflow"

    def _register_functions(self) -> None:
        """Register pipeline functions."""
        self.builder.register_function("validate_input", validate_input)
        self.builder.register_function("extract_features", extract_features)
        self.builder.register_function("calculate_score", calculate_score)
        self.builder.register_function("generate_summary", generate_summary)
