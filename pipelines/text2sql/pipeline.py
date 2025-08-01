"""Text-to-SQL Pipeline.

This pipeline converts natural language questions into SQL queries using LLM agents and tools.
"""

from pydantic import BaseModel, Field

from pipelines.base import PipelineDefinition


class Text2SQLPipelineInput(BaseModel):
    """Input model for the Text-to-SQL pipeline."""

    question: str = Field(..., description="Natural language question to convert to SQL")
    database_name: str = Field(..., description="Name of the database to query")


class Text2SQLPipeline(PipelineDefinition):
    """Text-to-SQL Pipeline for intelligent SQL generation.

    This pipeline uses a reasoning agent with tools to convert natural language questions into
    accurate SQL queries by exploring database schemas and understanding the question requirements.
    """

    @property
    def name(self) -> str:
        """Pipeline name."""
        return "text2sql_pipeline"

    @property
    def description(self) -> str:
        """Pipeline description."""
        return "Intelligent Text-to-SQL generation with agent-based reasoning and tool usage"

    def _register_functions(self) -> None:
        """Register functions for this pipeline."""
        # Register the tool router as a service
        # Note: In practice, this would be injected through ports
        pass

    def get_input_type(self) -> type:
        """Get the input type for this pipeline."""
        return Text2SQLPipelineInput
