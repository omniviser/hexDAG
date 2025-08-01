"""Ontology analysis pipeline implementation."""

from ..base import PipelineDefinition
from .functions import (
    load_ontology_context,
    load_ontology_data,
    metadata_resolver,
    ontology_analyzer,
    query_matcher,
    result_formatter,
)


class OntologyPipeline(PipelineDefinition):
    """Pipeline for business ontology analysis and validation."""

    @property
    def name(self) -> str:
        """Pipeline name."""
        return "ontology_pipeline"

    @property
    def description(self) -> str:
        """Pipeline description."""
        return "Business ontology analysis and validation with path finding"

    def _register_functions(self) -> None:
        """Register pipeline functions."""
        self.builder.register_function("load_ontology_context", load_ontology_context)
        self.builder.register_function("load_ontology_data", load_ontology_data)
        self.builder.register_function("metadata_resolver", metadata_resolver)
        self.builder.register_function("ontology_analyzer", ontology_analyzer)
        self.builder.register_function("query_matcher", query_matcher)
        self.builder.register_function("result_formatter", result_formatter)
