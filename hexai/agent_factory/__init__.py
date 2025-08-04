"""Pipeline modules for hexAI framework.

This package contains pre-built pipeline applications that demonstrate the capabilities of the hexAI
framework for specific use cases.
"""

from .base import PipelineCatalog, PipelineDefinition, get_catalog
from .models import (  # New ontology models aligned with database schema; Query models
    Ontology,
    OntologyNode,
    OntologyRelation,
    QueryIntent,
    RelationshipType,
    SQLQuery,
)
from .yaml_builder import YamlPipelineBuilder

__all__ = [
    # Agent Factory System
    "PipelineDefinition",
    "PipelineCatalog",
    "get_catalog",
    "YamlPipelineBuilder",
    # Agent Factory Models
    "Ontology",
    "OntologyNode",
    "OntologyRelation",
    # Query Models
    "QueryIntent",
    "SQLQuery",
    "RelationshipType",
]
