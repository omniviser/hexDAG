"""Hex-DAG Agent Framework.

A modular, deterministic, and extensible architecture for orchestrating LLM-powered agents alongside
traditional code with YAML pipeline configuration.
"""

# Adapter exports for testing and development
from hexai.adapters import InMemoryMemory, LLMFactoryAdapter
from hexai.adapters.enhanced_database import EnhancedDatabaseAdapter
from hexai.adapters.mock import (
    MockDatabaseAdapter,
    MockEmbeddingSelectorPort,
    MockLLM,
    MockOntologyPort,
)

# Core framework exports
from hexai.app.application.nodes import (
    ConditionalNode,
    FunctionNode,
    LLMNode,
    LoopNode,
    ReActAgentNode,
)
from hexai.app.application.orchestrator import Orchestrator
from hexai.app.application.pipeline_builder import PipelineBuilder
from hexai.app.application.prompt import FewShotPromptTemplate, PromptTemplate
from hexai.app.domain import DirectedGraph, NodeSpec

# Port interfaces
from hexai.app.ports import LLM, DatabasePort, LongTermMemory, OntologyPort, ToolRouter

# Pipeline system exports
from hexai.pipelines import (
    PipelineDefinition,
    PipelineCatalog,
    get_catalog,
    Ontology,
    OntologyNode,
    OntologyRelation,
    QueryIntent,
    SQLQuery,
    RelationshipType,
)

__all__ = [
    # Core Framework - DAG Building and Execution
    "Orchestrator",
    "DirectedGraph",
    "NodeSpec",
    "PipelineBuilder",
    # Node Factories
    "FunctionNode",
    "LLMNode",
    "ReActAgentNode",
    "LoopNode",
    "ConditionalNode",
    # Templating System
    "PromptTemplate",
    "FewShotPromptTemplate",
    # Port Interfaces
    "LLM",
    "LongTermMemory",
    "ToolRouter",
    "DatabasePort",
    "OntologyPort",
    # Testing and Development Adapters
    "InMemoryMemory",
    "LLMFactoryAdapter",
    "MockLLM",
    "MockDatabaseAdapter",
    "MockOntologyPort",
    "MockEmbeddingSelectorPort",
    # Enhanced Adapters
    "EnhancedDatabaseAdapter",
    # Pipeline System
    "PipelineDefinition",
    "PipelineCatalog",
    "get_catalog",
    # Pipeline Models
    "Ontology",
    "OntologyNode", 
    "OntologyRelation",
    "QueryIntent",
    "SQLQuery",
    "RelationshipType",
]
