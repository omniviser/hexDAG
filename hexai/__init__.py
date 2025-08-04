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

# Agent Factory system exports
from hexai.agent_factory import (
    Ontology,
    OntologyNode,
    OntologyRelation,
    PipelineCatalog,
    PipelineDefinition,
    QueryIntent,
    RelationshipType,
    SQLQuery,
    get_catalog,
)
from hexai.agent_factory.yaml_builder import YamlPipelineBuilder

# Core framework exports
from hexai.core.application.nodes import (
    ConditionalNode,
    FunctionNode,
    LLMNode,
    LoopNode,
    ReActAgentNode,
)
from hexai.core.application.orchestrator import Orchestrator
from hexai.core.application.prompt import FewShotPromptTemplate, PromptTemplate
from hexai.core.domain import DirectedGraph, NodeSpec

# Port interfaces
from hexai.core.ports import LLM, DatabasePort, LongTermMemory, OntologyPort, ToolRouter

__all__ = [
    # Core Framework - DAG Building and Execution
    "Orchestrator",
    "DirectedGraph",
    "NodeSpec",
    "YamlPipelineBuilder",
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
    # Agent Factory System
    "PipelineDefinition",
    "PipelineCatalog",
    "get_catalog",
    # Agent Factory Models
    "Ontology",
    "OntologyNode",
    "OntologyRelation",
    "QueryIntent",
    "SQLQuery",
    "RelationshipType",
]
