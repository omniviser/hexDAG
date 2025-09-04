"""Hex-DAG Agent Framework.

A modular, deterministic, and extensible architecture for orchestrating LLM-powered agents alongside
traditional code with YAML pipeline configuration.
"""

# Adapter exports for testing and development
from hexai.adapters import InMemoryMemory
from hexai.adapters.mock import MockDatabaseAdapter, MockEmbeddingSelectorPort, MockLLM

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

# Port interfaces xd
from hexai.core.ports import LLM, DatabasePort, LongTermMemory, OntologyPort, ToolRouter

# Initialize the component registry early so it's available for all imports
# This ensures core components are loaded and plugins are discovered
from hexai.core.registry import registry
from hexai.core.registry.types import ComponentType  # For internal framework use

# Load core components - using setattr to avoid mypy issues
setattr(registry, "_core_loading", True)  # Allow core namespace registration
import hexai.core.nodes  # noqa: F401, E402 - triggers decorator registration

setattr(registry, "_core_loading", False)  # Block core namespace registration


__all__ = [
    # Registry System
    "registry",
    "ComponentType",
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
    "MockLLM",
    "MockDatabaseAdapter",
    "MockEmbeddingSelectorPort",
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
