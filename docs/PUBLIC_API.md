# PUBLIC API for hexAI 🚀

This document lists all officially supported public symbols of the `hexai` package.
Anything **not listed here** is considered internal and may change without notice.

---

## Core Framework
- `Orchestrator`
- `DirectedGraph`
- `NodeSpec`
- `YamlPipelineBuilder`

## Node Factories
- `FunctionNode`
- `LLMNode`
- `ReActAgentNode`
- `LoopNode`
- `ConditionalNode`

## Prompting
- `PromptTemplate`
- `FewShotPromptTemplate`

## Port Interfaces
- `LLM`
- `LongTermMemory`
- `ToolRouter`
- `DatabasePort`
- `OntologyPort`

## Adapters (for testing & dev only)
- `InMemoryMemory`
- `MockLLM`
- `MockDatabaseAdapter`
- `MockEmbeddingSelectorPort`

## Agent Factory
- `PipelineDefinition`
- `PipelineCatalog`
- `get_catalog`
- `Ontology`
- `OntologyNode`
- `OntologyRelation`
- `QueryIntent`
- `SQLQuery`
- `RelationshipType`

---

## Notes
- Public API is synchronized with `hexai/__init__.py` (`__all__`).
- Changes to this list require updating both this file **and** `__all__`.
- Follow semantic versioning: breaking API changes → MAJOR version bump.
