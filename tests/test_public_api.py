# tests/test_public_api.py
import hexai

EXPECTED = {
 "Orchestrator","DirectedGraph","NodeSpec","YamlPipelineBuilder",
 "FunctionNode","LLMNode","ReActAgentNode","LoopNode","ConditionalNode",
 "PromptTemplate","FewShotPromptTemplate",
 "LLM","LongTermMemory","ToolRouter","DatabasePort","OntologyPort",
 "InMemoryMemory","MockLLM","MockDatabaseAdapter","MockEmbeddingSelectorPort",
 "PipelineDefinition","PipelineCatalog","get_catalog",
 "Ontology","OntologyNode","OntologyRelation","QueryIntent","SQLQuery","RelationshipType",
}

def test_public_api_matches_dunder_all():
    assert hasattr(hexai, "__all__")
    assert set(hexai.__all__) == EXPECTED
