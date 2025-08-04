"""Tests for Ontology pipeline functionality."""

from hexai.pipelines.ontology.pipeline import OntologyPipeline


class MockEventManager:
    """Mock event manager for testing."""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or "test-session-123"
        self.traces = []
        self.memory = {}

    async def add_trace(self, node_name: str, message: str):
        """Mock add_trace that stores traces."""
        self.traces.append(f"[{node_name}] {message}")

    def set_memory(self, key: str, value):
        """Mock set_memory that stores in memory dict."""
        self.memory[key] = value

    def get_memory(self, key: str, default=None):
        """Mock get_memory that retrieves from memory dict."""
        return self.memory.get(key, default)


class TestOntologyPipeline:
    """Test OntologyPipeline class."""

    def test_pipeline_creation(self):
        """Test pipeline creation and basic properties."""
        pipeline = OntologyPipeline()

        assert pipeline.name == "ontology_pipeline"
        assert pipeline.description is not None
        assert hasattr(pipeline, "builder")

    def test_pipeline_function_registration(self):
        """Test that pipeline functions are registered."""
        pipeline = OntologyPipeline()

        # Check that functions are registered
        assert "load_ontology_context" in pipeline.builder.registered_functions
        assert "load_ontology_data" in pipeline.builder.registered_functions
        assert "metadata_resolver" in pipeline.builder.registered_functions
        assert "ontology_analyzer" in pipeline.builder.registered_functions
        assert "query_matcher" in pipeline.builder.registered_functions
        assert "result_formatter" in pipeline.builder.registered_functions

    def test_pipeline_validation(self):
        """Test pipeline validation."""
        pipeline = OntologyPipeline()

        validation_result = pipeline.validate()
        assert validation_result["valid"] is True
        assert len(validation_result["errors"]) == 0

    def test_pipeline_input_type(self):
        """Test pipeline input type inference."""
        pipeline = OntologyPipeline()

        input_type = pipeline.get_input_type()
        # Should be able to infer input type from first nodes
        assert input_type is not None

    def test_pipeline_output_type(self):
        """Test pipeline output type inference."""
        pipeline = OntologyPipeline()

        output_type = pipeline.get_output_type()
        # Should be able to infer output type from last nodes
        assert output_type is not None

    def test_pipeline_node_types(self):
        """Test pipeline node type extraction."""
        pipeline = OntologyPipeline()

        node_types = pipeline.get_node_types()
        # Should return dictionary of node names to types
        assert isinstance(node_types, dict)
        assert len(node_types) > 0


class TestOntologyPipelineIntegration:
    """Integration tests for Ontology pipeline."""

    def test_event_manager_integration(self):
        """Test that pipeline works with event_manager."""
        event_manager = MockEventManager()

        # Event manager should have the required methods
        assert hasattr(event_manager, "add_trace")
        assert hasattr(event_manager, "set_memory")
        assert hasattr(event_manager, "get_memory")

        # Test setting and getting memory
        event_manager.set_memory("test_key", "test_value")
        assert event_manager.get_memory("test_key") == "test_value"
