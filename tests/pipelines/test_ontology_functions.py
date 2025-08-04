"""Tests for ontology pipeline functions with simplified schema."""

from hexai.adapters.mock import MockOntologyPort
import pytest

from hexai.pipelines.ontology.functions import (
    OntologyPipelineInput,
    load_ontology_context,
    load_ontology_data,
    metadata_resolver,
    ontology_analyzer,
    query_matcher,
    result_formatter,
)


class MockEventManager:
    """Mock event manager for testing."""

    def __init__(self):
        """Initialize the mock event manager."""
        self.traces = []
        self.memory = {}

    def add_trace(self, node_name: str, message: str):
        """Add a trace message."""
        self.traces.append(f"{node_name}: {message}")

    def set_memory(self, key: str, value):
        """Set a memory value."""
        self.memory[key] = value

    def get_memory(self, key: str, default=None):
        """Get a memory value."""
        return self.memory.get(key, default)


class TestLoadOntologyContext:
    """Test the load_ontology_context function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.event_manager = MockEventManager()
        self.mock_ontology = MockOntologyPort()
        self.ports = {"ontology": self.mock_ontology, "event_manager": self.event_manager}

    @pytest.mark.asyncio
    async def test_load_context_success(self):
        """Test successful context loading."""
        input_data = {"user_query": "Show customer orders"}

        result = await load_ontology_context(input_data, **self.ports)

        assert "user_query" in result
        assert result["user_query"] == "Show customer orders"
        assert "database_nodes" in result
        assert "ontology_nodes" in result
        assert "ontology_relations" in result
        assert result["metadata"]["node_name"] == "load_ontology_context"

        # Check database nodes structure
        database_nodes = result["database_nodes"]
        assert "Bart Ontology" in database_nodes
        assert "Customer" in database_nodes["Bart Ontology"]
        assert "Order" in database_nodes["Bart Ontology"]

        # Check that event manager was used
        assert len(self.event_manager.traces) > 0
        assert any("load_ontology_context" in trace for trace in self.event_manager.traces)

    @pytest.mark.asyncio
    async def test_load_context_with_pydantic_model(self):
        """Test loading context with Pydantic model input."""
        input_data = OntologyPipelineInput(user_query="Show customer orders")

        result = await load_ontology_context(input_data.model_dump(), **self.ports)

        assert "user_query" in result
        assert result["user_query"] == "Show customer orders"
        assert "database_nodes" in result
        assert "ontology_nodes" in result
        assert "ontology_relations" in result

    @pytest.mark.asyncio
    async def test_load_context_no_ontology_port(self):
        """Test error when no ontology port provided."""
        input_data = {"user_query": "test"}
        ports = {"event_manager": self.event_manager}

        with pytest.raises(ValueError, match="No ontology port provided"):
            await load_ontology_context(input_data, **ports)

    @pytest.mark.asyncio
    async def test_load_context_empty_query(self):
        """Test context loading with empty query."""
        input_data = {"user_query": ""}

        result = await load_ontology_context(input_data, **self.ports)

        assert result["user_query"] == ""
        assert "database_nodes" in result
        assert "ontology_nodes" in result
        assert "ontology_relations" in result


class TestLoadOntologyData:
    """Test the load_ontology_data function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.event_manager = MockEventManager()
        self.mock_ontology = MockOntologyPort()
        self.ports = {"ontology": self.mock_ontology, "event_manager": self.event_manager}

    @pytest.mark.asyncio
    async def test_load_data_success(self):
        """Test successful data loading."""
        input_data = {
            "user_query": "Show customer orders",
            "ontology_nodes": [
                {"id": "1", "name": "Customer", "type": "Entity"},
                {"id": "2", "name": "Order", "type": "Entity"},
            ],
        }

        result = await load_ontology_data(input_data, **self.ports)

        assert "user_query" in result
        assert "ontology_nodes" in result
        assert "ontology_relations" in result
        assert result["metadata"]["node_name"] == "load_ontology_data"

        # Check that event manager was used
        assert len(self.event_manager.traces) > 0
        assert any("load_ontology_data" in trace for trace in self.event_manager.traces)

    @pytest.mark.asyncio
    async def test_load_data_no_ontology_port(self):
        """Test data loading without ontology port."""
        input_data = {"user_query": "test"}
        ports = {"event_manager": self.event_manager}

        with pytest.raises(ValueError, match="No ontology port provided"):
            await load_ontology_data(input_data, **ports)


class TestMetadataResolver:
    """Test the metadata_resolver function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.event_manager = MockEventManager()

    @pytest.mark.asyncio
    async def test_metadata_resolver_success(self):
        """Test successful metadata resolution."""
        input_data = {
            "user_query": "Show customer orders",
            "ontology_nodes": [
                {
                    "id": "1",
                    "name": "Customer",
                    "type": "Entity",
                    "description": "Customer entity",
                    "ontology_id": "onto1",
                },
                {
                    "id": "2",
                    "name": "Order",
                    "type": "Entity",
                    "description": "Order entity",
                    "ontology_id": "onto1",
                },
            ],
            "ontology_relations": [
                {
                    "id": "rel1",
                    "name": "has_order",
                    "source_node_id": "1",
                    "target_node_id": "2",
                    "type": "Relationship",
                }
            ],
        }

        result = await metadata_resolver(input_data, event_manager=self.event_manager)

        assert "node_metadata" in result
        assert "relation_metadata" in result
        assert "user_query" in result
        assert result["metadata"]["node_name"] == "metadata_resolver"

        # Check node metadata
        assert "1" in result["node_metadata"]
        assert result["node_metadata"]["1"]["name"] == "Customer"
        assert result["node_metadata"]["1"]["description"] == "Customer entity"

        # Check relation metadata
        assert "rel1" in result["relation_metadata"]
        assert result["relation_metadata"]["rel1"]["name"] == "has_order"

        # Check that event manager was used
        assert len(self.event_manager.traces) > 0
        assert any("metadata_resolver" in trace for trace in self.event_manager.traces)

    @pytest.mark.asyncio
    async def test_metadata_resolver_empty_input(self):
        """Test metadata resolver with empty input."""
        input_data = {"user_query": "", "ontology_nodes": [], "ontology_relations": []}

        result = await metadata_resolver(input_data, event_manager=self.event_manager)

        assert result["node_metadata"] == {}
        assert result["relation_metadata"] == {}
        assert result["user_query"] == ""

    @pytest.mark.asyncio
    async def test_metadata_resolver_without_event_manager(self):
        """Test metadata resolver without event manager."""
        input_data = {
            "user_query": "Test query",
            "ontology_nodes": [{"id": "1", "name": "Test"}],
            "ontology_relations": [],
        }

        # Should not raise an error
        result = await metadata_resolver(input_data)

        assert "node_metadata" in result
        assert "1" in result["node_metadata"]


class TestOntologyAnalyzer:
    """Test the ontology_analyzer function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.event_manager = MockEventManager()

    @pytest.mark.asyncio
    async def test_ontology_analyzer_success(self):
        """Test successful ontology analysis."""
        input_data = {
            "user_query": "Show customer orders",
            "ontology_nodes": [
                {"id": "1", "name": "Customer", "type": "Entity"},
                {"id": "2", "name": "Order", "type": "Entity"},
            ],
            "ontology_relations": [
                {"id": "rel1", "name": "has_order", "source_node_id": "1", "target_node_id": "2"}
            ],
        }

        result = await ontology_analyzer(input_data, event_manager=self.event_manager)

        assert "analysis_result" in result
        assert "user_query" in result
        assert "ontology_nodes" in result
        assert result["metadata"]["node_name"] == "ontology_analyzer"

        # Check that event manager was used
        assert len(self.event_manager.traces) > 0
        assert any("ontology_analyzer" in trace for trace in self.event_manager.traces)


class TestQueryMatcher:
    """Test the query_matcher function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.event_manager = MockEventManager()

    @pytest.mark.asyncio
    async def test_query_matcher_success(self):
        """Test successful query matching."""
        input_data = {
            "user_query": "Show customer orders",
            "ontology_nodes": [
                {"id": "1", "name": "Customer", "type": "Entity"},
                {"id": "2", "name": "Order", "type": "Entity"},
            ],
        }

        result = await query_matcher(input_data, event_manager=self.event_manager)

        assert "matched_entities" in result
        assert "user_query" in result
        assert "ontology_nodes" in result
        assert result["metadata"]["node_name"] == "query_matcher"

        # Check that event manager was used
        assert len(self.event_manager.traces) > 0
        assert any("query_matcher" in trace for trace in self.event_manager.traces)


class TestResultFormatter:
    """Test the result_formatter function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.event_manager = MockEventManager()

    @pytest.mark.asyncio
    async def test_result_formatter_success(self):
        """Test successful result formatting."""
        input_data = {
            "user_query": "Show customer orders",
            "ontology_nodes": [{"id": "1", "name": "Customer", "type": "Entity"}],
            "analysis_result": {"score": 0.95, "confidence": "high"},
        }

        result = await result_formatter(input_data, event_manager=self.event_manager)

        assert "formatted_result" in result
        assert "user_query" in result
        assert "ontology_nodes" in result
        assert result["metadata"]["node_name"] == "result_formatter"

        # Check that event manager was used
        assert len(self.event_manager.traces) > 0
        assert any("result_formatter" in trace for trace in self.event_manager.traces)


class TestOntologyFunctionIntegration:
    """Integration tests for ontology functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.event_manager = MockEventManager()
        self.mock_ontology = MockOntologyPort()
        self.ports = {"ontology": self.mock_ontology, "event_manager": self.event_manager}

    @pytest.mark.asyncio
    async def test_function_pipeline_integration(self):
        """Test that functions work together in pipeline order."""
        input_data = {"user_query": "Show customer orders"}

        # Step 1: Load ontology context
        context_result = await load_ontology_context(input_data, **self.ports)

        # Step 2: Load ontology data
        data_result = await load_ontology_data(context_result, **self.ports)

        # Step 3: Resolve metadata
        metadata_result = await metadata_resolver(data_result, event_manager=self.event_manager)

        # Step 4: Analyze ontology
        analysis_result = await ontology_analyzer(metadata_result, event_manager=self.event_manager)

        # Step 5: Match query
        match_result = await query_matcher(analysis_result, event_manager=self.event_manager)

        # Step 6: Format result
        final_result = await result_formatter(match_result, event_manager=self.event_manager)

        # Verify the pipeline flow
        assert "user_query" in final_result
        assert "formatted_result" in final_result
        assert final_result["user_query"] == "Show customer orders"

        # Verify event manager usage throughout the pipeline
        assert len(self.event_manager.traces) >= 6  # At least one trace per function

    @pytest.mark.asyncio
    async def test_event_manager_integration(self):
        """Test that all functions properly use event manager."""
        input_data = {"user_query": "Test event manager integration"}

        await load_ontology_context(input_data, **self.ports)

        # Check that traces were added
        assert len(self.event_manager.traces) > 0
        assert any("load_ontology_context" in trace for trace in self.event_manager.traces)
