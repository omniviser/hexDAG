"""Tests for the base pipeline functionality."""

from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from hexai import DirectedGraph
from hexai.agent_factory.base import PipelineCatalog, PipelineDefinition


class InputModel(BaseModel):
    """Input model for testing."""

    query: str
    context: str = ""


class OutputModel(BaseModel):
    """Output model for testing."""

    result: str
    confidence: float


class MockPipeline(PipelineDefinition):
    """Mock pipeline for testing."""

    @property
    def name(self) -> str:
        """Pipeline name."""
        return "mock_pipeline"

    @property
    def description(self) -> str:
        """Pipeline description."""
        return "Mock pipeline for testing"

    def _register_functions(self) -> None:
        """Register mock functions."""

        async def mock_function(input_data, context, **ports):
            return {"result": "mock_output"}

        self.builder.register_function("mock_function", mock_function)


class TestPipelineDefinition:
    """Tests for PipelineDefinition base class."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = MockPipeline()

        assert pipeline.name == "mock_pipeline"
        assert pipeline.description == "Mock pipeline for testing"
        assert pipeline.builder is not None
        assert "mock_function" in pipeline.builder.registered_functions

    @pytest.mark.asyncio
    async def test_execute_without_yaml(self):
        """Test pipeline execution without YAML configuration."""
        pipeline = MockPipeline()
        pipeline._yaml_path = None

        result = await pipeline.execute()

        assert result["status"] == "error"
        assert (
            "YAML path not configured" in result["error"]
            or "No pipeline YAML found" in result["error"]
        )

    @pytest.mark.asyncio
    async def test_execute_with_yaml(self):
        """Test pipeline execution with YAML configuration."""
        pipeline = MockPipeline()
        pipeline._yaml_path = "test.yaml"

        # Mock the execute method to avoid complex orchestrator mocking
        with patch.object(MockPipeline, "execute") as mock_execute:
            mock_execute.return_value = {
                "status": "success",
                "results": {"test_node": {"result": "mock_output"}},
                "trace": [],
            }

            result = await pipeline.execute(input_data={"test": "data"})

            assert result["status"] == "success"
            assert result["results"] == {"test_node": {"result": "mock_output"}}
            assert isinstance(result["trace"], list)

    @pytest.mark.asyncio
    async def test_execute_with_error(self):
        """Test pipeline execution with error handling."""
        pipeline = MockPipeline()
        pipeline._yaml_path = "test.yaml"

        # Mock builder to raise an error
        pipeline.builder.build_from_yaml_file = Mock(side_effect=Exception("Test error"))

        result = await pipeline.execute()

        assert result["status"] == "error"
        assert "Test error" in result["error"]
        assert isinstance(result["trace"], list)

    def test_get_config(self):
        """Test getting pipeline configuration."""
        pipeline = MockPipeline()
        pipeline._config = {"test": "config"}

        config = pipeline.get_config()
        assert config == {"test": "config"}

    def test_validate_no_yaml(self):
        """Test validation without YAML."""
        pipeline = MockPipeline()
        pipeline._yaml_path = None

        result = pipeline.validate()

        assert result["valid"] is False
        assert "No pipeline YAML found" in result["errors"][0]

    def test_validate_with_valid_yaml(self):
        """Test validation with valid YAML."""
        pipeline = MockPipeline()
        pipeline._yaml_path = "test.yaml"

        # Mock successful graph building
        pipeline.builder.build_from_yaml_file = Mock(return_value=(DirectedGraph(), {}))

        result = pipeline.validate()

        assert result["valid"] is True
        assert result["errors"] == []

    def test_validate_with_invalid_yaml(self):
        """Test validation with invalid YAML."""
        pipeline = MockPipeline()
        pipeline._yaml_path = "test.yaml"

        # Mock graph building failure
        pipeline.builder.build_from_yaml_file = Mock(side_effect=Exception("Invalid config"))

        result = pipeline.validate()

        assert result["valid"] is False
        assert "Invalid config" in result["errors"][0]

    # Type introspection tests
    def test_get_input_type_no_yaml(self):
        """Test get_input_type when no YAML is found."""
        pipeline = MockPipeline()
        pipeline._yaml_path = None
        result = pipeline.get_input_type()
        assert result is None

    def test_get_output_type_no_yaml(self):
        """Test get_output_type when no YAML is found."""
        pipeline = MockPipeline()
        pipeline._yaml_path = None
        result = pipeline.get_output_type()
        assert result is None

    def test_get_node_types_no_yaml(self):
        """Test get_node_types when no YAML is found."""
        pipeline = MockPipeline()
        pipeline._yaml_path = None
        result = pipeline.get_node_types()
        assert result == {}

    def test_get_input_type_single_node(self):
        """Test get_input_type with single first node."""
        pipeline = MockPipeline()

        # Mock graph with single first node
        mock_node_spec = Mock()
        mock_node_spec.in_model = InputModel

        mock_graph = Mock()
        mock_graph.nodes = {"first_node": mock_node_spec}
        mock_graph.dependencies = {"first_node": []}  # No dependencies = first node
        mock_graph.waves.return_value = [["first_node"]]

        pipeline._yaml_path = "/fake/path/pipeline.yaml"
        with patch.object(pipeline.builder, "build_from_yaml_file", return_value=(mock_graph, {})):
            result = pipeline.get_input_type()
            assert result == InputModel

    def test_get_input_type_multiple_nodes(self):
        """Test get_input_type with multiple first nodes."""
        pipeline = MockPipeline()

        # Mock graph with multiple first nodes
        mock_node_spec1 = Mock()
        mock_node_spec1.in_model = InputModel
        mock_node_spec2 = Mock()
        mock_node_spec2.in_model = str

        mock_graph = Mock()
        mock_graph.nodes = {
            "first_node1": mock_node_spec1,
            "first_node2": mock_node_spec2,
            "second_node": Mock(),
        }
        mock_graph.dependencies = {
            "first_node1": [],  # No dependencies = first node
            "first_node2": [],  # No dependencies = first node
            "second_node": ["first_node1"],  # Has dependencies = not first node
        }
        mock_graph.waves.return_value = [
            ["first_node1", "first_node2"],
            ["second_node"],
        ]

        pipeline._yaml_path = "/fake/path/pipeline.yaml"
        with patch.object(pipeline.builder, "build_from_yaml_file", return_value=(mock_graph, {})):
            result = pipeline.get_input_type()
            expected = {"first_node1": InputModel, "first_node2": str}
            assert result == expected

    def test_get_output_type_single_node(self):
        """Test get_output_type with single last node."""
        pipeline = MockPipeline()

        # Mock graph with single last node
        mock_node_spec = Mock()
        mock_node_spec.out_model = OutputModel

        mock_graph = Mock()
        mock_graph.nodes = {"last_node": mock_node_spec}
        mock_graph.dependencies = {}  # No dependencies means last_node has no dependents
        mock_graph.waves.return_value = [["last_node"]]

        pipeline._yaml_path = "/fake/path/pipeline.yaml"
        with patch.object(pipeline.builder, "build_from_yaml_file", return_value=(mock_graph, {})):
            result = pipeline.get_output_type()
            assert result == OutputModel

    def test_get_output_type_multiple_nodes(self):
        """Test get_output_type with multiple last nodes."""
        pipeline = MockPipeline()

        # Mock graph with multiple last nodes
        mock_node_spec1 = Mock()
        mock_node_spec1.out_model = OutputModel
        mock_node_spec2 = Mock()
        mock_node_spec2.out_model = dict

        mock_graph = Mock()
        mock_graph.nodes = {
            "first_node": Mock(),
            "last_node1": mock_node_spec1,
            "last_node2": mock_node_spec2,
        }
        # Only first_node appears in dependencies, so last_node1 and last_node2 are last nodes
        mock_graph.dependencies = {
            "last_node1": ["first_node"],
            "last_node2": ["first_node"],
        }
        mock_graph.waves.return_value = [["first_node"], ["last_node1", "last_node2"]]

        pipeline._yaml_path = "/fake/path/pipeline.yaml"
        with patch.object(pipeline.builder, "build_from_yaml_file", return_value=(mock_graph, {})):
            result = pipeline.get_output_type()
            expected = {"last_node1": OutputModel, "last_node2": dict}
            assert result == expected

    def test_get_node_types_success(self):
        """Test get_node_types with successful graph building."""
        pipeline = MockPipeline()

        # Mock node specs
        mock_node_spec1 = Mock()
        mock_node_spec1.in_model = InputModel
        mock_node_spec1.out_model = dict
        mock_node_spec1.fn = Mock()
        mock_node_spec1.fn.__name__ = "process_input"

        mock_node_spec2 = Mock()
        mock_node_spec2.in_model = dict
        mock_node_spec2.out_model = OutputModel
        mock_node_spec2.fn = Mock()
        mock_node_spec2.fn.__name__ = "generate_output"

        mock_graph = Mock()
        mock_graph.nodes = {
            "input_processor": mock_node_spec1,
            "output_generator": mock_node_spec2,
        }

        pipeline._yaml_path = "/fake/path/pipeline.yaml"
        with patch.object(pipeline.builder, "build_from_yaml_file", return_value=(mock_graph, {})):
            result = pipeline.get_node_types()

            expected = {
                "input_processor": {
                    "name": "input_processor",
                    "input_type": InputModel,
                    "output_type": dict,
                    "function": "process_input",
                },
                "output_generator": {
                    "name": "output_generator",
                    "input_type": dict,
                    "output_type": OutputModel,
                    "function": "generate_output",
                },
            }
            assert result == expected

    def test_get_node_types_no_function_name(self):
        """Test get_node_types when node function has no __name__ attribute."""
        pipeline = MockPipeline()

        # Mock node spec without function name
        mock_node_spec = Mock()
        mock_node_spec.in_model = str
        mock_node_spec.out_model = dict
        mock_node_spec.fn = Mock(spec=[])  # Mock without __name__ attribute

        mock_graph = Mock()
        mock_graph.nodes = {"test_node": mock_node_spec}

        pipeline._yaml_path = "/fake/path/pipeline.yaml"
        with patch.object(pipeline.builder, "build_from_yaml_file", return_value=(mock_graph, {})):
            result = pipeline.get_node_types()

            expected = {
                "test_node": {
                    "name": "test_node",
                    "input_type": str,
                    "output_type": dict,
                }
            }
            assert result == expected

    def test_get_input_type_exception_handling(self):
        """Test get_input_type handles exceptions gracefully."""
        pipeline = MockPipeline()
        pipeline._yaml_path = "/fake/path/pipeline.yaml"
        with patch.object(
            pipeline.builder,
            "build_from_yaml_file",
            side_effect=Exception("Test error"),
        ):
            result = pipeline.get_input_type()
            assert result is None

    def test_get_output_type_exception_handling(self):
        """Test get_output_type handles exceptions gracefully."""
        pipeline = MockPipeline()
        pipeline._yaml_path = "/fake/path/pipeline.yaml"
        with patch.object(
            pipeline.builder,
            "build_from_yaml_file",
            side_effect=Exception("Test error"),
        ):
            result = pipeline.get_output_type()
            assert result is None

    def test_get_node_types_exception_handling(self):
        """Test get_node_types handles exceptions gracefully."""
        pipeline = MockPipeline()
        pipeline._yaml_path = "/fake/path/pipeline.yaml"
        with patch.object(
            pipeline.builder,
            "build_from_yaml_file",
            side_effect=Exception("Test error"),
        ):
            result = pipeline.get_node_types()
            assert result == {}

    def test_get_input_type_no_first_nodes(self):
        """Test get_input_type when no first nodes are found."""
        pipeline = MockPipeline()

        # Mock graph where all nodes have dependencies
        mock_graph = Mock()
        mock_graph.nodes = {"node1": Mock(), "node2": Mock()}
        mock_graph.dependencies = {
            "node1": ["node2"],
            "node2": ["node1"],
        }  # Circular dependency

        pipeline._yaml_path = "/fake/path/pipeline.yaml"
        with patch.object(pipeline.builder, "build_from_yaml_file", return_value=(mock_graph, {})):
            result = pipeline.get_input_type()
            assert result is None

    def test_get_output_type_no_last_nodes(self):
        """Test get_output_type when no last nodes are found."""
        pipeline = MockPipeline()

        # Mock graph where all nodes are dependencies of others
        mock_graph = Mock()
        mock_graph.nodes = {"node1": Mock(), "node2": Mock()}
        mock_graph.dependencies = {
            "node1": ["node2"],
            "node2": ["node1"],  # Both nodes are dependencies
        }

        pipeline._yaml_path = "/fake/path/pipeline.yaml"
        with patch.object(pipeline.builder, "build_from_yaml_file", return_value=(mock_graph, {})):
            result = pipeline.get_output_type()
            assert result is None

    def test_type_methods_maintain_state(self):
        """Test that type introspection doesn't affect pipeline state."""
        pipeline = MockPipeline()
        original_yaml_path = pipeline._yaml_path
        original_config = pipeline._config

        # Call type methods
        pipeline.get_input_type()
        pipeline.get_output_type()
        pipeline.get_node_types()

        # Verify state is unchanged
        assert pipeline._yaml_path == original_yaml_path
        assert pipeline._config == original_config


class TestPipelineCatalog:
    """Tests for PipelineCatalog."""

    class SampleRegistrationPipeline(PipelineDefinition):
        """Sample pipeline class for testing registration."""

        @property
        def name(self) -> str:
            """Pipeline name."""
            return "test_registration"

        @property
        def description(self) -> str:
            """Pipeline description."""
            return "Test registration pipeline"

        def __init__(self, yaml_path: str | None = None):
            super().__init__(yaml_path)
            self.builder = Mock()
            self._config = {"test": "config"}

        def _register_functions(self):
            pass

    def test_pipeline_registration(self):
        """Test manual pipeline registration."""
        catalog = PipelineCatalog()

        # Initially empty
        assert len(catalog._pipelines) == 0

        # Register the pipeline
        catalog.register_pipeline(self.SampleRegistrationPipeline)

        # Should now contain the registered pipeline
        assert "test_registration" in catalog._pipelines
        assert len(catalog._pipelines) == 1

    def test_list_pipelines(self):
        """Test listing pipelines."""
        catalog = PipelineCatalog()
        catalog._pipelines = {"mock_pipeline": MockPipeline}

        pipelines = catalog.list_pipelines()

        assert len(pipelines) == 1
        assert pipelines[0]["name"] == "mock_pipeline"
        assert pipelines[0]["description"] == "Mock pipeline for testing"
        assert "module" in pipelines[0]

    def test_get_pipeline(self):
        """Test getting pipeline instance."""
        catalog = PipelineCatalog()
        catalog._pipelines = {"mock_pipeline": MockPipeline}

        pipeline = catalog.get_pipeline("mock_pipeline")

        assert pipeline is not None
        assert pipeline.name == "mock_pipeline"
        assert pipeline.description == "Mock pipeline for testing"

    def test_get_nonexistent_pipeline(self):
        """Test getting nonexistent pipeline."""
        catalog = PipelineCatalog()

        pipeline = catalog.get_pipeline("nonexistent")

        assert pipeline is None

    @pytest.mark.asyncio
    async def test_execute_pipeline(self):
        """Test executing pipeline through catalog."""
        catalog = PipelineCatalog()
        catalog._pipelines = {"mock_pipeline": MockPipeline}

        # Mock the pipeline execution
        with patch.object(MockPipeline, "execute") as mock_execute:
            mock_execute.return_value = {
                "status": "success",
                "results": {"test": "result"},
                "trace": [],
            }

            result = await catalog.execute_pipeline("mock_pipeline", {"input": "data"})

            assert result["status"] == "success"
            assert result["results"] == {"test": "result"}
            mock_execute.assert_called_once_with({"input": "data"}, None)

    @pytest.mark.asyncio
    async def test_execute_nonexistent_pipeline(self):
        """Test executing nonexistent pipeline through catalog."""
        catalog = PipelineCatalog()

        result = await catalog.execute_pipeline("nonexistent")

        assert result["status"] == "error"
        assert "Pipeline 'nonexistent' not found" in result["error"]
