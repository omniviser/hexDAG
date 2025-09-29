"""Tests for SchemaExtractor class."""

from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from hexai.core.domain.dag import DirectedGraph
from hexai.visualization.schema_extractor import SchemaExtractor


class TestInputModel(BaseModel):
    """Test input model."""

    text: str
    count: int = 10


class TestOutputModel(BaseModel):
    """Test output model."""

    result: str
    score: float


def typed_function(input_data: TestInputModel) -> TestOutputModel:
    """Function with type hints for testing."""
    return TestOutputModel(result="test", score=0.9)


def untyped_function(data):
    """Function without type hints."""
    return {"result": "test"}


class TestSchemaExtractor:
    """Tests for SchemaExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create a SchemaExtractor instance."""
        return SchemaExtractor()

    def test_convert_type_to_schema_dict_pydantic(self, extractor):
        """Test converting Pydantic model to schema dict."""
        schema = extractor._convert_type_to_schema_dict(TestInputModel)

        assert schema is not None
        assert "text" in schema
        assert "count" in schema
        assert schema["text"] == "str"
        assert schema["count"] == "int"

    def test_convert_type_to_schema_dict_with_annotations(self, extractor):
        """Test converting type with annotations to schema dict."""

        class AnnotatedClass:
            __annotations__ = {"field1": str, "field2": int}

        schema = extractor._convert_type_to_schema_dict(AnnotatedClass)

        assert schema is not None
        assert schema["field1"] == "str"
        assert schema["field2"] == "int"

    def test_convert_type_to_schema_dict_dict(self, extractor):
        """Test converting dict to schema dict."""
        input_dict = {"field1": "str", "field2": "int"}
        schema = extractor._convert_type_to_schema_dict(input_dict)

        assert schema == input_dict

    def test_convert_type_to_schema_dict_invalid(self, extractor):
        """Test converting invalid type returns None."""
        schema = extractor._convert_type_to_schema_dict("invalid")
        assert schema is None

        schema = extractor._convert_type_to_schema_dict(42)
        assert schema is None

    def test_extract_function_input_schema(self, extractor):
        """Test extracting input schema from function."""
        schema = extractor.extract_function_input_schema(typed_function)

        assert schema is not None
        assert "text" in schema
        assert "count" in schema

    def test_extract_function_input_schema_no_hints(self, extractor):
        """Test extracting input schema from function without hints."""
        schema = extractor.extract_function_input_schema(untyped_function)
        assert schema is None

    def test_extract_function_output_schema(self, extractor):
        """Test extracting output schema from function."""
        schema = extractor.extract_function_output_schema(typed_function)

        assert schema is not None
        assert "result" in schema
        assert "score" in schema

    def test_extract_function_output_schema_no_hints(self, extractor):
        """Test extracting output schema from function without hints."""
        schema = extractor.extract_function_output_schema(untyped_function)
        assert schema is None

    def test_extract_function_schemas(self, extractor):
        """Test extracting both input and output schemas."""
        input_schema, output_schema = extractor.extract_function_schemas(typed_function)

        assert input_schema is not None
        assert output_schema is not None
        assert "text" in input_schema
        assert "result" in output_schema

    def test_extract_node_schemas_with_models(self, extractor):
        """Test extracting schemas from node with models."""
        node_spec = Mock()
        node_spec.in_model = TestInputModel
        node_spec.out_model = TestOutputModel

        input_schema, output_schema = extractor.extract_node_schemas(node_spec)

        assert input_schema is not None
        assert output_schema is not None
        assert "text" in input_schema
        assert "result" in output_schema

    def test_extract_node_schemas_with_function(self, extractor):
        """Test extracting schemas from node with function."""
        node_spec = Mock()
        node_spec.in_model = None
        node_spec.out_model = None
        node_spec.fn = typed_function
        node_spec.fn.__annotations__ = typed_function.__annotations__

        input_schema, output_schema = extractor.extract_node_schemas(node_spec)

        assert input_schema is not None
        assert output_schema is not None

    def test_extract_node_schemas_no_info(self, extractor):
        """Test extracting schemas from node without info."""
        node_spec = Mock()
        node_spec.in_model = None
        node_spec.out_model = None

        # Remove fn attribute if it exists
        if hasattr(node_spec, "fn"):
            delattr(node_spec, "fn")

        input_schema, output_schema = extractor.extract_node_schemas(node_spec)

        assert input_schema is None
        assert output_schema is None

    def test_extract_all_schemas_no_pipeline_name(self, extractor):
        """Test extracting all schemas without pipeline name."""
        graph = DirectedGraph()
        node_spec = Mock()
        node_spec.in_model = TestInputModel
        node_spec.out_model = TestOutputModel
        node_spec.type = "function"
        graph.nodes = {"test_node": node_spec}

        schemas = extractor.extract_all_schemas(graph)

        assert "test_node" in schemas
        assert schemas["test_node"]["input_schema"] is not None
        assert schemas["test_node"]["output_schema"] is not None
        assert schemas["test_node"]["type"] == "function"

    @patch("hexai.visualization.schema_extractor.Path")
    @patch("hexai.agent_factory.compiler.compile_pipeline")
    def test_load_compiled_schemas_success(self, mock_compile, mock_path, extractor):
        """Test loading compiled schemas successfully."""
        # Mock path existence
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.cwd.return_value = mock_path_instance
        mock_path_instance.__truediv__ = lambda self, other: mock_path_instance

        # Mock compilation result
        mock_compiled = Mock()
        mock_compiled.node_configs = [
            {
                "id": "node1",
                "type": "function",
                "params": {
                    "input_schema": {"field": "str"},
                    "output_schema": {"result": "str"},
                    "fn": "test_func",
                },
            }
        ]
        mock_compiled.input_schema = {"data": "str"}
        mock_compile.return_value = mock_compiled

        schemas, input_schema = extractor.load_compiled_schemas("test_pipeline")

        assert "node1" in schemas
        assert schemas["node1"]["input_schema"] == {"field": "str"}
        assert schemas["node1"]["output_schema"] == {"result": "str"}
        assert schemas["node1"]["type"] == "function"
        assert schemas["node1"]["function_name"] == "test_func"
        assert input_schema == {"data": "str"}

    @patch("hexai.visualization.schema_extractor.Path")
    def test_load_compiled_schemas_no_yaml(self, mock_path, extractor):
        """Test loading compiled schemas when YAML not found."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = False
        mock_path.cwd.return_value = mock_path_instance
        mock_path_instance.__truediv__ = lambda self, other: mock_path_instance

        schemas, input_schema = extractor.load_compiled_schemas("test_pipeline")

        assert schemas == {}
        assert input_schema is None

    def test_format_schema_label_none(self, extractor):
        """Test formatting schema label with None."""
        result = extractor.format_schema_label("Test", None)
        assert result == "Test"

    def test_format_schema_label_pydantic(self, extractor):
        """Test formatting schema label with Pydantic model."""
        result = extractor.format_schema_label("Test", TestInputModel)

        assert "Test" in result
        assert "TestInputModel" in result
        assert "text: str" in result
        assert "count: int" in result

    def test_format_schema_label_dict(self, extractor):
        """Test formatting schema label with dict."""
        schema = {"field1": "str", "field2": "int", "field3": "float"}
        result = extractor.format_schema_label("Test", schema)

        assert "Test" in result
        assert "field1: str" in result
        assert "field2: int" in result
        assert "field3: float" in result

    def test_format_schema_label_dict_truncate(self, extractor):
        """Test formatting schema label with large dict."""
        schema = {f"field{i}": "str" for i in range(10)}
        result = extractor.format_schema_label("Test", schema)

        assert "Test" in result
        assert "..." in result  # Should be truncated

    def test_format_schema_label_type(self, extractor):
        """Test formatting schema label with type."""
        result = extractor.format_schema_label("Test", str)

        assert "Test" in result
        assert "(str)" in result

    def test_format_schema_label_long_string(self, extractor):
        """Test formatting schema label with long string."""
        long_string = "a" * 50
        result = extractor.format_schema_label("Test", long_string)

        assert "Test" in result
        assert "..." in result  # Should be truncated

    @patch("hexai.visualization.schema_extractor.logger")
    def test_load_compiled_schemas_import_error(self, mock_logger, extractor):
        """Test handling import error when loading schemas."""
        with patch("hexai.agent_factory.compiler.compile_pipeline", side_effect=ImportError):
            schemas, input_schema = extractor.load_compiled_schemas("test_pipeline")

            assert schemas == {}
            assert input_schema is None
            mock_logger.debug.assert_called()

    @patch("hexai.visualization.schema_extractor.Path")
    @patch("hexai.agent_factory.compiler.compile_pipeline")
    def test_load_compiled_schemas_compile_error(self, mock_compile, mock_path, extractor):
        """Test handling compilation error."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.cwd.return_value = mock_path_instance
        mock_path_instance.__truediv__ = lambda self, other: mock_path_instance

        mock_compile.side_effect = Exception("Compilation failed")

        schemas, input_schema = extractor.load_compiled_schemas("test_pipeline")

        assert schemas == {}
        assert input_schema is None
