"""Tests for the data mapping system."""

from hexai.app.application.data_mapping import (
    DEFAULT_FIELD_MAPPINGS,
    DataAggregator,
    DataMapper,
    SchemaAligner,
    create_passthrough_mapping,
    create_prefixed_mapping,
    create_rename_mapping,
)
from pydantic import BaseModel


class TestUserData(BaseModel):
    """Test Pydantic model."""

    name: str
    email: str
    metadata: dict[str, str] = {}


class TestDataMapper:
    """Test the DataMapper class."""

    def test_simple_field_mapping(self):
        """Test simple field mapping."""
        mapper = DataMapper()
        node_results = {
            "processor": {"text": "Hello world", "count": 5},
            "validator": {"status": "valid", "score": 0.95},
        }

        input_mapping = {"content": "processor.text", "validation": "validator.status"}

        result = mapper.map_data(node_results, input_mapping)

        assert result == {"content": "Hello world", "validation": "valid"}

    def test_nested_field_mapping(self):
        """Test nested field access."""
        mapper = DataMapper()
        node_results = {
            "processor": {"data": {"nested": {"value": "deep_value"}, "simple": "simple_value"}}
        }

        input_mapping = {
            "deep": "processor.data.nested.value",
            "shallow": "processor.data.simple",
        }

        result = mapper.map_data(node_results, input_mapping)

        assert result == {"deep": "deep_value", "shallow": "simple_value"}

    def test_pydantic_model_extraction(self):
        """Test extraction from Pydantic models."""
        mapper = DataMapper()
        user_data = TestUserData(
            name="Alice", email="alice@example.com", metadata={"role": "admin"}
        )

        node_results = {"user_processor": user_data}

        input_mapping = {
            "username": "user_processor.name",
            "user_email": "user_processor.email",
            "user_role": "user_processor.metadata.role",
        }

        result = mapper.map_data(node_results, input_mapping)

        assert result == {
            "username": "Alice",
            "user_email": "alice@example.com",
            "user_role": "admin",
        }

    def test_direct_node_reference(self):
        """Test direct node reference without field specification."""
        mapper = DataMapper()
        node_results = {"processor": {"text": "Hello", "count": 5}}

        input_mapping = {"processor_output": "processor"}

        result = mapper.map_data(node_results, input_mapping)

        assert result == {"processor_output": {"text": "Hello", "count": 5}}

    def test_missing_field_handling(self):
        """Test handling of missing fields."""
        mapper = DataMapper()
        node_results = {"processor": {"text": "Hello"}}

        input_mapping = {
            "content": "processor.text",
            "missing": "processor.nonexistent",
            "missing_node": "nonexistent.field",
        }

        result = mapper.map_data(node_results, input_mapping)

        # Only existing fields should be included
        assert result == {"content": "Hello"}

    def test_primitive_value_extraction(self):
        """Test extraction from primitive values."""
        mapper = DataMapper()
        node_results = {"counter": 42, "message": "Hello"}

        input_mapping = {"count": "counter", "text": "message"}

        result = mapper.map_data(node_results, input_mapping)

        assert result == {"count": 42, "text": "Hello"}


class TestSchemaAligner:
    """Test the SchemaAligner class."""

    def test_direct_field_matches(self):
        """Test direct field name matching."""
        aligner = SchemaAligner()
        source_schema = {"text": str, "metadata": dict, "count": int}
        target_schema = {"text": str, "count": int, "extra": str}

        mapping = aligner.align_schemas(source_schema, target_schema)

        assert mapping == {"text": "text", "count": "count"}

    def test_alternative_field_mapping(self):
        """Test alternative field name mapping."""
        aligner = SchemaAligner()
        source_schema = {"text": str, "metadata": dict}
        target_schema = {"content": str, "config": dict}

        mapping = aligner.align_schemas(source_schema, target_schema)

        assert mapping == {"content": "text", "config": "metadata"}

    def test_partial_mapping(self):
        """Test partial mapping when some fields don't match."""
        aligner = SchemaAligner()
        source_schema = {"text": str, "unknown_field": str}
        target_schema = {"content": str, "unmappable_field": str}

        mapping = aligner.align_schemas(source_schema, target_schema)

        assert mapping == {"content": "text"}

    def test_mapping_suggestions(self):
        """Test mapping suggestions."""
        aligner = SchemaAligner()
        source_schema = {"text": str, "metadata": dict, "user_id": str}
        target_schema = {"content": str, "config": dict, "id": str}

        suggestions = aligner.suggest_mappings(source_schema, target_schema)

        assert "content" in suggestions
        assert "text" in suggestions["content"]
        assert "config" in suggestions
        assert "metadata" in suggestions["config"]
        assert "id" in suggestions
        assert "user_id" in suggestions["id"]

    def test_custom_field_mappings(self):
        """Test SchemaAligner with custom field mappings."""
        custom_mappings = {
            "title": ["name", "label"],
            "description": ["content", "text"],
        }

        aligner = SchemaAligner(mode="custom", custom_mappings=custom_mappings)
        source_schema = {"name": str, "content": str}
        target_schema = {"title": str, "description": str}

        mapping = aligner.align_schemas(source_schema, target_schema)

        assert mapping == {"title": "name", "description": "content"}

    def test_field_mapping_modes(self):
        """Test different field mapping modes."""
        # None mode - no mappings
        aligner_none = SchemaAligner(mode="none")
        assert aligner_none.field_mappings == {}

        # Default mode
        aligner_default = SchemaAligner(mode="default")
        assert aligner_default.field_mappings == DEFAULT_FIELD_MAPPINGS

        # Custom mode requires mappings
        try:
            SchemaAligner(mode="custom")
            raise AssertionError("Should raise ValueError")
        except ValueError as e:
            assert "custom_mappings required" in str(e)


class TestDataAggregator:
    """Test the DataAggregator class."""

    def test_single_dependency_aggregation(self):
        """Test aggregation with single dependency."""
        aggregator = DataAggregator()
        node_results = {"processor": {"text": "Hello", "count": 5}}

        result = aggregator.aggregate_structured(node_results, ["processor"])

        assert result == {"text": "Hello", "count": 5}

    def test_structured_aggregation(self):
        """Test structured aggregation preserving namespaces."""
        aggregator = DataAggregator()
        node_results = {
            "processor": {"text": "Hello", "count": 5},
            "validator": {"status": "valid", "score": 0.95},
        }

        result = aggregator.aggregate_structured(node_results, ["processor", "validator"])

        expected = {
            "processor": {"text": "Hello", "count": 5},
            "validator": {"status": "valid", "score": 0.95},
        }
        assert result == expected

    def test_pydantic_model_aggregation(self):
        """Test aggregation with Pydantic models."""
        aggregator = DataAggregator()
        user_data = TestUserData(name="Alice", email="alice@example.com")

        node_results = {"user_processor": user_data, "validator": {"status": "valid"}}

        # Structured aggregation
        structured = aggregator.aggregate_structured(node_results, ["user_processor", "validator"])

        expected_structured = {
            "user_processor": {"name": "Alice", "email": "alice@example.com", "metadata": {}},
            "validator": {"status": "valid"},
        }
        assert structured == expected_structured

    def test_empty_dependencies(self):
        """Test aggregation with empty dependencies."""
        aggregator = DataAggregator()
        node_results = {"processor": {"text": "Hello"}}

        result = aggregator.aggregate_structured(node_results, [])

        assert result == {}

    def test_missing_dependencies(self):
        """Test aggregation with missing dependencies."""
        aggregator = DataAggregator()
        node_results = {"processor": {"text": "Hello"}}

        result = aggregator.aggregate_structured(node_results, ["processor", "missing"])

        expected = {"processor": {"text": "Hello"}}
        assert result == expected


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_passthrough_mapping(self):
        """Test passthrough mapping creation."""
        fields = ["text", "metadata", "count"]
        mapping = create_passthrough_mapping(fields)

        expected = {"text": "text", "metadata": "metadata", "count": "count"}
        assert mapping == expected

    def test_create_rename_mapping(self):
        """Test rename mapping creation."""
        renames = {"content": "text", "config": "metadata"}
        mapping = create_rename_mapping(renames)

        assert mapping == {"content": "text", "config": "metadata"}

    def test_create_prefixed_mapping(self):
        """Test prefixed mapping creation."""
        fields = ["text", "count"]
        mapping = create_prefixed_mapping(fields, "processor", "proc_")

        expected = {"proc_text": "processor.text", "proc_count": "processor.count"}
        assert mapping == expected

    def test_create_prefixed_mapping_no_prefix(self):
        """Test prefixed mapping without prefix."""
        fields = ["text", "count"]
        mapping = create_prefixed_mapping(fields, "processor")

        expected = {"text": "processor.text", "count": "processor.count"}
        assert mapping == expected


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_text_processing_pipeline(self):
        """Test a realistic text processing pipeline scenario."""
        # Simulate pipeline: input -> processor -> validator -> analyzer

        # Node results from previous executions
        node_results = {
            "processor": {
                "processed_text": "Hello, World!",
                "word_count": 2,
                "metadata": {"language": "en", "encoding": "utf-8"},
            },
            "validator": {
                "is_valid": True,
                "validation_score": 0.95,
                "issues": [],
            },
        }

        # Analyzer node needs specific input format
        mapper = DataMapper()
        input_mapping = {
            "content": "processor.processed_text",
            "language": "processor.metadata.language",
            "quality_score": "validator.validation_score",
            "is_content_valid": "validator.is_valid",
        }

        result = mapper.map_data(node_results, input_mapping)

        expected = {
            "content": "Hello, World!",
            "language": "en",
            "quality_score": 0.95,
            "is_content_valid": True,
        }
        assert result == expected

    def test_schema_alignment_suggestion(self):
        """Test schema alignment for a realistic scenario."""
        # Source schema from an API response processor
        source_schema = {
            "response_text": str,
            "status_code": int,
            "headers": dict,
            "timestamp": str,
        }

        # Target schema for a text analyzer
        target_schema = {
            "content": str,
            "metadata": dict,
            "created_at": str,
        }

        aligner = SchemaAligner()
        suggestions = aligner.suggest_mappings(source_schema, target_schema)

        # Should suggest reasonable mappings (no fuzzy matching, only exact + alternatives)
        assert "metadata" in suggestions
        assert "headers" in suggestions["metadata"]
        assert "created_at" in suggestions
        assert "timestamp" in suggestions["created_at"]
