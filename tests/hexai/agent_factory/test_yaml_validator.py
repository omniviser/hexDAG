"""Tests for YAML pipeline validator."""

from hexai.agent_factory.yaml_validator import ValidationReport, YamlValidator


class TestValidationReport:
    """Tests for ValidationReport class."""

    def test_empty_result_is_valid(self):
        """Test that empty result is valid."""
        result = ValidationReport()
        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        assert len(result.suggestions) == 0

    def test_result_with_errors_is_invalid(self):
        """Test that result with errors is invalid."""
        result = ValidationReport()
        result.add_error("Test error")
        assert not result.is_valid
        assert len(result.errors) == 1

    def test_warnings_dont_affect_validity(self):
        """Test that warnings don't make result invalid."""
        result = ValidationReport()
        result.add_warning("Test warning")
        result.add_suggestion("Test suggestion")
        assert result.is_valid
        assert len(result.warnings) == 1
        assert len(result.suggestions) == 1


class TestYamlValidator:
    """Tests for YamlValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = YamlValidator()

    def test_valid_minimal_config(self):
        """Test validation of minimal valid config."""
        config = {"nodes": [{"id": "node1", "type": "function", "params": {"fn": "test"}}]}
        result = self.validator.validate(config)
        assert result.is_valid

    def test_missing_nodes_field(self):
        """Test validation fails when nodes field is missing."""
        config = {"name": "test"}
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("nodes" in error for error in result.errors)

    def test_invalid_nodes_type(self):
        """Test validation fails when nodes is not a list."""
        config = {"nodes": "not_a_list"}
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("list" in error for error in result.errors)

    def test_empty_nodes_warning(self):
        """Test warning for empty nodes list."""
        config = {"nodes": []}
        result = self.validator.validate(config)
        assert result.is_valid
        assert len(result.warnings) > 0
        assert any("no nodes" in warning for warning in result.warnings)

    def test_duplicate_node_ids(self):
        """Test detection of duplicate node IDs."""
        config = {
            "nodes": [{"id": "node1", "type": "function"}, {"id": "node1", "type": "function"}]
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("Duplicate" in error for error in result.errors)

    def test_missing_node_id(self):
        """Test detection of missing node ID."""
        config = {"nodes": [{"type": "function", "params": {"fn": "test"}}]}
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("Missing required field 'id'" in error for error in result.errors)

    def test_invalid_node_type(self):
        """Test detection of invalid node type."""
        config = {"nodes": [{"id": "node1", "type": "invalid_type"}]}
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("Invalid type" in error for error in result.errors)

    def test_function_node_missing_fn(self):
        """Test function node without fn parameter."""
        config = {"nodes": [{"id": "node1", "type": "function", "params": {}}]}
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("require 'fn' parameter" in error for error in result.errors)

    def test_llm_node_missing_template(self):
        """Test LLM node without prompt_template."""
        config = {"nodes": [{"id": "node1", "type": "llm", "params": {}}]}
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("require 'prompt_template'" in error for error in result.errors)

    def test_invalid_dependency(self):
        """Test detection of non-existent dependency."""
        config = {"nodes": [{"id": "node1", "type": "function", "depends_on": ["node2"]}]}
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("does not exist" in error for error in result.errors)

    def test_dependency_cycle_detection(self):
        """Test detection of dependency cycles."""
        config = {
            "nodes": [
                {"id": "node1", "type": "function", "depends_on": ["node2"]},
                {"id": "node2", "type": "function", "depends_on": ["node3"]},
                {"id": "node3", "type": "function", "depends_on": ["node1"]},
            ]
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("cycle" in error.lower() for error in result.errors)

    def test_valid_dependencies(self):
        """Test valid dependency configuration."""
        config = {
            "nodes": [
                {"id": "node1", "type": "function", "params": {"fn": "test"}},
                {
                    "id": "node2",
                    "type": "function",
                    "params": {"fn": "test"},
                    "depends_on": ["node1"],
                },
                {
                    "id": "node3",
                    "type": "function",
                    "params": {"fn": "test"},
                    "depends_on": ["node1", "node2"],
                },
            ]
        }
        result = self.validator.validate(config)
        assert result.is_valid

    def test_field_mapping_validation(self):
        """Test field mapping validation."""
        config = {
            "common_field_mappings": {"standard": {"field1": "source.field1"}},
            "nodes": [
                {
                    "id": "node1",
                    "type": "function",
                    "params": {"fn": "test", "field_mapping": "standard"},
                }
            ],
        }
        result = self.validator.validate(config)
        assert result.is_valid

    def test_unknown_field_mapping_reference(self):
        """Test warning for unknown field mapping."""
        config = {
            "nodes": [
                {
                    "id": "node1",
                    "type": "function",
                    "params": {"fn": "test", "field_mapping": "unknown"},
                }
            ]
        }
        result = self.validator.validate(config)
        assert result.is_valid  # Just a warning
        assert any("unknown field mapping" in warning for warning in result.warnings)

    def test_invalid_field_mapping_format(self):
        """Test error for invalid field mapping format."""
        config = {
            "nodes": [
                {"id": "node1", "type": "function", "params": {"fn": "test", "field_mapping": 123}}
            ]
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("must be a string or dictionary" in error for error in result.errors)

    def test_field_mapping_dependency_suggestion(self):
        """Test suggestion for missing dependency in field mapping."""
        config = {
            "nodes": [
                {"id": "source", "type": "function", "params": {"fn": "test"}},
                {
                    "id": "consumer",
                    "type": "function",
                    "params": {"fn": "test", "input_mapping": {"data": "source.value"}},
                    "depends_on": [],
                },
            ]
        }
        result = self.validator.validate(config)
        assert result.is_valid
        assert any("Consider adding 'source' to dependencies" in s for s in result.suggestions)

    def test_non_dict_config(self):
        """Test validation of non-dict config."""
        config = ["not", "a", "dict"]
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("must be a dictionary" in error for error in result.errors)

    def test_invalid_common_field_mappings(self):
        """Test validation of invalid common_field_mappings."""
        config = {"common_field_mappings": "not_a_dict", "nodes": []}
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("must be a dictionary" in error for error in result.errors)
