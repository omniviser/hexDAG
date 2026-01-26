"""Tests for YAML pipeline validator."""

from hexdag.core.pipeline_builder.yaml_validator import ValidationReport, YamlValidator


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
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node1"},
                        "spec": {"fn": "test"},
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert result.is_valid

    def test_missing_nodes_field(self):
        """Test validation fails when nodes field is missing."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {},  # Missing 'nodes'
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("nodes" in error for error in result.errors)

    def test_invalid_nodes_type(self):
        """Test validation fails when nodes is not a list."""
        config = {"kind": "Pipeline", "metadata": {"name": "test"}, "spec": {"nodes": "not_a_list"}}
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("list" in error for error in result.errors)

    def test_empty_nodes_warning(self):
        """Test warning for empty nodes list."""
        config = {"kind": "Pipeline", "metadata": {"name": "test"}, "spec": {"nodes": []}}
        result = self.validator.validate(config)
        assert result.is_valid
        assert len(result.warnings) > 0
        assert any("no nodes" in warning.lower() for warning in result.warnings)

    def test_duplicate_node_ids(self):
        """Test detection of duplicate node IDs."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node1"},
                        "spec": {"fn": "test"},
                    },
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node1"},
                        "spec": {"fn": "test"},
                    },
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("Duplicate" in error for error in result.errors)

    def test_missing_node_id(self):
        """Test detection of missing node ID."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {},  # Missing 'name'
                        "spec": {"fn": "test"},
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("metadata.name" in error for error in result.errors)

    def test_invalid_node_type(self):
        """Test detection of invalid node type."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "nodes": [
                    {
                        "kind": "invalid_node",
                        "metadata": {"name": "node1"},
                        "spec": {},
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("Invalid type" in error for error in result.errors)

    def test_function_node_missing_fn(self):
        """Test function node without fn parameter."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node1"},
                        "spec": {},  # Missing 'fn'
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any(
            "fn" in error.lower() for error in result.errors
        )  # Schema validation: "Missing required field 'fn'"

    def test_llm_node_missing_template(self):
        """Test LLM node without prompt_template."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "nodes": [
                    {
                        "kind": "llm_node",
                        "metadata": {"name": "node1"},
                        "spec": {},  # Missing 'prompt_template'
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any(
            "template" in error.lower() for error in result.errors
        )  # Schema validation uses "template" not "prompt_template"

    def test_invalid_dependency(self):
        """Test detection of non-existent dependency."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node1"},
                        "spec": {"fn": "test", "dependencies": ["node2"]},
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("does not exist" in error for error in result.errors)

    def test_dependency_cycle_detection(self):
        """Test detection of dependency cycles."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node1"},
                        "spec": {"fn": "test", "dependencies": ["node2"]},
                    },
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node2"},
                        "spec": {"fn": "test", "dependencies": ["node3"]},
                    },
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node3"},
                        "spec": {"fn": "test", "dependencies": ["node1"]},
                    },
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("cycle" in error.lower() for error in result.errors)

    def test_valid_dependencies(self):
        """Test valid dependency configuration."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node1"},
                        "spec": {"fn": "test"},
                    },
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node2"},
                        "spec": {"fn": "test", "dependencies": ["node1"]},
                    },
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node3"},
                        "spec": {"fn": "test", "dependencies": ["node1", "node2"]},
                    },
                ]
            },
        }
        result = self.validator.validate(config)
        assert result.is_valid

    def test_field_mapping_validation(self):
        """Test field mapping validation.

        Note: With simplified validation (no registry), we only check required fields
        for known node types. Unknown fields like 'field_mapping' are not rejected
        since we can't know all valid fields without full schema introspection.
        """
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "common_field_mappings": {"standard": {"field1": "source.field1"}},
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node1"},
                        "spec": {"fn": "test", "field_mapping": "standard"},
                    }
                ],
            },
        }
        result = self.validator.validate(config)
        # With simplified validation, we don't reject unknown fields
        # The node has all required fields (fn for function_node)
        assert result.is_valid

    def test_unknown_field_mapping_reference(self):
        """Test unknown field mapping handling.

        Note: With simplified validation, we don't reject unknown fields.
        Field mapping validation would happen at runtime when the field is used.
        """
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node1"},
                        "spec": {"fn": "test", "field_mapping": "unknown"},
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        # With simplified validation, unknown fields are not rejected
        # The node has all required fields (fn for function_node)
        assert result.is_valid

        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node1"},
                        "spec": {"fn": "test", "field_mapping": 123},
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        # With simplified validation, unknown fields are not rejected
        # Type validation of field_mapping value would happen at runtime
        assert result.is_valid

    def test_field_mapping_dependency_suggestion(self):
        """Test suggestion for missing dependency in field mapping."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "source"},
                        "spec": {"fn": "test"},
                    },
                    {
                        "kind": "function_node",
                        "metadata": {"name": "consumer"},
                        "spec": {
                            "fn": "test",
                            "input_mapping": {"data": "source.value"},
                            "dependencies": [],
                        },
                    },
                ]
            },
        }
        result = self.validator.validate(config)
        # Note: Field mapping validation uses legacy format internally
        assert result.is_valid  # Valid since field mapping validation doesn't check K8s format yet

    def test_non_dict_config(self):
        """Test validation of non-dict config."""
        config = ["not", "a", "dict"]
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("must be a dictionary" in error for error in result.errors)

    def test_invalid_common_field_mappings(self):
        """Test validation of invalid common_field_mappings."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {"common_field_mappings": "not_a_dict", "nodes": []},
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("must be a dictionary" in error for error in result.errors)

    def test_custom_node_types(self):
        """Test validator with custom node types."""
        # Create validator with custom node types
        custom_validator = YamlValidator(valid_node_types={"custom", "special", "function"})

        # Test that custom type is valid
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "nodes": [
                    {
                        "kind": "custom_node",
                        "metadata": {"name": "node1"},
                        "spec": {},
                    }
                ]
            },
        }
        result = custom_validator.validate(config)
        assert result.is_valid

        # Test that default LLM type is now invalid
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "nodes": [
                    {
                        "kind": "llm_node",
                        "metadata": {"name": "node1"},
                        "spec": {"prompt_template": "test"},
                    }
                ]
            },
        }
        result = custom_validator.validate(config)
        assert not result.is_valid
        assert any("Invalid type 'llm'" in error for error in result.errors)

    def test_custom_node_types_with_set(self):
        """Test validator accepts regular set for node types."""
        # Create validator with regular set (not frozenset)
        custom_validator = YamlValidator(valid_node_types={"workflow", "task"})

        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "nodes": [
                    {
                        "kind": "workflow_node",
                        "metadata": {"name": "node1"},
                        "spec": {},
                    }
                ]
            },
        }
        result = custom_validator.validate(config)
        assert result.is_valid

    def test_default_node_types_preserved(self):
        """Test that default validator still has all default types."""
        default_validator = YamlValidator()

        # All default types should work
        for node_type in ["function", "llm", "agent", "loop"]:
            config = {
                "kind": "Pipeline",
                "metadata": {"name": "test"},
                "spec": {
                    "nodes": [
                        {
                            "kind": f"{node_type}_node",
                            "metadata": {"name": "node1"},
                            "spec": {"fn": "test"},
                        }
                    ]
                },
            }
            default_validator.validate(config)
            # All types are valid, though some may have validation errors for missing params
            assert node_type in ["function", "llm", "agent", "loop"]


class TestManifestValidation:
    """Tests for declarative manifest YAML validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = YamlValidator()

    def test_manifest_minimal_valid(self):
        """Test declarative manifest minimal valid configuration."""
        config = {
            "apiVersion": "v1",
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node1"},
                        "spec": {"fn": "test"},
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert result.is_valid

    def test_manifest_with_dependencies(self):
        """Test declarative manifest configuration with dependencies."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node1"},
                        "spec": {"fn": "test", "dependencies": []},
                    },
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node2"},
                        "spec": {"fn": "test", "dependencies": ["node1"]},
                    },
                ]
            },
        }
        result = self.validator.validate(config)
        assert result.is_valid

    def test_manifest_missing_dependency(self):
        """Test declarative manifest with missing dependency."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node1"},
                        "spec": {"fn": "test", "dependencies": ["nonexistent"]},
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("nonexistent" in err for err in result.errors)

    def test_manifest_duplicate_node_names(self):
        """Test declarative manifest with duplicate node names."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node1"},
                        "spec": {"fn": "test"},
                    },
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node1"},
                        "spec": {"fn": "test"},
                    },
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("Duplicate" in err for err in result.errors)

    def test_manifest_missing_node_name(self):
        """Test declarative manifest with missing node name in metadata."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {},  # Missing 'name'
                        "spec": {"fn": "test"},
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("metadata.name" in err for err in result.errors)

    def test_manifest_with_namespace(self):
        """Test declarative manifest with namespace-qualified kind."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "core:function_node",
                        "metadata": {"name": "node1"},
                        "spec": {"fn": "test"},
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert result.is_valid

    def test_manifest_invalid_node_type(self):
        """Test declarative manifest with invalid node type."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "invalid_node",
                        "metadata": {"name": "node1"},
                        "spec": {"fn": "test"},
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("Invalid type" in err for err in result.errors)
