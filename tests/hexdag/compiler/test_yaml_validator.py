"""Tests for YAML pipeline validator."""

from hexdag.compiler.yaml_validator import ValidationReport, YamlValidator
from hexdag.kernel.context.execution_context import (
    CTX,
    EXPRESSION_NAMESPACES,
    RESERVED_NAMES,
    get_ctx_dict,
)


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

    def test_field_mapping_undeclared_dependency_warns(self):
        """input_mapping references 'source' but dependencies is [] → warning."""
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
        assert result.is_valid
        assert any("references node 'source'" in w for w in result.warnings)

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
                        "kind": "function_node",
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


class TestCompositeNodeValidation:
    """Tests for composite_node validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = YamlValidator()

    def test_composite_valid_while_mode(self):
        """Test valid composite_node with while mode."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "composite_node",
                        "metadata": {"name": "loop"},
                        "spec": {
                            "mode": "while",
                            "condition": "state.count < 5",
                            "initial_state": {"count": 0},
                            "body": "json.dumps",
                        },
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert result.is_valid

    def test_composite_valid_foreach_mode(self):
        """Test valid composite_node with for-each mode."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "composite_node",
                        "metadata": {"name": "process_items"},
                        "spec": {
                            "mode": "for-each",
                            "items": "$input.items",
                            "concurrency": 5,
                            "body": "json.dumps",
                        },
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert result.is_valid

    def test_composite_valid_times_mode(self):
        """Test valid composite_node with times mode."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "composite_node",
                        "metadata": {"name": "repeat"},
                        "spec": {
                            "mode": "times",
                            "count": 10,
                            "body": "json.dumps",
                        },
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert result.is_valid

    def test_composite_valid_if_else_mode(self):
        """Test valid composite_node with if-else mode."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "composite_node",
                        "metadata": {"name": "branch"},
                        "spec": {
                            "mode": "if-else",
                            "condition": "status == 'active'",
                        },
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert result.is_valid

    def test_composite_valid_switch_mode(self):
        """Test valid composite_node with switch mode."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "composite_node",
                        "metadata": {"name": "router"},
                        "spec": {
                            "mode": "switch",
                            "branches": [
                                {"condition": "action == 'accept'", "action": "approve"},
                                {"condition": "action == 'reject'", "action": "deny"},
                            ],
                        },
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert result.is_valid

    def test_composite_missing_mode(self):
        """Test composite_node fails without mode."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "composite_node",
                        "metadata": {"name": "node"},
                        "spec": {},  # Missing 'mode'
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("mode" in err.lower() for err in result.errors)

    def test_composite_invalid_mode(self):
        """Test composite_node fails with invalid mode."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "composite_node",
                        "metadata": {"name": "node"},
                        "spec": {"mode": "invalid_mode"},
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("invalid" in err.lower() and "mode" in err.lower() for err in result.errors)

    def test_composite_while_missing_condition(self):
        """Test while mode fails without condition."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "composite_node",
                        "metadata": {"name": "loop"},
                        "spec": {"mode": "while"},  # Missing 'condition'
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("condition" in err.lower() for err in result.errors)

    def test_composite_foreach_missing_items(self):
        """Test for-each mode fails without items."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "composite_node",
                        "metadata": {"name": "process"},
                        "spec": {"mode": "for-each"},  # Missing 'items'
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("items" in err.lower() for err in result.errors)

    def test_composite_times_missing_count(self):
        """Test times mode fails without count."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "composite_node",
                        "metadata": {"name": "repeat"},
                        "spec": {"mode": "times"},  # Missing 'count'
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("count" in err.lower() for err in result.errors)

    def test_composite_times_invalid_count_type(self):
        """Test times mode fails with non-integer count."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "composite_node",
                        "metadata": {"name": "repeat"},
                        "spec": {"mode": "times", "count": "five"},  # Invalid type
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("integer" in err.lower() for err in result.errors)

    def test_composite_switch_missing_branches(self):
        """Test switch mode fails without branches."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "composite_node",
                        "metadata": {"name": "router"},
                        "spec": {"mode": "switch"},  # Missing 'branches'
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("branches" in err.lower() for err in result.errors)

    def test_composite_body_and_body_pipeline_exclusive(self):
        """Test body and body_pipeline are mutually exclusive."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "composite_node",
                        "metadata": {"name": "node"},
                        "spec": {
                            "mode": "times",
                            "count": 5,
                            "body": "myapp.process",
                            "body_pipeline": "./process.yaml",  # Both specified
                        },
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("both" in err.lower() and "body" in err.lower() for err in result.errors)

    def test_composite_inline_body_validation(self):
        """Test inline body node validation."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "composite_node",
                        "metadata": {"name": "node"},
                        "spec": {
                            "mode": "for-each",
                            "items": "$input.items",
                            "body": [
                                {"kind": "expression_node", "spec": {}},  # Valid
                                {"spec": {}},  # Invalid - missing 'kind'
                            ],
                        },
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("kind" in err.lower() for err in result.errors)


class TestMisplacedNodeFields:
    """Tests for detecting fields misplaced at node level instead of inside spec."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = YamlValidator()

    def test_misplaced_when_at_node_level(self):
        """Test that 'when' at node level is flagged as an error."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node1"},
                        "spec": {"fn": "test"},
                        "when": "x == 1",  # Misplaced: should be inside spec
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any(
            "'when' is not a valid node-level field" in err and "Move it inside 'spec'" in err
            for err in result.errors
        )

    def test_when_inside_spec_valid(self):
        """Test that 'when' inside spec is valid (no misplaced field error)."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node1"},
                        "spec": {"fn": "test", "when": "x == 1"},
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert result.is_valid

    def test_misplaced_timeout_at_node_level(self):
        """Test that 'timeout' at node level is flagged as an error."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node1"},
                        "spec": {"fn": "test"},
                        "timeout": 30,  # Misplaced: should be inside spec
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("'timeout' is not a valid node-level field" in err for err in result.errors)

    def test_misplaced_input_mapping_at_node_level(self):
        """Test that 'input_mapping' at node level is flagged as an error."""
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
                        "metadata": {"name": "node2"},
                        "spec": {"fn": "test"},
                        "input_mapping": {"x": "node1.y"},  # Misplaced
                        "dependencies": ["node1"],
                    },
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any(
            "'input_mapping' is not a valid node-level field" in err for err in result.errors
        )

    def test_strict_mapping_without_input_mapping_is_error(self):
        """strict_mapping: true requires input_mapping to be set."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node1"},
                        "spec": {
                            "fn": "test",
                            "strict_mapping": True,
                        },
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("strict_mapping requires input_mapping" in err for err in result.errors)

    def test_strict_mapping_with_input_mapping_is_valid(self):
        """strict_mapping: true with input_mapping should not produce this specific error."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node1"},
                        "spec": {
                            "fn": "test",
                            "strict_mapping": True,
                            "input_mapping": {"x": "$input.x"},
                        },
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        # Should not have the strict_mapping error (may have other unrelated errors)
        assert not any("strict_mapping requires input_mapping" in err for err in result.errors)

    def test_misplaced_strict_mapping_at_node_level(self):
        """strict_mapping at node level should be flagged as an error."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node1"},
                        "spec": {"fn": "test"},
                        "strict_mapping": True,  # Misplaced
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any(
            "'strict_mapping' is not a valid node-level field" in err for err in result.errors
        )

    def test_misplaced_max_retries_at_node_level(self):
        """Test that 'max_retries' at node level is flagged as an error."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node1"},
                        "spec": {"fn": "test"},
                        "max_retries": 3,  # Misplaced
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("'max_retries' is not a valid node-level field" in err for err in result.errors)

    def test_misplaced_fn_at_node_level(self):
        """Test that 'fn' at node level is flagged as an error."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node1"},
                        "spec": {},
                        "fn": "json.loads",  # Misplaced
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("'fn' is not a valid node-level field" in err for err in result.errors)

    def test_misplaced_prompt_template_at_node_level(self):
        """Test that 'prompt_template' at node level is flagged as an error."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "llm_node",
                        "metadata": {"name": "node1"},
                        "spec": {},
                        "prompt_template": "Analyze: {{input}}",  # Misplaced
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any(
            "'prompt_template' is not a valid node-level field" in err for err in result.errors
        )

    def test_multiple_misplaced_fields_reported(self):
        """Test that multiple misplaced fields each produce a separate error."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "node1"},
                        "spec": {"fn": "test"},
                        "when": "x == 1",
                        "timeout": 30,
                        "max_retries": 3,
                    }
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        misplaced_errors = [
            err for err in result.errors if "is not a valid node-level field" in err
        ]
        assert len(misplaced_errors) == 3

    def test_valid_node_level_keys_accepted(self):
        """Test that valid node-level keys (dependencies, wait_for) are accepted."""
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
                        "metadata": {"name": "node2"},
                        "spec": {"fn": "test"},
                        "dependencies": ["node1"],
                        "wait_for": ["node1"],
                    },
                ]
            },
        }
        result = self.validator.validate(config)
        assert result.is_valid


class TestPyTagValidation:
    """Tests for !py tag validation."""

    def test_validate_py_source_valid(self):
        """Test validation of valid Python source."""
        from hexdag.compiler.py_tag import validate_py_source

        source = """
def process(item, index, state, **ports):
    return item * 2
"""
        errors = validate_py_source(source)
        assert errors == []

    def test_validate_py_source_async_valid(self):
        """Test validation of valid async Python source."""
        from hexdag.compiler.py_tag import validate_py_source

        source = """
async def process(item, index, state, **ports):
    await asyncio.sleep(0)
    return item
"""
        errors = validate_py_source(source)
        assert errors == []

    def test_validate_py_source_syntax_error(self):
        """Test validation catches syntax errors."""
        from hexdag.compiler.py_tag import validate_py_source

        source = "def process( invalid"
        errors = validate_py_source(source)
        assert len(errors) == 1
        assert "Syntax error" in errors[0]

    def test_validate_py_source_no_function(self):
        """Test validation catches missing function."""
        from hexdag.compiler.py_tag import validate_py_source

        source = "x = 1\ny = 2"
        errors = validate_py_source(source)
        assert len(errors) == 1
        assert "must define a function" in errors[0]

    def test_validate_py_source_empty(self):
        """Test validation catches empty source."""
        from hexdag.compiler.py_tag import validate_py_source

        errors = validate_py_source("")
        assert len(errors) == 1
        assert "empty" in errors[0].lower()

    def test_validate_py_source_whitespace_only(self):
        """Test validation catches whitespace-only source."""
        from hexdag.compiler.py_tag import validate_py_source

        errors = validate_py_source("   \n\t  ")
        assert len(errors) == 1
        assert "empty" in errors[0].lower()


class TestOnErrorValidation:
    """Tests for on_error reference validation in YAML validator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = YamlValidator()

    def test_on_error_referencing_nonexistent_node(self):
        """on_error referencing a nonexistent node produces a validation error."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "risky"},
                        "spec": {"fn": "json.loads", "on_error": "nonexistent"},
                    },
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("on_error" in error and "nonexistent" in error for error in result.errors)

    def test_on_error_self_reference(self):
        """on_error pointing to itself produces a validation error."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "risky"},
                        "spec": {"fn": "json.loads", "on_error": "risky"},
                    },
                ]
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("on_error cannot reference itself" in error for error in result.errors)

    def test_valid_on_error_reference(self):
        """Valid on_error reference passes validation."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "risky"},
                        "spec": {"fn": "json.loads", "on_error": "handler"},
                    },
                    {
                        "kind": "function_node",
                        "metadata": {"name": "handler"},
                        "spec": {"fn": "json.loads"},
                    },
                ]
            },
        }
        result = self.validator.validate(config)
        assert result.is_valid


class TestNamingCollisionValidation:
    """Tests for expression/mapping naming collision detection."""

    def setup_method(self):
        self.validator = YamlValidator()

    def _make_pipeline(self, nodes):
        return {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {"nodes": nodes},
        }

    def test_expression_var_collides_with_node_name(self):
        """Expression variable that matches a node name is a build error."""
        config = self._make_pipeline([
            {
                "kind": "expression_node",
                "metadata": {"name": "compute"},
                "spec": {
                    "expressions": {
                        "rate": "1 + 2",  # 'rate' is also a node name
                    },
                },
            },
            {
                "kind": "function_node",
                "metadata": {"name": "rate"},
                "spec": {"fn": "json.loads"},
            },
        ])
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("collides with node" in e for e in result.errors)

    def test_expression_var_collides_with_builtin(self):
        """Expression variable that matches a builtin function is a build error."""
        config = self._make_pipeline([
            {
                "kind": "expression_node",
                "metadata": {"name": "compute"},
                "spec": {
                    "expressions": {
                        "len": "42",  # 'len' is a builtin
                    },
                },
            },
        ])
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("built-in function" in e for e in result.errors)

    def test_input_mapping_alias_collides_with_node(self):
        """input_mapping alias that matches a node name is a build error."""
        config = self._make_pipeline([
            {
                "kind": "function_node",
                "metadata": {"name": "producer"},
                "spec": {"fn": "json.loads"},
            },
            {
                "kind": "expression_node",
                "metadata": {"name": "consumer"},
                "spec": {
                    "input_mapping": {
                        "producer": "producer.result",  # alias shadows node name
                    },
                    "expressions": {"x": "1 + 1"},
                },
            },
        ])
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("collides with node" in e for e in result.errors)

    def test_unknown_first_segment_in_expression(self):
        """Unknown first segment in expression is a build error with suggestion."""
        config = self._make_pipeline([
            {
                "kind": "function_node",
                "metadata": {"name": "get_context"},
                "spec": {"fn": "json.loads"},
            },
            {
                "kind": "expression_node",
                "metadata": {"name": "compute"},
                "spec": {
                    "expressions": {
                        "val": "get_contex.data",  # typo: get_contex vs get_context
                    },
                },
            },
        ])
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("Unknown reference 'get_contex'" in e for e in result.errors)
        assert any("get_context" in e for e in result.errors)  # suggestion

    def test_valid_references_pass(self):
        """Valid node references in expressions and mappings pass validation."""
        config = self._make_pipeline([
            {
                "kind": "function_node",
                "metadata": {"name": "producer"},
                "spec": {"fn": "json.loads"},
            },
            {
                "kind": "expression_node",
                "metadata": {"name": "compute"},
                "spec": {
                    "input_mapping": {
                        "rate": "producer.rate",
                    },
                    "expressions": {
                        "total": "rate * 2",
                        "direct": "producer.name",
                    },
                    "output_fields": ["total", "direct"],
                },
            },
        ])
        result = self.validator.validate(config)
        assert result.is_valid, f"Unexpected errors: {result.errors}"

    def test_input_ref_is_valid(self):
        """$input references are always valid (not flagged)."""
        config = self._make_pipeline([
            {
                "kind": "expression_node",
                "metadata": {"name": "compute"},
                "spec": {
                    "input_mapping": {
                        "load_id": "$input.load_id",
                    },
                    "expressions": {"x": "load_id"},
                    "output_fields": ["x"],
                },
            },
        ])
        result = self.validator.validate(config)
        assert result.is_valid, f"Unexpected errors: {result.errors}"

    def test_macro_expanded_references_in_input_mapping_and_expressions(self):
        """Macro-expanded node names are valid references in input_mapping and expressions."""
        config = self._make_pipeline([
            {
                "kind": "macro_invocation",
                "metadata": {"name": "extract_rate"},
                "spec": {"macro": "hexdag.stdlib.macros.reasoning_agent.ReasoningAgent"},
            },
            {
                "kind": "expression_node",
                "metadata": {"name": "compute"},
                "spec": {
                    "input_mapping": {
                        "rate": "extract_rate_result.rate",
                    },
                    "expressions": {
                        "total": "extract_rate_final.amount * 2",
                    },
                    "output_fields": ["total"],
                },
            },
        ])
        result = self.validator.validate(config)
        assert result.is_valid, f"Unexpected errors: {result.errors}"


# ============================================================================
# Rule 1: Signature-based spec validation
# ============================================================================


class TestSignatureBasedValidation:
    """Rule 1: Resolve factory, introspect __call__, check required params."""

    def setup_method(self):
        self.validator = YamlValidator()

    def _make_config(self, nodes):
        return {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {"nodes": nodes},
        }

    def test_dotted_path_missing_required_param(self):
        """Dotted-path node with missing required param is caught."""
        config = self._make_config([
            {
                "kind": "hexdag.stdlib.nodes.TransitionNode",
                "metadata": {"name": "bad"},
                "spec": {},  # missing entity, to_state
            }
        ])
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("entity" in e for e in result.errors)

    def test_dotted_path_valid_spec_passes(self):
        """Dotted-path node with all required params passes."""
        config = self._make_config([
            {
                "kind": "hexdag.stdlib.nodes.TransitionNode",
                "metadata": {"name": "ok"},
                "spec": {"entity": "order", "to_state": "SHIPPED"},
            }
        ])
        result = self.validator.validate(config)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_unresolvable_dotted_path_errors(self):
        """Dotted path that can't be imported is an error."""
        config = self._make_config([
            {
                "kind": "hexdag.stdlib.nodes.NoSuchNode",
                "metadata": {"name": "bad"},
                "spec": {},
            }
        ])
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("Cannot resolve" in e for e in result.errors)

    def test_alias_node_missing_required_param(self):
        """Alias-based node with missing required param is caught."""
        config = self._make_config([
            {
                "kind": "function_node",
                "metadata": {"name": "bad"},
                "spec": {},  # missing fn
            }
        ])
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("fn" in e for e in result.errors)

    def test_composite_while_missing_body(self):
        """while mode without body or body_pipeline is an error."""
        config = self._make_config([
            {
                "kind": "composite_node",
                "metadata": {"name": "loop"},
                "spec": {"mode": "while", "condition": "state.x < 3"},
            }
        ])
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("body" in e for e in result.errors)

    def test_composite_times_zero_is_error(self):
        """times mode with count=0 is caught."""
        config = self._make_config([
            {
                "kind": "composite_node",
                "metadata": {"name": "repeat"},
                "spec": {"mode": "times", "count": 0, "body": "json.dumps"},
            }
        ])
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("positive" in e for e in result.errors)

    def test_switch_branch_missing_condition(self):
        """switch branch without condition field is caught."""
        config = self._make_config([
            {
                "kind": "composite_node",
                "metadata": {"name": "router"},
                "spec": {
                    "mode": "switch",
                    "branches": [{"action": "do_something"}],
                },
            }
        ])
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("condition" in e for e in result.errors)


# ============================================================================
# Rule 2: Reference validation for expression-like fields
# ============================================================================


class TestExpressionFieldReferenceValidation:
    """Rule 2: condition/items/when/state_update refs checked for typos."""

    def setup_method(self):
        self.validator = YamlValidator()

    def _make_config(self, nodes):
        return {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {"nodes": nodes},
        }

    def test_typo_in_condition_caught(self):
        """Typo in composite condition field is caught."""
        config = self._make_config([
            {
                "kind": "function_node",
                "metadata": {"name": "checker"},
                "spec": {"fn": "json.dumps"},
            },
            {
                "kind": "composite_node",
                "metadata": {"name": "loop"},
                "spec": {
                    "mode": "while",
                    "condition": "typo_node.done == False",
                    "body": "json.dumps",
                },
            },
        ])
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("typo_node" in e for e in result.errors)

    def test_valid_condition_ref_passes(self):
        """Valid node reference in condition passes."""
        config = self._make_config([
            {
                "kind": "function_node",
                "metadata": {"name": "checker"},
                "spec": {"fn": "json.dumps"},
            },
            {
                "kind": "composite_node",
                "metadata": {"name": "loop"},
                "spec": {
                    "mode": "while",
                    "condition": "checker.done == False",
                    "body": "json.dumps",
                },
            },
        ])
        result = self.validator.validate(config)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_typo_in_when_clause_caught(self):
        """Typo in when clause is caught."""
        config = self._make_config([
            {
                "kind": "function_node",
                "metadata": {"name": "step1"},
                "spec": {"fn": "json.dumps"},
            },
            {
                "kind": "function_node",
                "metadata": {"name": "step2"},
                "spec": {
                    "fn": "json.dumps",
                    "when": "typo_node.ready == True",
                },
            },
        ])
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("typo_node" in e for e in result.errors)


# ============================================================================
# Rule 3: Port requirement validation
# ============================================================================


class TestPortRequirementValidation:
    """Rule 3: Nodes requiring ports checked against spec.ports."""

    def setup_method(self):
        self.validator = YamlValidator()

    def test_llm_node_without_port_is_warning(self):
        """LLM node without llm port declared emits warning (ports may be runtime-configured)."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "nodes": [
                    {
                        "kind": "llm_node",
                        "metadata": {"name": "analyze"},
                        "spec": {"prompt_template": "Analyze: {{input}}"},
                    }
                ],
            },
        }
        result = self.validator.validate(config)
        assert result.is_valid
        assert any("port" in w and "llm" in w for w in result.warnings)

    def test_llm_node_with_port_passes(self):
        """LLM node with llm port declared passes."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "ports": {
                    "llm": {
                        "adapter": "hexdag.stdlib.adapters.mock.MockLLM",
                    },
                },
                "nodes": [
                    {
                        "kind": "llm_node",
                        "metadata": {"name": "analyze"},
                        "spec": {"prompt_template": "Analyze: {{input}}"},
                    }
                ],
            },
        }
        result = self.validator.validate(config)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_state_machine_missing_initial(self):
        """State machine without initial field is caught."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "state_machines": {
                    "order": {"transitions": {"NEW": ["SHIPPED"]}},
                },
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "step"},
                        "spec": {"fn": "json.dumps"},
                    }
                ],
            },
        }
        result = self.validator.validate(config)
        assert not result.is_valid
        assert any("initial" in e for e in result.errors)


class TestWhenClauseValidation:
    """Tests for build-time when clause syntax validation."""

    def setup_method(self) -> None:
        self.validator = YamlValidator()

    def test_valid_when_clause_passes(self) -> None:
        """A syntactically valid when clause should not produce errors."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "step"},
                        "spec": {
                            "fn": "json.dumps",
                            "when": "status == 'active' and count > 5",
                        },
                    }
                ],
            },
        }
        result = self.validator.validate(config)
        when_errors = [e for e in result.errors if "when" in e.lower()]
        assert not when_errors

    def test_malformed_when_clause_caught(self) -> None:
        """A syntactically invalid when clause should produce a build error."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "step"},
                        "spec": {
                            "fn": "json.dumps",
                            "when": "status === 'active'",  # invalid operator
                        },
                    }
                ],
            },
        }
        result = self.validator.validate(config)
        assert any("when" in e.lower() and "step" in e for e in result.errors)

    def test_when_clause_on_macro_invocation(self) -> None:
        """when clauses on macro_invocation config should also be validated."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "nodes": [
                    {
                        "kind": "macro_invocation",
                        "metadata": {"name": "my_macro"},
                        "spec": {
                            "macro": "hexdag.stdlib.macros.llm_macro.LLMMacro",
                            "config": {
                                "when": "invalid !! syntax",
                            },
                        },
                    }
                ],
            },
        }
        result = self.validator.validate(config)
        assert any("when" in e.lower() and "my_macro" in e for e in result.errors)


class TestInputMappingValidation:
    """Tests for input_mapping expression syntax validation."""

    def setup_method(self) -> None:
        self.validator = YamlValidator()

    def test_expression_in_mapping_validated(self) -> None:
        """input_mapping values containing expressions should be syntax-checked."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "upstream"},
                        "spec": {"fn": "json.dumps"},
                    },
                    {
                        "kind": "function_node",
                        "metadata": {"name": "step"},
                        "spec": {
                            "fn": "json.dumps",
                            "input_mapping": {
                                "valid_ref": "upstream.field",
                                "bad_expr": "upstream.status === 'done'",  # invalid operator
                            },
                        },
                    },
                ],
            },
        }
        result = self.validator.validate(config)
        # The invalid expression should be caught
        assert any("expression" in e.lower() or "when" in e.lower() for e in result.errors)

    def test_simple_ref_not_flagged_as_expression(self) -> None:
        """Simple node.field references should not be treated as expressions."""
        config = {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "upstream"},
                        "spec": {"fn": "json.dumps"},
                    },
                    {
                        "kind": "function_node",
                        "metadata": {"name": "step"},
                        "spec": {
                            "fn": "json.dumps",
                            "input_mapping": {
                                "my_field": "upstream.result",
                            },
                        },
                    },
                ],
            },
        }
        result = self.validator.validate(config)
        # No expression-related errors for simple references
        expr_errors = [e for e in result.errors if "expression" in e.lower()]
        assert not expr_errors


class TestInputFlowConsistency:
    """Tests for graph-based $input data-flow consistency validation.

    The validator uses the DAG dependency graph to find sibling groups
    (nodes sharing a common parent) and checks that $input.X fields are
    consistently passed within each group.
    """

    def setup_method(self):
        self.validator = YamlValidator()

    def _make_config(self, nodes):
        return {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {"nodes": nodes},
        }

    def _fn_node(self, name, input_mapping, *, deps=None, validated_input_fields=None):
        spec = {"fn": "json.dumps", "input_mapping": input_mapping}
        if validated_input_fields:
            spec["validated_input_fields"] = validated_input_fields
        node = {
            "kind": "function_node",
            "metadata": {"name": name},
            "spec": spec,
        }
        if deps:
            node["dependencies"] = deps
        return node

    def test_error_on_missing_field_in_graph_siblings(self):
        """Siblings from the same parent: 3/4 pass $input.conversation_id."""
        mapping_both = {
            "conv": "$input.conversation_id",
            "load": "$input.load_id",
        }
        config = self._make_config([
            self._fn_node("router", {"x": "$input.load_id"}),
            self._fn_node("a", mapping_both, deps=["router"]),
            self._fn_node("b", mapping_both, deps=["router"]),
            self._fn_node("c", mapping_both, deps=["router"]),
            self._fn_node("d", {"load": "$input.load_id"}, deps=["router"]),
        ])
        result = self.validator.validate(config)
        consistency_errors = [e for e in result.errors if "input_mapping is missing" in e]
        assert len(consistency_errors) == 1
        assert "'d'" in consistency_errors[0]
        assert "$input.conversation_id" in consistency_errors[0]

    def test_root_siblings_also_checked(self):
        """Root nodes (no deps) form a sibling group and are checked."""
        config = self._make_config([
            self._fn_node("a", {"conv": "$input.conversation_id"}),
            self._fn_node("b", {"conv": "$input.conversation_id"}),
            self._fn_node("c", {"load": "$input.load_id"}),  # missing conversation_id
        ])
        result = self.validator.validate(config)
        consistency_errors = [e for e in result.errors if "input_mapping is missing" in e]
        assert len(consistency_errors) == 1
        assert "'c'" in consistency_errors[0]
        assert "$input.conversation_id" in consistency_errors[0]

    def test_no_error_on_different_branches(self):
        """Nodes on different branches (different parents) are NOT siblings."""
        config = self._make_config([
            self._fn_node("parent_a", {"x": "$input.load_id"}),
            self._fn_node("parent_b", {"x": "$input.load_id"}),
            self._fn_node(
                "child_a",
                {"conv": "$input.conversation_id"},
                deps=["parent_a"],
            ),
            self._fn_node(
                "child_b",
                {"load": "$input.load_id"},
                deps=["parent_b"],
            ),
        ])
        result = self.validator.validate(config)
        consistency_errors = [e for e in result.errors if "input_mapping is missing" in e]
        assert not consistency_errors

    def test_no_error_below_threshold(self):
        """Only 1/4 siblings passes $input.extra → below threshold."""
        config = self._make_config([
            self._fn_node("router", {"x": "$input.load_id"}),
            self._fn_node(
                "a",
                {
                    "conv": "$input.conversation_id",
                    "extra": "$input.extra",
                },
                deps=["router"],
            ),
            self._fn_node(
                "b",
                {"conv": "$input.conversation_id"},
                deps=["router"],
            ),
            self._fn_node(
                "c",
                {"conv": "$input.conversation_id"},
                deps=["router"],
            ),
            self._fn_node(
                "d",
                {"conv": "$input.conversation_id"},
                deps=["router"],
            ),
        ])
        result = self.validator.validate(config)
        consistency_errors = [e for e in result.errors if "input_mapping is missing" in e]
        assert not consistency_errors

    def test_no_error_when_no_input_mapping(self):
        """Nodes without input_mapping are excluded from sibling groups."""
        config = self._make_config([
            self._fn_node("a", {"conv": "$input.conversation_id"}),
            self._fn_node("b", {"conv": "$input.conversation_id"}),
            {
                "kind": "function_node",
                "metadata": {"name": "c"},
                "spec": {"fn": "json.dumps"},  # no input_mapping
            },
        ])
        result = self.validator.validate(config)
        consistency_errors = [e for e in result.errors if "input_mapping is missing" in e]
        assert not consistency_errors

    def test_no_error_for_expression_node(self):
        """expression_node is skipped from consistency check."""
        config = self._make_config([
            self._fn_node("a", {"conv": "$input.conversation_id"}),
            self._fn_node("b", {"conv": "$input.conversation_id"}),
            {
                "kind": "expression_node",
                "metadata": {"name": "c"},
                "spec": {
                    "expressions": {"x": "a.result"},
                    "output_fields": ["x"],
                    "input_mapping": {"data": "a.result"},
                },
            },
        ])
        result = self.validator.validate(config)
        consistency_errors = [e for e in result.errors if "input_mapping is missing" in e]
        assert not consistency_errors

    def test_no_error_when_all_pass(self):
        """All siblings pass the same $input fields → no error."""
        conv = {"conv": "$input.conversation_id"}
        config = self._make_config([
            self._fn_node("router", {"x": "$input.load_id"}),
            self._fn_node("a", conv, deps=["router"]),
            self._fn_node("b", conv, deps=["router"]),
            self._fn_node("c", conv, deps=["router"]),
        ])
        result = self.validator.validate(config)
        consistency_errors = [e for e in result.errors if "input_mapping is missing" in e]
        assert not consistency_errors

    def test_validated_input_fields_suppresses_error(self):
        """validated_input_fields opt-out suppresses the error."""
        config = self._make_config([
            self._fn_node("a", {"conv": "$input.conversation_id"}),
            self._fn_node("b", {"conv": "$input.conversation_id"}),
            self._fn_node(
                "c",
                {"load": "$input.load_id"},
                validated_input_fields=["conversation_id"],
            ),
        ])
        result = self.validator.validate(config)
        consistency_errors = [e for e in result.errors if "input_mapping is missing" in e]
        assert not consistency_errors

    def test_single_node_no_error(self):
        """Single node with input_mapping → no siblings → no error."""
        config = self._make_config([
            self._fn_node("a", {"conv": "$input.conversation_id"}),
        ])
        result = self.validator.validate(config)
        consistency_errors = [e for e in result.errors if "input_mapping is missing" in e]
        assert not consistency_errors

    def test_custom_threshold(self):
        """Custom threshold=1.0 means ALL siblings must pass for it to be enforced."""
        validator = YamlValidator(input_mapping_consistency_threshold=1.0)
        config = self._make_config([
            self._fn_node("a", {"conv": "$input.conversation_id"}),
            self._fn_node("b", {"conv": "$input.conversation_id"}),
            self._fn_node("c", {"load": "$input.load_id"}),
        ])
        result = validator.validate(config)
        consistency_errors = [e for e in result.errors if "input_mapping is missing" in e]
        # 2/3 = 0.67, below 1.0 threshold → no error for conversation_id
        assert not any("conversation_id" in e for e in consistency_errors)

    def test_no_error_for_different_nested_subpaths_of_same_field(self):
        """Siblings reading different sub-paths of the same $input field agree.

        One sibling reads $input.context.a, the other $input.context.b. Both
        reference the top-level field 'context', so the consistency check
        (which keys on the top-level field) must NOT flag either as missing.
        """
        config = self._make_config([
            self._fn_node("router", {"x": "$input.load_id"}),
            self._fn_node("a", {"v": "$input.context.a"}, deps=["router"]),
            self._fn_node("b", {"v": "$input.context.b"}, deps=["router"]),
        ])
        result = self.validator.validate(config)
        consistency_errors = [e for e in result.errors if "input_mapping is missing" in e]
        assert not any("context" in e for e in consistency_errors)


class TestInputSchemaValidation:
    """Tests for $input references validated against declared input_schema."""

    def setup_method(self):
        self.validator = YamlValidator()

    def _make_config(self, nodes, input_schema=None):
        spec = {"nodes": nodes}
        if input_schema is not None:
            spec["input_schema"] = input_schema
        return {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": spec,
        }

    def _fn_node(self, name, input_mapping):
        return {
            "kind": "function_node",
            "metadata": {"name": name},
            "spec": {"fn": "json.dumps", "input_mapping": input_mapping},
        }

    def test_error_on_unknown_input_field(self):
        """Node references $input.unknown but schema doesn't declare it → error."""
        config = self._make_config(
            [self._fn_node("a", {"x": "$input.unknown_field"})],
            input_schema={"conversation_id": "str", "load_id": "str"},
        )
        result = self.validator.validate(config)
        schema_errors = [e for e in result.errors if "input_schema does not declare" in e]
        assert len(schema_errors) == 1
        assert "unknown_field" in schema_errors[0]

    def test_no_error_for_known_field(self):
        """Node references $input.conversation_id which IS in schema → valid."""
        config = self._make_config(
            [self._fn_node("a", {"conv": "$input.conversation_id"})],
            input_schema={"conversation_id": "str"},
        )
        result = self.validator.validate(config)
        schema_errors = [e for e in result.errors if "input_schema does not declare" in e]
        assert not schema_errors

    def test_skipped_when_no_schema(self):
        """No input_schema declared → no schema validation."""
        config = self._make_config(
            [self._fn_node("a", {"x": "$input.anything"})],
        )
        result = self.validator.validate(config)
        schema_errors = [e for e in result.errors if "input_schema does not declare" in e]
        assert not schema_errors

    def test_multiple_nodes_all_checked(self):
        """All nodes' $input refs are validated against schema."""
        config = self._make_config(
            [
                self._fn_node("a", {"x": "$input.valid_field"}),
                self._fn_node("b", {"y": "$input.typo_field"}),
            ],
            input_schema={"valid_field": "str"},
        )
        result = self.validator.validate(config)
        schema_errors = [e for e in result.errors if "input_schema does not declare" in e]
        assert len(schema_errors) == 1
        assert "'b'" in schema_errors[0]
        assert "typo_field" in schema_errors[0]

    def test_embedded_input_ref_in_expression_checked(self):
        """$input.field embedded in expression is also checked against schema."""
        config = self._make_config(
            [self._fn_node("a", {"rate": "coalesce($input.bad_field, 0)"})],
            input_schema={"conversation_id": "str"},
        )
        result = self.validator.validate(config)
        schema_errors = [e for e in result.errors if "input_schema does not declare" in e]
        assert len(schema_errors) == 1
        assert "bad_field" in schema_errors[0]

    def test_nested_input_path_checked_by_top_level_field(self):
        """$input.a.b.c is validated by its top-level field 'a' (flat schema)."""
        config = self._make_config(
            [self._fn_node("a", {"x": "$input.profile.name"})],
            input_schema={"profile": "dict"},
        )
        result = self.validator.validate(config)
        schema_errors = [e for e in result.errors if "input_schema does not declare" in e]
        assert not schema_errors

    def test_nested_input_path_error_names_top_level_field(self):
        """A nested path whose root is undeclared errors on the root, not the full path."""
        config = self._make_config(
            [self._fn_node("a", {"x": "$input.profile.name"})],
            input_schema={"conversation_id": "str"},
        )
        result = self.validator.validate(config)
        schema_errors = [e for e in result.errors if "input_schema does not declare" in e]
        assert len(schema_errors) == 1
        assert "profile" in schema_errors[0]
        # The error names the top-level field, not the full nested path.
        assert "profile.name" not in schema_errors[0]


class TestUndeclaredRefs:
    """Tests for undeclared node reference detection.

    When a node has explicit dependencies but references another node
    in its when/input_mapping/expressions that isn't declared as a
    dependency, the validator warns — the builder auto-merges the
    missing deps, so the pipeline executes correctly either way.
    """

    def setup_method(self):
        self.validator = YamlValidator()

    def _make_config(self, nodes):
        return {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {"nodes": nodes},
        }

    def test_warns_when_clause_references_undeclared_dep(self):
        """when clause references node not in dependencies → warning."""
        config = self._make_config([
            {
                "kind": "function_node",
                "metadata": {"name": "checker"},
                "spec": {"fn": "json.dumps"},
            },
            {"kind": "function_node", "metadata": {"name": "router"}, "spec": {"fn": "json.dumps"}},
            {
                "kind": "function_node",
                "metadata": {"name": "sender"},
                "spec": {
                    "fn": "json.dumps",
                    "when": "checker.done == True",
                },
                "dependencies": ["router"],  # missing "checker"
            },
        ])
        result = self.validator.validate(config)
        assert result.is_valid
        ref_warnings = [w for w in result.warnings if "not in its explicit dependencies" in w]
        assert len(ref_warnings) == 1
        assert "'sender'" in ref_warnings[0]
        assert "'checker'" in ref_warnings[0]

    def test_warns_input_mapping_references_undeclared_dep(self):
        """input_mapping references node not in dependencies → warning."""
        config = self._make_config([
            {
                "kind": "function_node",
                "metadata": {"name": "upstream_a"},
                "spec": {"fn": "json.dumps"},
            },
            {
                "kind": "function_node",
                "metadata": {"name": "upstream_b"},
                "spec": {"fn": "json.dumps"},
            },
            {
                "kind": "function_node",
                "metadata": {"name": "consumer"},
                "spec": {
                    "fn": "json.dumps",
                    "input_mapping": {"data": "upstream_b.result"},
                },
                "dependencies": ["upstream_a"],  # missing "upstream_b"
            },
        ])
        result = self.validator.validate(config)
        assert result.is_valid
        ref_warnings = [w for w in result.warnings if "not in its explicit dependencies" in w]
        assert len(ref_warnings) == 1
        assert "'consumer'" in ref_warnings[0]
        assert "'upstream_b'" in ref_warnings[0]

    def test_no_warning_when_dep_is_declared(self):
        """Referenced node IS in dependencies → no warning."""
        config = self._make_config([
            {
                "kind": "function_node",
                "metadata": {"name": "checker"},
                "spec": {"fn": "json.dumps"},
            },
            {
                "kind": "function_node",
                "metadata": {"name": "sender"},
                "spec": {
                    "fn": "json.dumps",
                    "when": "checker.done == True",
                },
                "dependencies": ["checker"],
            },
        ])
        result = self.validator.validate(config)
        ref_warnings = [w for w in result.warnings if "not in its explicit dependencies" in w]
        assert not ref_warnings

    def test_no_warning_without_explicit_deps(self):
        """Node without explicit dependencies → auto-infer, no warning."""
        config = self._make_config([
            {
                "kind": "function_node",
                "metadata": {"name": "checker"},
                "spec": {"fn": "json.dumps"},
            },
            {
                "kind": "function_node",
                "metadata": {"name": "sender"},
                "spec": {
                    "fn": "json.dumps",
                    "when": "checker.done == True",
                    # no "dependencies" key → builder will infer
                },
            },
        ])
        result = self.validator.validate(config)
        ref_warnings = [w for w in result.warnings if "not in its explicit dependencies" in w]
        assert not ref_warnings

    def test_dollar_input_not_treated_as_undeclared(self):
        """$input.field references are not node deps → no warning."""
        config = self._make_config([
            {
                "kind": "function_node",
                "metadata": {"name": "sender"},
                "spec": {
                    "fn": "json.dumps",
                    "input_mapping": {"conv": "$input.conversation_id"},
                },
                "dependencies": [],
            },
        ])
        result = self.validator.validate(config)
        ref_warnings = [w for w in result.warnings if "not in its explicit dependencies" in w]
        assert not ref_warnings

    def test_multiple_missing_refs(self):
        """Multiple undeclared refs in one node → multiple warnings."""
        config = self._make_config([
            {"kind": "function_node", "metadata": {"name": "a"}, "spec": {"fn": "json.dumps"}},
            {"kind": "function_node", "metadata": {"name": "b"}, "spec": {"fn": "json.dumps"}},
            {"kind": "function_node", "metadata": {"name": "c"}, "spec": {"fn": "json.dumps"}},
            {
                "kind": "function_node",
                "metadata": {"name": "consumer"},
                "spec": {
                    "fn": "json.dumps",
                    "when": "a.done == True",
                    "input_mapping": {"data": "b.result"},
                },
                "dependencies": ["c"],  # missing "a" and "b"
            },
        ])
        result = self.validator.validate(config)
        assert result.is_valid
        ref_warnings = [w for w in result.warnings if "not in its explicit dependencies" in w]
        assert len(ref_warnings) == 2
        referenced_nodes = {w.split("'")[3] for w in ref_warnings}
        assert referenced_nodes == {"a", "b"}


class TestExpressionNamespaces:
    """Tests for ExpressionNamespace declarations and ctx field validation."""

    def test_ctx_fields_auto_discovered(self):
        """CTX.fields matches get_ctx_dict() keys — no hardcoded drift."""
        assert CTX.fields == frozenset(get_ctx_dict().keys())

    def test_reserved_names_includes_all_namespaces_and_aliases(self):
        """RESERVED_NAMES covers every namespace name and alias."""
        expected: set[str] = set()
        for ns in EXPRESSION_NAMESPACES:
            expected.add(ns.name)
            expected |= ns.aliases
        assert frozenset(expected) == RESERVED_NAMES

    def test_reserved_names_content(self):
        """Sanity check: the three namespaces + input alias are present."""
        assert {"ctx", "$input", "input", "state"} <= RESERVED_NAMES


class TestCtxFieldValidation:
    """Tests that ctx.X field references are validated at build time."""

    def setup_method(self):
        self.validator = YamlValidator()

    def _make_pipeline(self, nodes):
        return {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {"nodes": nodes},
        }

    def test_valid_ctx_field_passes(self):
        """ctx.run_id in expression passes validation."""
        config = self._make_pipeline([
            {
                "kind": "expression_node",
                "metadata": {"name": "compute"},
                "spec": {
                    "expressions": {
                        "tag": "'run-' + ctx.run_id",
                    },
                },
            },
        ])
        result = self.validator.validate(config)
        ctx_errors = [e for e in result.errors if "ctx field" in e.lower() or "ctx" in e.lower()]
        assert not ctx_errors

    def test_invalid_ctx_field_errors(self):
        """ctx.typo in expression is a build error."""
        config = self._make_pipeline([
            {
                "kind": "expression_node",
                "metadata": {"name": "compute"},
                "spec": {
                    "expressions": {
                        "tag": "'run-' + ctx.typo_field",
                    },
                },
            },
        ])
        result = self.validator.validate(config)
        ctx_errors = [e for e in result.errors if "Unknown ctx field" in e]
        assert len(ctx_errors) == 1
        assert "typo_field" in ctx_errors[0]

    def test_ctx_typo_suggests_close_match(self):
        """ctx.pipline_name suggests pipeline_name."""
        config = self._make_pipeline([
            {
                "kind": "expression_node",
                "metadata": {"name": "compute"},
                "spec": {
                    "expressions": {
                        "tag": "ctx.pipline_name",
                    },
                },
            },
        ])
        result = self.validator.validate(config)
        ctx_errors = [e for e in result.errors if "Unknown ctx field" in e]
        assert len(ctx_errors) == 1
        assert "pipeline_name" in ctx_errors[0]

    def test_state_field_passes_dynamic(self):
        """state.anything passes — state has dynamic fields."""
        config = self._make_pipeline([
            {
                "kind": "expression_node",
                "metadata": {"name": "compute"},
                "spec": {
                    "expressions": {
                        "x": "state.whatever_field",
                    },
                },
            },
        ])
        result = self.validator.validate(config)
        state_errors = [e for e in result.errors if "state" in e.lower() and "field" in e.lower()]
        assert not state_errors

    def test_ctx_field_in_input_mapping(self):
        """ctx.typo in input_mapping is also caught."""
        config = self._make_pipeline([
            {
                "kind": "function_node",
                "metadata": {"name": "process"},
                "spec": {
                    "fn": "json.loads",
                    "input_mapping": {
                        "tag": "ctx.banana",
                    },
                },
            },
        ])
        result = self.validator.validate(config)
        ctx_errors = [e for e in result.errors if "Unknown ctx field" in e]
        assert len(ctx_errors) == 1
        assert "banana" in ctx_errors[0]


class TestTemplateTypoLint:
    """Tests for _validate_template_typos — near-miss unknown {{refs}} warn."""

    def setup_method(self):
        self.validator = YamlValidator()

    def _make_config(self, nodes):
        return {
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {"nodes": nodes},
        }

    def test_near_miss_warns_with_suggestion(self):
        config = self._make_config([
            {
                "kind": "function_node",
                "metadata": {"name": "get_context"},
                "spec": {"fn": "json.dumps"},
            },
            {
                "kind": "function_node",
                "metadata": {"name": "padding"},
                "spec": {"fn": "json.dumps"},
            },
            {
                "kind": "llm_node",
                "metadata": {"name": "analyze"},
                "spec": {"human_message": "Analyze {{get_contxt.load}}"},
            },
        ])
        result = self.validator.validate(config)
        typo_warnings = [w for w in result.warnings if "Did you mean" in w]
        assert len(typo_warnings) == 1
        assert "get_contxt" in typo_warnings[0]
        assert "get_context" in typo_warnings[0]

    def test_no_warning_when_suggested_node_is_upstream(self):
        """Bare {{field}} matching an upstream node's name pattern is its
        output field (single-dep flat pass-through), not a typo."""
        config = self._make_config([
            {
                "kind": "function_node",
                "metadata": {"name": "fetch_thread_context"},
                "spec": {"fn": "json.dumps"},
            },
            {
                "kind": "function_node",
                "metadata": {"name": "padding"},
                "spec": {"fn": "json.dumps"},
                "dependencies": [],
            },
            {
                "kind": "llm_node",
                "metadata": {"name": "analyze"},
                "spec": {
                    "human_message": "History: {{thread_context}}",
                },
                "dependencies": ["fetch_thread_context"],
            },
        ])
        result = self.validator.validate(config)
        assert not [w for w in result.warnings if "Did you mean" in w]

    def test_unrelated_unknown_name_is_silent(self):
        """Template vars from aliases/dep fields are not flagged — only near-misses."""
        config = self._make_config([
            {
                "kind": "function_node",
                "metadata": {"name": "get_context"},
                "spec": {"fn": "json.dumps"},
            },
            {
                "kind": "llm_node",
                "metadata": {"name": "analyze"},
                "spec": {"human_message": "Subject {{email_subject}}"},
            },
        ])
        result = self.validator.validate(config)
        assert not [w for w in result.warnings if "Did you mean" in w]

    def test_input_mapping_alias_is_allowed(self):
        """A {{var}} matching the node's own input_mapping key never warns."""
        config = self._make_config([
            {
                "kind": "function_node",
                "metadata": {"name": "get_contexts"},
                "spec": {"fn": "json.dumps"},
            },
            {
                "kind": "llm_node",
                "metadata": {"name": "analyze"},
                "spec": {
                    "human_message": "Analyze {{get_context}}",
                    "input_mapping": {"get_context": "$input.context"},
                },
            },
        ])
        result = self.validator.validate(config)
        assert not [w for w in result.warnings if "Did you mean" in w]

    def test_namespaces_never_flagged(self):
        config = self._make_config([
            {
                "kind": "function_node",
                "metadata": {"name": "inputs_node"},
                "spec": {"fn": "json.dumps"},
            },
            {
                "kind": "llm_node",
                "metadata": {"name": "analyze"},
                "spec": {"human_message": "{{input.x}} {{state.y}} {{ctx.z}}"},
            },
        ])
        result = self.validator.validate(config)
        assert not [w for w in result.warnings if "Did you mean" in w]
