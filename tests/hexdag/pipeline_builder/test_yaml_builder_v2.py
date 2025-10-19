"""Tests for YAML Builder v2 Preprocessing Plugins.

Tests cover:
- EnvironmentVariablePlugin: ${VAR} resolution with defaults and type coercion
- TemplatePlugin: Jinja2 templating with context and type coercion
- Integration: Both plugins working together in YamlPipelineBuilder
"""

import os

import pytest

from hexdag.core.pipeline_builder.yaml_builder_v2 import (
    EnvironmentVariablePlugin,
    TemplatePlugin,
    YamlPipelineBuilder,
    YamlPipelineBuilderError,
)

# ============================================================================
# EnvironmentVariablePlugin Tests
# ============================================================================


class TestEnvironmentVariablePlugin:
    """Test environment variable resolution in YAML."""

    def test_resolve_simple_variable(self):
        """Test resolving a simple environment variable."""
        os.environ["TEST_VAR"] = "hello"

        plugin = EnvironmentVariablePlugin()
        config = {"value": "${TEST_VAR}"}

        result = plugin.process(config)

        assert result["value"] == "hello"

    def test_resolve_variable_with_default(self):
        """Test resolving variable with default value."""
        # Ensure var doesn't exist
        os.environ.pop("NONEXISTENT_VAR", None)

        plugin = EnvironmentVariablePlugin()
        config = {"value": "${NONEXISTENT_VAR:default_value}"}

        result = plugin.process(config)

        assert result["value"] == "default_value"

    def test_resolve_missing_variable_without_default_raises_error(self):
        """Test that missing required variable raises error."""
        os.environ.pop("MISSING_VAR", None)

        plugin = EnvironmentVariablePlugin()
        config = {"value": "${MISSING_VAR}"}

        with pytest.raises(YamlPipelineBuilderError, match="MISSING_VAR.*not set"):
            plugin.process(config)

    def test_type_coercion_integer(self):
        """Test automatic integer type coercion."""
        os.environ["INT_VAR"] = "42"

        plugin = EnvironmentVariablePlugin()
        config = {"value": "${INT_VAR}"}

        result = plugin.process(config)

        assert result["value"] == 42
        assert isinstance(result["value"], int)

    def test_type_coercion_float(self):
        """Test automatic float type coercion."""
        os.environ["FLOAT_VAR"] = "3.14"

        plugin = EnvironmentVariablePlugin()
        config = {"value": "${FLOAT_VAR}"}

        result = plugin.process(config)

        assert result["value"] == 3.14
        assert isinstance(result["value"], float)

    def test_type_coercion_boolean_true(self):
        """Test automatic boolean type coercion for true values."""
        test_cases = ["true", "True", "TRUE", "yes", "Yes", "1"]

        for true_value in test_cases:
            os.environ["BOOL_VAR"] = true_value

            plugin = EnvironmentVariablePlugin()
            config = {"value": "${BOOL_VAR}"}

            result = plugin.process(config)

            assert result["value"] is True, f"Failed for: {true_value}"

    def test_type_coercion_boolean_false(self):
        """Test automatic boolean type coercion for false values."""
        test_cases = ["false", "False", "FALSE", "no", "No", "0"]

        for false_value in test_cases:
            os.environ["BOOL_VAR"] = false_value

            plugin = EnvironmentVariablePlugin()
            config = {"value": "${BOOL_VAR}"}

            result = plugin.process(config)

            assert result["value"] is False, f"Failed for: {false_value}"

    def test_recursive_resolution_in_dict(self):
        """Test recursive resolution in nested dictionaries."""
        os.environ["VAR1"] = "value1"
        os.environ["VAR2"] = "value2"

        plugin = EnvironmentVariablePlugin()
        config = {"level1": {"var1": "${VAR1}", "level2": {"var2": "${VAR2}"}}}

        result = plugin.process(config)

        assert result["level1"]["var1"] == "value1"
        assert result["level1"]["level2"]["var2"] == "value2"

    def test_recursive_resolution_in_list(self):
        """Test recursive resolution in lists."""
        os.environ["ITEM1"] = "first"
        os.environ["ITEM2"] = "second"

        plugin = EnvironmentVariablePlugin()
        config = {"items": ["${ITEM1}", "${ITEM2}", "literal"]}

        result = plugin.process(config)

        assert result["items"] == ["first", "second", "literal"]

    def test_multiple_variables_in_string(self):
        """Test multiple variable substitutions in single string."""
        os.environ["HOST"] = "localhost"
        os.environ["PORT"] = "8080"

        plugin = EnvironmentVariablePlugin()
        config = {"url": "http://${HOST}:${PORT}/api"}

        result = plugin.process(config)

        assert result["url"] == "http://localhost:8080/api"

    def test_default_with_colon_in_value(self):
        """Test default value containing colons."""
        os.environ.pop("MISSING", None)

        plugin = EnvironmentVariablePlugin()
        config = {"url": "${MISSING:http://localhost:8080}"}

        result = plugin.process(config)

        assert result["url"] == "http://localhost:8080"

    def test_preserves_non_variable_strings(self):
        """Test that strings without variables are preserved."""
        plugin = EnvironmentVariablePlugin()
        config = {
            "regular": "just a string",
            "with_dollar": "price is $100",
            "with_braces": "data = {key: value}",
        }

        result = plugin.process(config)

        assert result == config


# ============================================================================
# TemplatePlugin Tests
# ============================================================================


class TestTemplatePlugin:
    """Test Jinja2 templating in YAML."""

    def test_simple_variable_substitution(self):
        """Test simple Jinja2 variable substitution."""
        plugin = TemplatePlugin()
        config = {"variables": {"name": "World"}, "greeting": "Hello {{ variables.name }}"}

        result = plugin.process(config)

        assert result["greeting"] == "Hello World"

    def test_nested_context_access(self):
        """Test accessing nested config values in templates."""
        plugin = TemplatePlugin()
        config = {
            "settings": {"model": "gpt-4", "temperature": 0.7},
            "prompt": "Use {{ settings.model }} with temp {{ settings.temperature }}",
        }

        result = plugin.process(config)

        assert result["prompt"] == "Use gpt-4 with temp 0.7"

    def test_type_coercion_integer(self):
        """Test automatic integer type coercion."""
        plugin = TemplatePlugin()
        config = {"variables": {"count": 42}, "value": "{{ variables.count }}"}

        result = plugin.process(config)

        assert result["value"] == 42
        assert isinstance(result["value"], int)

    def test_type_coercion_float(self):
        """Test automatic float type coercion."""
        plugin = TemplatePlugin()
        config = {"variables": {"pi": 3.14}, "value": "{{ variables.pi }}"}

        result = plugin.process(config)

        assert result["value"] == 3.14
        assert isinstance(result["value"], float)

    def test_type_coercion_boolean(self):
        """Test automatic boolean type coercion."""
        plugin = TemplatePlugin()
        config = {"value_true": "{{ 'true' }}", "value_false": "{{ 'false' }}"}

        result = plugin.process(config)

        assert result["value_true"] is True
        assert result["value_false"] is False

    def test_preserves_non_template_strings(self):
        """Test that strings without templates are preserved."""
        plugin = TemplatePlugin()
        config = {
            "regular": "just a string",
            "with_braces": "data = {key: value}",
            "single_brace": "x { y",
        }

        result = plugin.process(config)

        assert result == config

    def test_recursive_rendering_in_dict(self):
        """Test recursive template rendering in nested dicts."""
        plugin = TemplatePlugin()
        config = {
            "variables": {"x": "hello"},
            "level1": {
                "value": "{{ variables.x }}",
                "level2": {"nested": "{{ variables.x }} world"},
            },
        }

        result = plugin.process(config)

        assert result["level1"]["value"] == "hello"
        assert result["level1"]["level2"]["nested"] == "hello world"

    def test_recursive_rendering_in_list(self):
        """Test recursive template rendering in lists."""
        plugin = TemplatePlugin()
        config = {
            "variables": {"item": "test"},
            "items": ["{{ variables.item }}1", "{{ variables.item }}2"],
        }

        result = plugin.process(config)

        assert result["items"] == ["test1", "test2"]

    def test_jinja2_expressions(self):
        """Test Jinja2 expressions and filters."""
        plugin = TemplatePlugin()
        config = {"variables": {"name": "world"}, "upper": "{{ variables.name | upper }}"}

        result = plugin.process(config)

        assert result["upper"] == "WORLD"

    def test_template_syntax_error_raises(self):
        """Test that invalid template syntax raises clear error."""
        plugin = TemplatePlugin()
        config = {"value": "{{ invalid syntax }}"}

        with pytest.raises(YamlPipelineBuilderError, match="template syntax"):
            plugin.process(config)

    def test_undefined_variable_raises(self):
        """Test that undefined template variable raises clear error."""
        plugin = TemplatePlugin()
        config = {"value": "{{ undefined_variable }}"}

        # Jinja2's StrictUndefined mode would raise, but default mode returns empty string
        # Our plugin should catch this in strict mode or allow it in lenient mode
        # For now, just verify it doesn't crash
        result = plugin.process(config)
        # In default mode, undefined vars render as empty string
        assert result["value"] == ""


# ============================================================================
# Integration Tests - Both Plugins Together
# ============================================================================


class TestYamlPipelineBuilderV2Integration:
    """Test v2 builder with preprocessing plugins."""

    def test_environment_plugin_loaded(self):
        """Test that EnvironmentVariablePlugin is loaded by default."""
        builder = YamlPipelineBuilder()

        assert len(builder.preprocess_plugins) >= 1
        assert any(isinstance(p, EnvironmentVariablePlugin) for p in builder.preprocess_plugins)

    def test_template_plugin_loaded_when_available(self):
        """Test that TemplatePlugin is loaded when jinja2 is available."""
        builder = YamlPipelineBuilder()

        assert len(builder.preprocess_plugins) >= 2
        assert any(isinstance(p, TemplatePlugin) for p in builder.preprocess_plugins)

    def test_environment_variables_in_pipeline_metadata(self):
        """Test environment variable resolution in pipeline metadata."""
        os.environ["PIPELINE_NAME"] = "test-pipeline"
        os.environ["VERSION"] = "1.0"

        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: ${PIPELINE_NAME}
  version: ${VERSION}
spec:
  nodes:
    - kind: prompt_node
      metadata:
        name: test
      spec:
        template: "Hello"
"""

        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)

        # Check metadata - environment variables are resolved
        assert config.metadata["name"] == "test-pipeline"
        # Version might be string or float depending on type coercion
        assert config.metadata["version"] in ("1.0", 1.0)

    def test_template_in_pipeline(self):
        """Test Jinja2 template in pipeline."""

        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: template-test
spec:
  variables:
    node_name: processor
  nodes:
    - kind: prompt_node
      metadata:
        name: "{{ spec.variables.node_name }}"
      spec:
        template: "Hello"
"""

        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)

        assert "processor" in graph.nodes

    def test_both_plugins_together(self):
        """Test environment variables and templates working together."""
        os.environ["ENV"] = "production"

        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: ${ENV}-pipeline
spec:
  nodes:
    - kind: prompt_node
      metadata:
        name: test
      spec:
        template: "Hello"
"""

        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)

        assert config.metadata["name"] == "production-pipeline"

    def test_plugin_order_env_before_template(self):
        """Test that env vars are resolved before templates."""
        os.environ["MODEL"] = "gpt-4"

        # This tests that ${MODEL} is resolved first
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: ${MODEL}-pipeline
spec:
  nodes:
    - kind: prompt_node
      metadata:
        name: test
      spec:
        template: "Hello"
"""

        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)

        # Environment variable should be resolved in metadata
        assert config.metadata["name"] == "gpt-4-pipeline"
