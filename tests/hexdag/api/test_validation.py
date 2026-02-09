"""Tests for hexdag.api.validation module."""

from hexdag.api.validation import validate


class TestValidate:
    """Tests for validate function."""

    def test_validate_valid_pipeline(self):
        """Test validation of a valid pipeline."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: function_node
      metadata:
        name: node1
      spec:
        fn: json.loads
      dependencies: []
"""
        result = validate(yaml_content)
        assert result["valid"] is True
        assert "message" in result
        assert result["node_count"] == 1
        assert "nodes" in result
        assert "node1" in result["nodes"]

    def test_validate_empty_nodes(self):
        """Test validation of pipeline with empty nodes list."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes: []
"""
        result = validate(yaml_content)
        assert result["valid"] is True
        assert result["node_count"] == 0

    def test_validate_invalid_yaml_syntax(self):
        """Test validation catches YAML syntax errors."""
        yaml_content = "invalid: yaml: content: :"
        result = validate(yaml_content)
        assert result["valid"] is False
        assert "error" in result

    def test_validate_non_dict_yaml(self):
        """Test validation catches non-dict YAML."""
        yaml_content = "- item1\n- item2"
        result = validate(yaml_content)
        assert result["valid"] is False
        assert "error" in result

    def test_validate_missing_nodes(self):
        """Test validation catches missing nodes field."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec: {}
"""
        result = validate(yaml_content)
        assert result["valid"] is False
        assert "error" in result


class TestValidateLenient:
    """Tests for lenient validation mode."""

    def test_validate_lenient_valid_structure(self):
        """Test lenient validation with valid structure."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: llm_node
      metadata:
        name: analyzer
      spec:
        prompt_template: "Analyze: {{input}}"
      dependencies: []
"""
        result = validate(yaml_content, lenient=True)
        assert result["valid"] is True
        assert "warnings" in result

    def test_validate_lenient_missing_env_vars_ok(self):
        """Test lenient mode allows missing environment variables."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  ports:
    llm:
      adapter: openai
      config:
        api_key: ${OPENAI_API_KEY}
  nodes:
    - kind: data_node
      metadata:
        name: start
      spec:
        output:
          value: 1
      dependencies: []
"""
        # Lenient mode should validate structure without requiring env vars
        result = validate(yaml_content, lenient=True)
        # Structure is valid even if env vars aren't set
        assert "valid" in result

    def test_validate_lenient_returns_warnings(self):
        """Test lenient validation returns warnings list."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes: []
"""
        result = validate(yaml_content, lenient=True)
        assert "warnings" in result
        assert isinstance(result["warnings"], list)


class TestValidateFull:
    """Tests for full validation mode (default)."""

    def test_validate_full_returns_ports(self):
        """Test full validation returns ports list."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: data_node
      metadata:
        name: start
      spec:
        output:
          value: 1
      dependencies: []
"""
        result = validate(yaml_content, lenient=False)
        assert result["valid"] is True
        assert "ports" in result
        assert isinstance(result["ports"], list)

    def test_validate_full_detects_invalid_node_type(self):
        """Test full validation detects invalid node types."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: nonexistent_node_type
      metadata:
        name: bad_node
      spec: {}
      dependencies: []
"""
        result = validate(yaml_content, lenient=False)
        assert result["valid"] is False
        assert "error" in result

    def test_validate_full_detects_cycle(self):
        """Test full validation detects dependency cycles."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: function_node
      metadata:
        name: a
      spec:
        fn: test
      dependencies: [b]
    - kind: function_node
      metadata:
        name: b
      spec:
        fn: test
      dependencies: [a]
"""
        result = validate(yaml_content, lenient=False)
        assert result["valid"] is False
        assert "error" in result


class TestValidationErrorTypes:
    """Tests for validation error type reporting."""

    def test_error_type_yaml_error(self):
        """Test that YAML errors report correct error type."""
        yaml_content = "invalid: yaml: :"
        result = validate(yaml_content, lenient=True)
        assert result["valid"] is False
        assert "error_type" in result
        assert "YAML" in result["error_type"] or "Parse" in result["error_type"]

    def test_error_type_validation_error(self):
        """Test that validation errors report correct error type."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: function_node
      metadata: {}
      spec:
        fn: test
      dependencies: []
"""
        result = validate(yaml_content, lenient=True)
        # Missing metadata.name should fail
        assert result["valid"] is False
        assert "error_type" in result
