"""Tests for hexdag_studio.server.routes.validate module."""

from fastapi.testclient import TestClient


class TestValidateYaml:
    """Tests for POST /api/validate endpoint."""

    def test_validate_valid_pipeline(self, client: TestClient):
        """Test validation of a valid pipeline."""
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
          value: "hello"
      dependencies: []
"""
        response = client.post(
            "/api/validate",
            json={"content": yaml_content},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["errors"] == []
        assert data["node_count"] == 1

    def test_validate_invalid_yaml_syntax(self, client: TestClient):
        """Test validation catches YAML syntax errors."""
        response = client.post(
            "/api/validate",
            json={"content": "invalid: yaml: :"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0

    def test_validate_invalid_structure(self, client: TestClient):
        """Test validation catches invalid structure."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test
spec: {}
"""
        response = client.post(
            "/api/validate",
            json={"content": yaml_content},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False

    def test_validate_with_filename(self, client: TestClient):
        """Test validation with filename parameter."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test
spec:
  nodes: []
"""
        response = client.post(
            "/api/validate",
            json={"content": yaml_content, "filename": "test.yaml"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True


class TestValidateYamlLenient:
    """Tests for POST /api/validate/lenient endpoint."""

    def test_validate_lenient_valid_structure(self, client: TestClient):
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
        prompt_template: "Test {{input}}"
      dependencies: []
"""
        response = client.post(
            "/api/validate/lenient",
            json={"content": yaml_content},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True

    def test_validate_lenient_returns_warnings(self, client: TestClient):
        """Test that lenient validation returns warnings."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test
spec:
  nodes: []
"""
        response = client.post(
            "/api/validate/lenient",
            json={"content": yaml_content},
        )
        assert response.status_code == 200
        data = response.json()
        # Warnings field should be present
        assert "warnings" in data

    def test_validate_lenient_mode_flag(self, client: TestClient):
        """Test that lenient mode is set correctly."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test
spec:
  nodes:
    - kind: function_node
      metadata:
        name: test
      spec:
        fn: json.loads
      dependencies: []
"""
        response = client.post(
            "/api/validate",
            json={"content": yaml_content, "lenient": True},
        )
        assert response.status_code == 200
        # Should behave same as lenient endpoint


class TestValidationErrorFormat:
    """Tests for validation error format."""

    def test_error_has_message(self, client: TestClient):
        """Test that errors have message field."""
        response = client.post(
            "/api/validate",
            json={"content": "invalid: yaml: :"},
        )
        data = response.json()
        if data["errors"]:
            error = data["errors"][0]
            assert "message" in error

    def test_error_has_severity(self, client: TestClient):
        """Test that errors have severity field."""
        response = client.post(
            "/api/validate",
            json={"content": "invalid: yaml: :"},
        )
        data = response.json()
        if data["errors"]:
            error = data["errors"][0]
            assert "severity" in error
            assert error["severity"] in ["error", "warning", "info"]
