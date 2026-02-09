"""Tests for hexdag_studio.server.routes.execute module."""

from fastapi.testclient import TestClient


class TestExecutePipeline:
    """Tests for POST /api/execute endpoint."""

    def test_execute_simple_pipeline(self, client: TestClient):
        """Test executing a simple pipeline."""
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
          value: "hello world"
      dependencies: []
"""
        response = client.post(
            "/api/execute",
            json={"content": yaml_content},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "duration_ms" in data
        assert data["error"] is None

    def test_execute_with_inputs(self, client: TestClient):
        """Test executing with input values."""
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
          message: "processed"
      dependencies: []
"""
        response = client.post(
            "/api/execute",
            json={
                "content": yaml_content,
                "inputs": {"user_input": "test data"},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_execute_invalid_pipeline_returns_error(self, client: TestClient):
        """Test executing an invalid pipeline returns error."""
        response = client.post(
            "/api/execute",
            json={"content": "invalid: yaml: :"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["error"] is not None

    def test_execute_returns_node_results(self, client: TestClient):
        """Test that execute returns node results."""
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
        response = client.post(
            "/api/execute",
            json={"content": yaml_content},
        )
        data = response.json()
        assert "nodes" in data
        assert isinstance(data["nodes"], list)

    def test_execute_with_timeout(self, client: TestClient):
        """Test executing with custom timeout."""
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
        response = client.post(
            "/api/execute",
            json={"content": yaml_content, "timeout": 60.0},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestDryRun:
    """Tests for POST /api/execute/dry-run endpoint."""

    def test_dry_run_valid_pipeline(self, client: TestClient):
        """Test dry run of a valid pipeline."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: data_node
      metadata:
        name: a
      spec:
        output:
          value: 1
      dependencies: []
    - kind: data_node
      metadata:
        name: b
      spec:
        output:
          value: 2
      dependencies: [a]
"""
        response = client.post(
            "/api/execute/dry-run",
            json={"content": yaml_content},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["node_count"] == 2
        assert "execution_order" in data
        assert "waves" in data

    def test_dry_run_invalid_pipeline(self, client: TestClient):
        """Test dry run of an invalid pipeline."""
        response = client.post(
            "/api/execute/dry-run",
            json={"content": "invalid: yaml: :"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert "error" in data

    def test_dry_run_returns_dependency_map(self, client: TestClient):
        """Test that dry run returns dependency information."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: data_node
      metadata:
        name: source
      spec:
        output:
          value: 1
      dependencies: []
    - kind: data_node
      metadata:
        name: consumer
      spec:
        output:
          value: 2
      dependencies: [source]
"""
        response = client.post(
            "/api/execute/dry-run",
            json={"content": yaml_content},
        )
        data = response.json()
        assert "dependency_map" in data
        assert "source" in data["dependency_map"]
        assert "consumer" in data["dependency_map"]


class TestExecuteStream:
    """Tests for POST /api/execute/stream endpoint."""

    def test_stream_returns_sse(self, client: TestClient):
        """Test that stream endpoint returns Server-Sent Events."""
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
        response = client.post(
            "/api/execute/stream",
            json={"content": yaml_content},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

    def test_stream_contains_events(self, client: TestClient):
        """Test that stream contains expected events."""
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
        response = client.post(
            "/api/execute/stream",
            json={"content": yaml_content},
        )
        # Read the stream content
        content = response.text
        # Should contain event markers
        assert "event:" in content
        assert "data:" in content
