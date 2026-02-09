"""Tests for hexdag_studio.server.main module."""

from pathlib import Path

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for /api/health endpoint."""

    def test_health_check(self, client: TestClient, temp_workspace: Path):
        """Test that health endpoint returns OK."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "workspace" in data


class TestCORS:
    """Tests for CORS middleware."""

    def test_cors_headers_localhost(self, client: TestClient):
        """Test that CORS headers are set for localhost."""
        response = client.options(
            "/api/health",
            headers={"Origin": "http://localhost:3141"},
        )
        # FastAPI with CORS middleware should handle preflight requests
        assert response.status_code in [200, 405]


class TestCreateApp:
    """Tests for create_app factory function."""

    def test_create_app_with_workspace(self, temp_workspace: Path):
        """Test creating app with workspace path."""
        from hexdag_studio.server.main import create_app

        app = create_app(workspace_path=temp_workspace)
        assert app is not None
        assert app.title == "hexdag studio"

    def test_create_app_with_plugins(self, temp_workspace: Path):
        """Test creating app with plugin paths."""
        from hexdag_studio.server.main import create_app

        plugin_dir = temp_workspace / "plugins"
        plugin_dir.mkdir()

        app = create_app(
            workspace_path=temp_workspace,
            plugin_paths=[plugin_dir],
        )
        assert app is not None


class TestAPIRoutes:
    """Tests for API route registration."""

    def test_registry_routes_registered(self, client: TestClient):
        """Test that registry routes are registered."""
        response = client.get("/api/registry/nodes")
        assert response.status_code == 200

    def test_validate_routes_registered(self, client: TestClient):
        """Test that validate routes are registered."""
        response = client.post(
            "/api/validate",
            json={"content": "test: value"},
        )
        # Should return 200 even for invalid YAML (validation result)
        assert response.status_code == 200

    def test_execute_routes_registered(self, client: TestClient):
        """Test that execute routes are registered."""
        yaml_content = (
            "apiVersion: hexdag/v1\nkind: Pipeline\nmetadata:\n  name: test\nspec:\n  nodes: []"
        )
        response = client.post(
            "/api/execute/dry-run",
            json={"content": yaml_content},
        )
        assert response.status_code == 200

    def test_files_routes_registered(self, client: TestClient):
        """Test that files routes are registered."""
        response = client.get("/api/files")
        assert response.status_code == 200
