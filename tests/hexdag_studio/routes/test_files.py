"""Tests for hexdag_studio.server.routes.files module."""

from pathlib import Path

from fastapi.testclient import TestClient


class TestListFiles:
    """Tests for GET /api/files endpoint."""

    def test_list_files_root(self, client: TestClient, temp_workspace: Path):
        """Test listing files in workspace root."""
        response = client.get("/api/files")
        assert response.status_code == 200
        data = response.json()
        assert "root" in data
        assert "files" in data
        assert isinstance(data["files"], list)

    def test_list_files_contains_yaml(self, client: TestClient, temp_workspace: Path):
        """Test that YAML files are listed."""
        response = client.get("/api/files")
        data = response.json()
        file_names = [f["name"] for f in data["files"]]
        assert "test_pipeline.yaml" in file_names

    def test_list_files_contains_directories(self, client: TestClient, temp_workspace: Path):
        """Test that directories are listed."""
        response = client.get("/api/files")
        data = response.json()
        file_names = [f["name"] for f in data["files"] if f["is_directory"]]
        assert "subdir" in file_names

    def test_list_files_subdirectory(self, client: TestClient, temp_workspace: Path):
        """Test listing files in a subdirectory."""
        response = client.get("/api/files?path=subdir")
        assert response.status_code == 200
        data = response.json()
        file_names = [f["name"] for f in data["files"]]
        assert "nested.yaml" in file_names

    def test_list_files_not_found(self, client: TestClient):
        """Test listing files in nonexistent directory."""
        response = client.get("/api/files?path=nonexistent")
        assert response.status_code == 404

    def test_list_files_file_path(self, client: TestClient, temp_workspace: Path):
        """Test listing files with file path (not directory)."""
        response = client.get("/api/files?path=test_pipeline.yaml")
        assert response.status_code == 400

    def test_list_files_excludes_hidden(self, client: TestClient, temp_workspace: Path):
        """Test that hidden files are excluded."""
        # Create a hidden file
        (temp_workspace / ".hidden.yaml").write_text("test: value")

        response = client.get("/api/files")
        data = response.json()
        file_names = [f["name"] for f in data["files"]]
        assert ".hidden.yaml" not in file_names


class TestReadFile:
    """Tests for GET /api/files/{path} endpoint."""

    def test_read_file_success(self, client: TestClient, temp_workspace: Path):
        """Test reading a YAML file."""
        response = client.get("/api/files/test_pipeline.yaml")
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert "path" in data
        assert "modified" in data
        assert "apiVersion" in data["content"]

    def test_read_file_nested(self, client: TestClient, temp_workspace: Path):
        """Test reading a file in a subdirectory."""
        response = client.get("/api/files/subdir/nested.yaml")
        assert response.status_code == 200
        data = response.json()
        assert "content" in data

    def test_read_file_not_found(self, client: TestClient):
        """Test reading a nonexistent file."""
        response = client.get("/api/files/nonexistent.yaml")
        assert response.status_code == 404

    def test_read_file_directory(self, client: TestClient, temp_workspace: Path):
        """Test reading a directory (should fail)."""
        response = client.get("/api/files/subdir")
        assert response.status_code == 400

    def test_read_file_non_yaml(self, client: TestClient, temp_workspace: Path):
        """Test reading a non-YAML file (should fail)."""
        # Create a non-YAML file
        (temp_workspace / "test.txt").write_text("hello")

        response = client.get("/api/files/test.txt")
        assert response.status_code == 400

    def test_read_file_traversal_attack(self, client: TestClient, temp_workspace: Path):
        """Test that path traversal outside workspace is handled.

        Note: Paths with traversal outside the workspace should either:
        - Be blocked by the path security check (403)
        - Fall through to SPA handler (200 with HTML, not file content)
        - Return not found (404)

        The key test is that /etc/passwd content is NOT returned.
        """
        response = client.get("/api/files/../../../etc/passwd")
        # Either error status or SPA fallback (200 with HTML)
        if response.status_code == 200:
            # If 200, ensure it's the SPA HTML, not the actual file content
            content_type = response.headers.get("content-type", "")
            # SPA returns HTML, not YAML file content
            assert "html" in content_type.lower() or "root:" not in response.text
        else:
            # Error statuses are acceptable
            assert response.status_code in (400, 403, 404)


class TestSaveFile:
    """Tests for PUT /api/files/{path} endpoint."""

    def test_save_file_success(self, client: TestClient, temp_workspace: Path):
        """Test saving a file."""
        new_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: updated-pipeline
spec:
  nodes: []
"""
        response = client.put(
            "/api/files/test_pipeline.yaml",
            json={"content": new_content},
        )
        assert response.status_code == 200
        data = response.json()
        assert "updated-pipeline" in data["content"]

    def test_save_file_creates_new(self, client: TestClient, temp_workspace: Path):
        """Test creating a new file."""
        new_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: new-pipeline
spec:
  nodes: []
"""
        response = client.put(
            "/api/files/new_pipeline.yaml",
            json={"content": new_content},
        )
        assert response.status_code == 200
        assert (temp_workspace / "new_pipeline.yaml").exists()

    def test_save_file_creates_directories(self, client: TestClient, temp_workspace: Path):
        """Test that parent directories are created."""
        new_content = "test: value"
        response = client.put(
            "/api/files/new_dir/nested/file.yaml",
            json={"content": new_content},
        )
        assert response.status_code == 200
        assert (temp_workspace / "new_dir" / "nested" / "file.yaml").exists()

    def test_save_file_non_yaml(self, client: TestClient, temp_workspace: Path):
        """Test saving a non-YAML file (should fail)."""
        response = client.put(
            "/api/files/test.txt",
            json={"content": "hello"},
        )
        assert response.status_code == 400

    def test_save_file_traversal_attack(self, client: TestClient, temp_workspace: Path):
        """Test that path traversal outside workspace is handled.

        Note: The actual behavior depends on path resolution and routing.
        The important thing is that it does not succeed (200).
        """
        response = client.put(
            "/api/files/../../../tmp/test.yaml",
            json={"content": "test: value"},
        )
        # Should not succeed - either forbidden (403), bad request (400),
        # method not allowed (405 if route doesn't match), or not found (404)
        assert response.status_code in (400, 403, 404, 405)


class TestDeleteFile:
    """Tests for DELETE /api/files/{path} endpoint."""

    def test_delete_file_success(self, client: TestClient, temp_workspace: Path):
        """Test deleting a file."""
        # Verify file exists
        assert (temp_workspace / "test_pipeline.yaml").exists()

        response = client.delete("/api/files/test_pipeline.yaml")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"

        # Verify file is gone
        assert not (temp_workspace / "test_pipeline.yaml").exists()

    def test_delete_file_not_found(self, client: TestClient):
        """Test deleting a nonexistent file."""
        response = client.delete("/api/files/nonexistent.yaml")
        assert response.status_code == 404

    def test_delete_file_directory(self, client: TestClient, temp_workspace: Path):
        """Test deleting a directory (should fail)."""
        response = client.delete("/api/files/subdir")
        assert response.status_code == 400

    def test_delete_file_non_yaml(self, client: TestClient, temp_workspace: Path):
        """Test deleting a non-YAML file (should fail)."""
        # Create a non-YAML file
        (temp_workspace / "test.txt").write_text("hello")

        response = client.delete("/api/files/test.txt")
        assert response.status_code == 400


class TestFileInfo:
    """Tests for file metadata in responses."""

    def test_file_info_has_size(self, client: TestClient, temp_workspace: Path):
        """Test that file info includes size."""
        response = client.get("/api/files")
        data = response.json()
        yaml_file = next(f for f in data["files"] if f["name"] == "test_pipeline.yaml")
        assert yaml_file["size"] is not None
        assert yaml_file["size"] > 0

    def test_file_info_has_modified(self, client: TestClient, temp_workspace: Path):
        """Test that file info includes modified timestamp."""
        response = client.get("/api/files")
        data = response.json()
        yaml_file = next(f for f in data["files"] if f["name"] == "test_pipeline.yaml")
        assert yaml_file["modified"] is not None

    def test_directory_has_no_size(self, client: TestClient, temp_workspace: Path):
        """Test that directories have no size."""
        response = client.get("/api/files")
        data = response.json()
        directory = next(f for f in data["files"] if f["is_directory"])
        assert directory["size"] is None
