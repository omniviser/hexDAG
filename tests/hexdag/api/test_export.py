"""Tests for hexdag.api.export module."""

from hexdag.api.export import export_project


class TestExportProject:
    """Tests for export_project function."""

    def test_export_project_success(self):
        """Test successful project export."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: my-pipeline
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
        result = export_project(yaml_content)
        assert result["success"] is True
        assert result["project_name"] == "my_pipeline"
        assert "files" in result
        assert len(result["files"]) > 0

    def test_export_project_creates_pyproject_toml(self):
        """Test that pyproject.toml is created."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-project
spec:
  nodes: []
"""
        result = export_project(yaml_content)
        assert result["success"] is True
        file_paths = [f["path"] for f in result["files"]]
        assert "pyproject.toml" in file_paths

    def test_export_project_creates_readme(self):
        """Test that README.md is created."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-project
spec:
  nodes: []
"""
        result = export_project(yaml_content)
        file_paths = [f["path"] for f in result["files"]]
        assert "README.md" in file_paths

    def test_export_project_creates_env_example(self):
        """Test that .env.example is created."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-project
spec:
  nodes: []
"""
        result = export_project(yaml_content)
        file_paths = [f["path"] for f in result["files"]]
        assert ".env.example" in file_paths

    def test_export_project_creates_gitignore(self):
        """Test that .gitignore is created."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-project
spec:
  nodes: []
"""
        result = export_project(yaml_content)
        file_paths = [f["path"] for f in result["files"]]
        assert ".gitignore" in file_paths

    def test_export_project_creates_main_py(self):
        """Test that main.py is created."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-project
spec:
  nodes: []
"""
        result = export_project(yaml_content)
        file_paths = [f["path"] for f in result["files"]]
        assert any("main.py" in p for p in file_paths)

    def test_export_project_creates_pipeline_yaml(self):
        """Test that pipeline.yaml is created with original content."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-project
spec:
  nodes: []
"""
        result = export_project(yaml_content)
        pipeline_file = next(
            (f for f in result["files"] if f["path"].endswith("pipeline.yaml")), None
        )
        assert pipeline_file is not None
        assert yaml_content.strip() in pipeline_file["content"]

    def test_export_project_custom_name(self):
        """Test export with custom project name."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: original-name
spec:
  nodes: []
"""
        result = export_project(yaml_content, project_name="custom_name")
        assert result["project_name"] == "custom_name"

    def test_export_project_slugify_name(self):
        """Test that project name is slugified."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: My Project Name!
spec:
  nodes: []
"""
        result = export_project(yaml_content)
        # Should be converted to valid Python package name
        assert result["project_name"] == "my_project_name"

    def test_export_project_with_docker(self):
        """Test export with Dockerfile included."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-project
spec:
  nodes: []
"""
        result = export_project(yaml_content, include_docker=True)
        file_paths = [f["path"] for f in result["files"]]
        assert "Dockerfile" in file_paths

    def test_export_project_without_docker(self):
        """Test export without Dockerfile by default."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-project
spec:
  nodes: []
"""
        result = export_project(yaml_content, include_docker=False)
        file_paths = [f["path"] for f in result["files"]]
        assert "Dockerfile" not in file_paths

    def test_export_project_python_version(self):
        """Test export with custom Python version."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-project
spec:
  nodes: []
"""
        result = export_project(yaml_content, python_version="3.11")
        pyproject = next(f for f in result["files"] if f["path"] == "pyproject.toml")
        assert "3.11" in pyproject["content"]

    def test_export_project_extracts_env_vars(self):
        """Test that environment variables are extracted to .env.example."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-project
spec:
  ports:
    llm:
      adapter: openai
      config:
        api_key: ${OPENAI_API_KEY}
        org_id: ${OPENAI_ORG_ID}
  nodes: []
"""
        result = export_project(yaml_content)
        env_file = next(f for f in result["files"] if f["path"] == ".env.example")
        assert "OPENAI_API_KEY" in env_file["content"]
        assert "OPENAI_ORG_ID" in env_file["content"]

    def test_export_project_invalid_yaml(self):
        """Test export with invalid YAML returns error."""
        yaml_content = "invalid: yaml: :"
        result = export_project(yaml_content)
        assert result["success"] is False
        assert "error" in result

    def test_export_project_empty_yaml(self):
        """Test export with empty/null YAML returns error."""
        result = export_project("")
        assert result["success"] is False

    def test_export_project_detects_adapters(self):
        """Test that adapters are detected and dependencies added."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-project
spec:
  ports:
    llm:
      adapter: openai
  nodes:
    - kind: llm_node
      metadata:
        name: analyzer
      spec:
        prompt_template: "Test"
      dependencies: []
"""
        result = export_project(yaml_content)
        pyproject = next(f for f in result["files"] if f["path"] == "pyproject.toml")
        # Should include openai dependency
        assert "openai" in pyproject["content"]


class TestPyprojectGeneration:
    """Tests for pyproject.toml generation."""

    def test_pyproject_has_required_fields(self):
        """Test that pyproject.toml has all required fields."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-project
spec:
  nodes: []
"""
        result = export_project(yaml_content)
        pyproject = next(f for f in result["files"] if f["path"] == "pyproject.toml")
        content = pyproject["content"]
        assert "[project]" in content
        assert "name =" in content
        assert "version =" in content
        assert "dependencies =" in content
        assert "[build-system]" in content


class TestReadmeGeneration:
    """Tests for README.md generation."""

    def test_readme_includes_pipeline_name(self):
        """Test that README includes pipeline name."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: awesome-pipeline
spec:
  nodes: []
"""
        result = export_project(yaml_content)
        readme = next(f for f in result["files"] if f["path"] == "README.md")
        assert "awesome-pipeline" in readme["content"]

    def test_readme_includes_node_list(self):
        """Test that README lists nodes."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test
spec:
  nodes:
    - kind: llm_node
      metadata:
        name: analyzer
      spec:
        prompt_template: "Test"
      dependencies: []
"""
        result = export_project(yaml_content)
        readme = next(f for f in result["files"] if f["path"] == "README.md")
        assert "analyzer" in readme["content"]


class TestDockerfileGeneration:
    """Tests for Dockerfile generation."""

    def test_dockerfile_uses_python_version(self):
        """Test that Dockerfile uses specified Python version."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-project
spec:
  nodes: []
"""
        result = export_project(yaml_content, include_docker=True, python_version="3.11")
        dockerfile = next(f for f in result["files"] if f["path"] == "Dockerfile")
        assert "python:3.11" in dockerfile["content"]

    def test_dockerfile_has_workdir(self):
        """Test that Dockerfile sets WORKDIR."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-project
spec:
  nodes: []
"""
        result = export_project(yaml_content, include_docker=True)
        dockerfile = next(f for f in result["files"] if f["path"] == "Dockerfile")
        assert "WORKDIR" in dockerfile["content"]
