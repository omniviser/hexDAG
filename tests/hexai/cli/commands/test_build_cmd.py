"""Tests for build command."""

import pytest
import yaml
from typer.testing import CliRunner

from hexai.cli.commands.build_cmd import app

runner = CliRunner()


@pytest.fixture
def sample_pipeline_yaml():
    """Sample pipeline YAML content."""
    return """
apiVersion: v1
kind: Pipeline
metadata:
  name: test-pipeline
  description: Test pipeline
spec:
  nodes:
    - kind: function_node
      metadata:
        name: step1
      spec:
        fn: my_function
        input_mapping:
          x: input.value
        dependencies: []
"""


@pytest.fixture
def temp_pipeline_file(tmp_path, sample_pipeline_yaml):
    """Create temporary pipeline file."""
    pipeline_file = tmp_path / "test-pipeline.yaml"
    pipeline_file.write_text(sample_pipeline_yaml)
    return pipeline_file


def test_build_single_pipeline(temp_pipeline_file, tmp_path):
    """Test building single pipeline."""
    output_dir = tmp_path / "build"

    result = runner.invoke(
        app,
        [str(temp_pipeline_file), "--output", str(output_dir)],
    )

    assert result.exit_code == 0
    assert "Docker build files generated successfully" in result.stdout

    # Check generated files
    assert (output_dir / "Dockerfile").exists()
    assert (output_dir / "docker-entrypoint.sh").exists()
    assert (output_dir / "README.md").exists()
    assert (output_dir / "requirements.txt").exists()
    assert (output_dir / ".dockerignore").exists()
    assert (output_dir / "pipelines").is_dir()
    assert (output_dir / "pipelines" / "test-pipeline.yaml").exists()
    assert (output_dir / "src").is_dir()


def test_build_multiple_pipelines(tmp_path, sample_pipeline_yaml):
    """Test building multiple pipelines."""
    # Create two pipeline files
    pipeline1 = tmp_path / "pipeline1.yaml"
    pipeline2 = tmp_path / "pipeline2.yaml"
    pipeline1.write_text(sample_pipeline_yaml)
    pipeline2.write_text(sample_pipeline_yaml.replace("test-pipeline", "pipeline2"))

    output_dir = tmp_path / "build"

    result = runner.invoke(
        app,
        [
            str(pipeline1),
            str(pipeline2),
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Docker build files generated successfully" in result.stdout

    # Check docker-compose is generated for multiple pipelines
    assert (output_dir / "docker-compose.yml").exists()

    # Verify docker-compose content
    with (output_dir / "docker-compose.yml").open() as f:
        compose_data = yaml.safe_load(f)
        assert "services" in compose_data
        assert "pipeline1" in compose_data["services"]
        assert "pipeline2" in compose_data["services"]


def test_build_custom_image_name(temp_pipeline_file, tmp_path):
    """Test building with custom image name."""
    output_dir = tmp_path / "build"

    result = runner.invoke(
        app,
        [
            str(temp_pipeline_file),
            "--output",
            str(output_dir),
            "--image",
            "my-custom-image:v1",
        ],
    )

    assert result.exit_code == 0

    # Check README contains custom image name
    with (output_dir / "README.md").open() as f:
        readme_content = f.read()
        assert "my-custom-image:v1" in readme_content


def test_build_custom_python_version(temp_pipeline_file, tmp_path):
    """Test building with custom Python version."""
    output_dir = tmp_path / "build"

    result = runner.invoke(
        app,
        [
            str(temp_pipeline_file),
            "--output",
            str(output_dir),
            "--python-version",
            "3.11",
        ],
    )

    assert result.exit_code == 0

    # Check Dockerfile contains custom Python version
    with (output_dir / "Dockerfile").open() as f:
        dockerfile_content = f.read()
        assert "python:3.11-slim" in dockerfile_content


def test_build_custom_base_image(temp_pipeline_file, tmp_path):
    """Test building with custom base image."""
    output_dir = tmp_path / "build"

    result = runner.invoke(
        app,
        [
            str(temp_pipeline_file),
            "--output",
            str(output_dir),
            "--base-image",
            "ubuntu:22.04",
        ],
    )

    assert result.exit_code == 0

    # Check Dockerfile contains custom base image
    with (output_dir / "Dockerfile").open() as f:
        dockerfile_content = f.read()
        assert "FROM ubuntu:22.04" in dockerfile_content


def test_build_no_compose(tmp_path, sample_pipeline_yaml):
    """Test building without docker-compose."""
    # Create two pipeline files
    pipeline1 = tmp_path / "pipeline1.yaml"
    pipeline2 = tmp_path / "pipeline2.yaml"
    pipeline1.write_text(sample_pipeline_yaml)
    pipeline2.write_text(sample_pipeline_yaml.replace("test-pipeline", "pipeline2"))

    output_dir = tmp_path / "build"

    result = runner.invoke(
        app,
        [
            str(pipeline1),
            str(pipeline2),
            "--output",
            str(output_dir),
            "--no-compose",
        ],
    )

    assert result.exit_code == 0

    # docker-compose should not be generated
    assert not (output_dir / "docker-compose.yml").exists()


def test_build_nonexistent_pipeline():
    """Test building with non-existent pipeline file."""
    result = runner.invoke(
        app,
        ["/nonexistent/pipeline.yaml"],
    )

    assert result.exit_code != 0


def test_build_invalid_yaml(tmp_path):
    """Test building with invalid YAML file."""
    invalid_file = tmp_path / "invalid.yaml"
    invalid_file.write_text("invalid: yaml: content: [")

    result = runner.invoke(
        app,
        [str(invalid_file)],
    )

    assert result.exit_code == 1
    assert "Failed to parse YAML" in result.stdout


def test_dockerfile_structure(temp_pipeline_file, tmp_path):
    """Test generated Dockerfile structure."""
    output_dir = tmp_path / "build"

    runner.invoke(
        app,
        [str(temp_pipeline_file), "--output", str(output_dir)],
    )

    with (output_dir / "Dockerfile").open() as f:
        content = f.read()

        # Check key elements
        assert "FROM python:" in content
        # Default extras are yaml,openai,anthropic,cli
        assert "pip install --no-cache-dir hexdag[yaml,openai,anthropic,cli]" in content
        assert "COPY pipelines/test-pipeline.yaml /app/pipelines/test-pipeline.yaml" in content
        assert "COPY docker-entrypoint.sh /usr/local/bin/" in content
        assert 'ENTRYPOINT ["docker-entrypoint.sh"]' in content
        assert "COPY requirements.txt /app/requirements.txt" in content
        assert "COPY src/ /app/src/" in content


def test_entrypoint_script_structure(temp_pipeline_file, tmp_path):
    """Test generated entrypoint script structure."""
    output_dir = tmp_path / "build"

    runner.invoke(
        app,
        [str(temp_pipeline_file), "--output", str(output_dir)],
    )

    with (output_dir / "docker-entrypoint.sh").open() as f:
        content = f.read()

        # Check key elements
        assert "#!/bin/bash" in content
        assert "test-pipeline" in content
        assert "YamlPipelineBuilder" in content
        assert "bootstrap_registry()" in content


def test_readme_generation(temp_pipeline_file, tmp_path):
    """Test README generation."""
    output_dir = tmp_path / "build"

    runner.invoke(
        app,
        [str(temp_pipeline_file), "--output", str(output_dir)],
    )

    with (output_dir / "README.md").open() as f:
        content = f.read()

        # Check key sections
        assert "# HexDAG Pipeline Container" in content
        assert "test-pipeline" in content
        assert "## Building the Image" in content
        assert "## Running Pipelines" in content
        assert "docker build" in content
        assert "docker run" in content


def test_dockerignore_generation(temp_pipeline_file, tmp_path):
    """Test .dockerignore generation."""
    output_dir = tmp_path / "build"

    runner.invoke(
        app,
        [str(temp_pipeline_file), "--output", str(output_dir)],
    )

    with (output_dir / ".dockerignore").open() as f:
        content = f.read()

        # Check common ignore patterns
        assert "__pycache__" in content
        assert "*.pyc" in content
        assert ".git" in content
        assert ".venv" in content


def test_custom_extras(temp_pipeline_file, tmp_path):
    """Test building with custom extras."""
    output_dir = tmp_path / "build"

    result = runner.invoke(
        app,
        [str(temp_pipeline_file), "--output", str(output_dir), "--extras", "yaml,viz"],
    )

    assert result.exit_code == 0

    with (output_dir / "Dockerfile").open() as f:
        content = f.read()
        assert "pip install --no-cache-dir hexdag[yaml,viz]" in content
        assert "Extras: yaml,viz" in content


def test_no_extras(temp_pipeline_file, tmp_path):
    """Test building with no extras (base install only)."""
    output_dir = tmp_path / "build"

    result = runner.invoke(
        app,
        [str(temp_pipeline_file), "--output", str(output_dir), "--extras", ""],
    )

    assert result.exit_code == 0

    with (output_dir / "Dockerfile").open() as f:
        content = f.read()
        assert "pip install --no-cache-dir hexdag\n" in content
        assert "Extras: none (base install only)" in content


def test_invalid_pipeline_structure_no_name(tmp_path):
    """Test validation fails for pipeline without name or metadata."""
    invalid_pipeline = tmp_path / "invalid.yaml"
    invalid_pipeline.write_text("""
spec:
  nodes:
    - kind: function_node
      metadata:
        name: step1
      spec:
        fn: my_function
""")

    result = runner.invoke(
        app,
        [str(invalid_pipeline)],
    )

    assert result.exit_code == 1
    assert "Invalid pipeline structure" in result.stdout
    # Handle line breaks in error message
    assert "name" in result.stdout and "metadata" in result.stdout


def test_invalid_pipeline_structure_no_nodes(tmp_path):
    """Test validation fails for pipeline without nodes."""
    invalid_pipeline = tmp_path / "invalid.yaml"
    invalid_pipeline.write_text("""
name: test-pipeline
description: Test pipeline
""")

    result = runner.invoke(
        app,
        [str(invalid_pipeline)],
    )

    assert result.exit_code == 1
    assert "Invalid pipeline structure" in result.stdout
    # Handle line breaks in error message
    assert "nodes" in result.stdout and "spec" in result.stdout


def test_invalid_pipeline_structure_empty_nodes(tmp_path):
    """Test validation fails for pipeline with empty nodes list."""
    invalid_pipeline = tmp_path / "invalid.yaml"
    invalid_pipeline.write_text("""
name: test-pipeline
nodes: []
""")

    result = runner.invoke(
        app,
        [str(invalid_pipeline)],
    )

    assert result.exit_code == 1
    assert "Invalid pipeline structure" in result.stdout
    # Handle line breaks in error message
    assert "at least" in result.stdout and "one node" in result.stdout


def test_invalid_pipeline_structure_node_no_type(tmp_path):
    """Test validation fails for node without type/kind."""
    invalid_pipeline = tmp_path / "invalid.yaml"
    invalid_pipeline.write_text("""
name: test-pipeline
nodes:
  - id: step1
    params:
      value: test
""")

    result = runner.invoke(
        app,
        [str(invalid_pipeline)],
    )

    assert result.exit_code == 1
    assert "Invalid pipeline structure" in result.stdout
    # Handle line breaks in error message
    assert "'type' or" in result.stdout and "'kind'" in result.stdout


def test_invalid_pipeline_structure_node_no_id(tmp_path):
    """Test validation fails for node without id/name."""
    invalid_pipeline = tmp_path / "invalid.yaml"
    invalid_pipeline.write_text("""
name: test-pipeline
nodes:
  - type: function
    params:
      value: test
""")

    result = runner.invoke(
        app,
        [str(invalid_pipeline)],
    )

    assert result.exit_code == 1
    assert "Invalid pipeline structure" in result.stdout
    # Handle line breaks in error message
    assert (
        "'id'" in result.stdout and "'name'" in result.stdout and "metadata.name" in result.stdout
    )


def test_old_format_pipeline_validation(tmp_path):
    """Test validation works for old format pipelines."""
    old_format_pipeline = tmp_path / "old-format.yaml"
    old_format_pipeline.write_text("""
name: old-pipeline
description: Old format pipeline
nodes:
  - type: function
    id: step1
    params:
      value: test
    depends_on: []
""")

    output_dir = tmp_path / "build"
    result = runner.invoke(
        app,
        [str(old_format_pipeline), "--output", str(output_dir)],
    )

    assert result.exit_code == 0
    assert "Valid: old-format.yaml" in result.stdout


def test_env_file_generation(temp_pipeline_file, tmp_path):
    """Test .env.example file is generated."""
    output_dir = tmp_path / "build"

    result = runner.invoke(
        app,
        [str(temp_pipeline_file), "--output", str(output_dir)],
    )

    assert result.exit_code == 0
    assert (output_dir / ".env.example").exists()

    with (output_dir / ".env.example").open() as f:
        content = f.read()
        assert "OPENAI_API_KEY" in content
        assert "ANTHROPIC_API_KEY" in content
        assert "DATABASE_URL" in content


def test_entrypoint_file_input_support(temp_pipeline_file, tmp_path):
    """Test entrypoint script includes file input support."""
    output_dir = tmp_path / "build"

    runner.invoke(
        app,
        [str(temp_pipeline_file), "--output", str(output_dir)],
    )

    with (output_dir / "docker-entrypoint.sh").open() as f:
        content = f.read()
        # Check for file input handling
        assert 'if input_str.startswith("@"):' in content
        assert "input_file = input_str[1:]" in content
        assert "FileNotFoundError" in content
        assert "Invalid JSON" in content


def test_entrypoint_error_handling(temp_pipeline_file, tmp_path):
    """Test entrypoint script includes comprehensive error handling."""
    output_dir = tmp_path / "build"

    runner.invoke(
        app,
        [str(temp_pipeline_file), "--output", str(output_dir)],
    )

    with (output_dir / "docker-entrypoint.sh").open() as f:
        content = f.read()
        # Check for error handling
        assert "try:" in content
        assert "except Exception as e:" in content
        assert "traceback.print_exc()" in content
        assert "sys.exit(1)" in content
