"""Tests for hexdag.cli.commands.pipeline_cmd module."""

import pytest
from typer.testing import CliRunner

from hexdag.cli.commands.pipeline_cmd import app


@pytest.fixture
def runner():
    """Fixture providing a Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def valid_pipeline_yaml(tmp_path):
    """Fixture providing a valid pipeline YAML file."""
    pipeline_path = tmp_path / "valid-pipeline.yaml"
    pipeline_content = """name: test_pipeline
description: A test pipeline

nodes:
  - id: input_node
    type: function
    params:
      fn: get_input
    depends_on: []

  - id: process_node
    type: llm
    params:
      prompt_template: "Process: {{input_node.output}}"
    depends_on: [input_node]

  - id: output_node
    type: function
    params:
      fn: format_output
    depends_on: [process_node]
"""
    pipeline_path.write_text(pipeline_content)
    return pipeline_path


@pytest.fixture
def pipeline_with_cycle(tmp_path):
    """Fixture providing a pipeline with a cycle."""
    pipeline_path = tmp_path / "cycle-pipeline.yaml"
    pipeline_content = """name: cycle_pipeline
description: Pipeline with cycle

nodes:
  - id: node_a
    type: function
    depends_on: [node_c]

  - id: node_b
    type: function
    depends_on: [node_a]

  - id: node_c
    type: function
    depends_on: [node_b]
"""
    pipeline_path.write_text(pipeline_content)
    return pipeline_path


@pytest.fixture
def parallel_pipeline_yaml(tmp_path):
    """Fixture providing a pipeline with parallel execution."""
    pipeline_path = tmp_path / "parallel-pipeline.yaml"
    pipeline_content = """name: parallel_pipeline
description: Pipeline with parallel nodes

nodes:
  - id: input
    type: function
    depends_on: []

  - id: process_a
    type: llm
    depends_on: [input]

  - id: process_b
    type: llm
    depends_on: [input]

  - id: process_c
    type: agent
    depends_on: [input]

  - id: combine
    type: function
    depends_on: [process_a, process_b, process_c]
"""
    pipeline_path.write_text(pipeline_content)
    return pipeline_path


@pytest.fixture
def invalid_yaml_pipeline(tmp_path):
    """Fixture providing an invalid YAML file."""
    pipeline_path = tmp_path / "invalid.yaml"
    pipeline_content = """name: broken
nodes:
  - id: test
    type: function
    depends_on: [
"""  # Unclosed bracket
    pipeline_path.write_text(pipeline_content)
    return pipeline_path


class TestValidateCommand:
    """Test the pipeline validate command."""

    def test_validate_valid_pipeline(self, runner, valid_pipeline_yaml):
        """Test validating a valid pipeline."""
        result = runner.invoke(app, ["validate", str(valid_pipeline_yaml)])
        assert result.exit_code == 0
        assert "Validating pipeline:" in result.stdout
        assert "✓ Pipeline validation passed" in result.stdout
        assert "Found 3 node(s)" in result.stdout

    def test_validate_nonexistent_file(self, runner, tmp_path):
        """Test validating a nonexistent file."""
        nonexistent = tmp_path / "nonexistent.yaml"
        result = runner.invoke(app, ["validate", str(nonexistent)])
        assert result.exit_code == 1
        assert "Pipeline file not found" in result.stdout

    def test_validate_invalid_yaml(self, runner, invalid_yaml_pipeline):
        """Test validating invalid YAML syntax."""
        result = runner.invoke(app, ["validate", str(invalid_yaml_pipeline)])
        assert result.exit_code == 1
        assert "YAML parsing error" in result.stdout

    def test_validate_pipeline_with_cycle(self, runner, pipeline_with_cycle):
        """Test validating a pipeline with a cycle."""
        result = runner.invoke(app, ["validate", str(pipeline_with_cycle)])
        assert result.exit_code == 1
        assert "Cycle detected" in result.stdout
        assert "node_a" in result.stdout
        assert "node_b" in result.stdout
        assert "node_c" in result.stdout

    def test_validate_missing_nodes_key(self, runner, tmp_path):
        """Test validating pipeline missing 'nodes' key."""
        pipeline_path = tmp_path / "no-nodes.yaml"
        pipeline_path.write_text("name: test\ndescription: missing nodes")
        result = runner.invoke(app, ["validate", str(pipeline_path)])
        assert result.exit_code == 1
        assert "'nodes' is required" in result.stdout

    def test_validate_duplicate_node_ids(self, runner, tmp_path):
        """Test validating pipeline with duplicate node IDs."""
        pipeline_path = tmp_path / "duplicate-ids.yaml"
        pipeline_content = """name: test
nodes:
  - id: duplicate
    type: function
  - id: duplicate
    type: function
"""
        pipeline_path.write_text(pipeline_content)
        result = runner.invoke(app, ["validate", str(pipeline_path)])
        assert result.exit_code == 1
        assert "Duplicate node ID" in result.stdout

    def test_validate_missing_dependency(self, runner, tmp_path):
        """Test validating pipeline with missing dependency reference."""
        pipeline_path = tmp_path / "missing-dep.yaml"
        pipeline_content = """name: test
nodes:
  - id: node_a
    type: function
    depends_on: [nonexistent_node]
"""
        pipeline_path.write_text(pipeline_content)
        result = runner.invoke(app, ["validate", str(pipeline_path)])
        assert result.exit_code == 1
        assert "dependency 'nonexistent_node' not found" in result.stdout


class TestPlanCommand:
    """Test the pipeline plan command."""

    def test_plan_linear_pipeline(self, runner, valid_pipeline_yaml):
        """Test execution plan for linear pipeline."""
        result = runner.invoke(app, ["plan", str(valid_pipeline_yaml)])
        assert result.exit_code == 0
        assert "Execution plan for: test_pipeline" in result.stdout
        assert "Wave 1:" in result.stdout
        assert "Wave 2:" in result.stdout
        assert "Wave 3:" in result.stdout
        assert "Total nodes: 3" in result.stdout
        assert "Max concurrency: 1" in result.stdout

    def test_plan_parallel_pipeline(self, runner, parallel_pipeline_yaml):
        """Test execution plan for pipeline with parallel nodes."""
        result = runner.invoke(app, ["plan", str(parallel_pipeline_yaml)])
        assert result.exit_code == 0
        assert "Execution plan for: parallel_pipeline" in result.stdout
        assert "Wave 2:" in result.stdout  # Should have parallel wave
        assert "Max concurrency: 3" in result.stdout  # 3 nodes in parallel
        assert "process_a" in result.stdout
        assert "process_b" in result.stdout
        assert "process_c" in result.stdout

    def test_plan_llm_count(self, runner, parallel_pipeline_yaml):
        """Test that plan correctly counts LLM calls."""
        result = runner.invoke(app, ["plan", str(parallel_pipeline_yaml)])
        assert result.exit_code == 0
        # Should count 2 LLM nodes + 1 agent node = 3 total
        assert "Expected LLM calls: 3" in result.stdout

    def test_plan_nonexistent_file(self, runner, tmp_path):
        """Test plan with nonexistent file."""
        nonexistent = tmp_path / "nonexistent.yaml"
        result = runner.invoke(app, ["plan", str(nonexistent)])
        assert result.exit_code == 1
        assert "Pipeline file not found" in result.stdout

    def test_plan_invalid_yaml(self, runner, invalid_yaml_pipeline):
        """Test plan with invalid YAML."""
        result = runner.invoke(app, ["plan", str(invalid_yaml_pipeline)])
        assert result.exit_code == 1
        # Error message varies, just check it failed with an error
        assert "Error" in result.stdout or "error" in result.stdout

    def test_plan_empty_pipeline(self, runner, tmp_path):
        """Test plan with empty nodes list."""
        pipeline_path = tmp_path / "empty.yaml"
        pipeline_path.write_text("name: empty\nnodes: []")
        result = runner.invoke(app, ["plan", str(pipeline_path)])
        assert result.exit_code == 0
        assert "Total nodes: 0" in result.stdout


class TestHelperFunctions:
    """Test helper functions used by commands."""

    def test_detect_cycles_no_cycle(self):
        """Test cycle detection with acyclic graph."""
        from hexdag.cli.commands.pipeline_cmd import _detect_cycles

        nodes = [
            {"id": "a", "depends_on": []},
            {"id": "b", "depends_on": ["a"]},
            {"id": "c", "depends_on": ["b"]},
        ]
        result = _detect_cycles(nodes)
        assert result is None

    def test_detect_cycles_with_cycle(self):
        """Test cycle detection with cyclic graph."""
        from hexdag.cli.commands.pipeline_cmd import _detect_cycles

        nodes = [
            {"id": "a", "depends_on": ["c"]},
            {"id": "b", "depends_on": ["a"]},
            {"id": "c", "depends_on": ["b"]},
        ]
        result = _detect_cycles(nodes)
        assert result is not None
        assert isinstance(result, list)
        assert "a" in result
        assert "b" in result
        assert "c" in result

    def test_detect_cycles_self_loop(self):
        """Test cycle detection with self-referencing node."""
        from hexdag.cli.commands.pipeline_cmd import _detect_cycles

        nodes = [{"id": "a", "depends_on": ["a"]}]
        result = _detect_cycles(nodes)
        assert result is not None
        assert "a" in result

    def test_calculate_waves_linear(self):
        """Test wave calculation for linear pipeline."""
        from hexdag.cli.commands.pipeline_cmd import _calculate_waves

        nodes = [
            {"id": "a", "depends_on": []},
            {"id": "b", "depends_on": ["a"]},
            {"id": "c", "depends_on": ["b"]},
        ]
        waves = _calculate_waves(nodes)
        assert len(waves) == 3
        assert waves[0] == ["a"]
        assert waves[1] == ["b"]
        assert waves[2] == ["c"]

    def test_calculate_waves_parallel(self):
        """Test wave calculation for parallel pipeline."""
        from hexdag.cli.commands.pipeline_cmd import _calculate_waves

        nodes = [
            {"id": "input", "depends_on": []},
            {"id": "p1", "depends_on": ["input"]},
            {"id": "p2", "depends_on": ["input"]},
            {"id": "p3", "depends_on": ["input"]},
            {"id": "combine", "depends_on": ["p1", "p2", "p3"]},
        ]
        waves = _calculate_waves(nodes)
        assert len(waves) == 3
        assert waves[0] == ["input"]
        assert set(waves[1]) == {"p1", "p2", "p3"}  # Parallel execution
        assert waves[2] == ["combine"]

    def test_calculate_waves_diamond(self):
        """Test wave calculation for diamond-shaped DAG."""
        from hexdag.cli.commands.pipeline_cmd import _calculate_waves

        nodes = [
            {"id": "start", "depends_on": []},
            {"id": "left", "depends_on": ["start"]},
            {"id": "right", "depends_on": ["start"]},
            {"id": "end", "depends_on": ["left", "right"]},
        ]
        waves = _calculate_waves(nodes)
        assert len(waves) == 3
        assert waves[0] == ["start"]
        assert set(waves[1]) == {"left", "right"}
        assert waves[2] == ["end"]
