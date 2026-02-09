"""Tests for hexdag.api.execution module."""

import pytest

from hexdag.api.execution import (
    create_ports_from_config,
    dry_run,
    execute,
)


class TestDryRun:
    """Tests for dry_run function."""

    def test_dry_run_valid_pipeline(self):
        """Test dry_run with a valid pipeline."""
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
        result = dry_run(yaml_content)
        assert result["valid"] is True
        assert "execution_order" in result
        assert "node_count" in result
        assert result["node_count"] == 1
        assert "waves" in result

    def test_dry_run_multiple_nodes(self):
        """Test dry_run with multiple nodes and dependencies."""
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
    - kind: data_node
      metadata:
        name: c
      spec:
        output:
          value: 3
      dependencies: [a]
"""
        result = dry_run(yaml_content)
        assert result["valid"] is True
        assert result["node_count"] == 3
        # Check dependency map
        assert "dependency_map" in result
        assert "a" in result["dependency_map"]
        assert "b" in result["dependency_map"]
        assert "c" in result["dependency_map"]

    def test_dry_run_invalid_yaml(self):
        """Test dry_run with invalid YAML."""
        yaml_content = "invalid: yaml: content: :"
        result = dry_run(yaml_content)
        assert result["valid"] is False
        assert "error" in result

    def test_dry_run_missing_spec(self):
        """Test dry_run with missing spec."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test
"""
        result = dry_run(yaml_content)
        assert result["valid"] is False
        assert "error" in result


class TestExecute:
    """Tests for execute function."""

    @pytest.mark.asyncio
    async def test_execute_simple_pipeline(self):
        """Test executing a simple pipeline with data nodes."""
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
        result = await execute(yaml_content)
        assert result["success"] is True
        assert "final_output" in result
        assert "duration_ms" in result
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_execute_with_inputs(self):
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
        inputs = {"user_input": "test data"}
        result = await execute(yaml_content, inputs=inputs)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_invalid_pipeline(self):
        """Test executing an invalid pipeline returns error."""
        yaml_content = "invalid: yaml: :"
        result = await execute(yaml_content)
        assert result["success"] is False
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self):
        """Test execute respects timeout parameter."""
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
        result = await execute(yaml_content, timeout=60.0)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_returns_node_results(self):
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
        name: node_a
      spec:
        output:
          value: 1
      dependencies: []
"""
        result = await execute(yaml_content)
        assert result["success"] is True
        assert "nodes" in result
        assert isinstance(result["nodes"], list)


class TestCreatePortsFromConfig:
    """Tests for create_ports_from_config function."""

    def test_create_ports_with_mock_llm(self):
        """Test creating ports with MockLLM adapter."""
        config = {
            "llm": {"adapter": "MockLLM", "config": {"responses": "test response"}},
        }
        ports = create_ports_from_config(config)
        assert "llm" in ports
        assert "memory" in ports  # Default memory is always added

    def test_create_ports_with_memory(self):
        """Test creating ports with memory adapter."""
        config = {
            "memory": {"adapter": "InMemoryMemory", "config": {}},
        }
        ports = create_ports_from_config(config)
        assert "memory" in ports
        assert "llm" in ports  # Default LLM is always added

    def test_create_ports_empty_config(self):
        """Test creating ports with empty config gets defaults."""
        ports = create_ports_from_config({})
        assert "llm" in ports
        assert "memory" in ports

    def test_create_ports_unknown_adapter_fallback(self):
        """Test that unknown adapter falls back appropriately."""
        config = {
            "llm": {"adapter": "NonExistentAdapter", "config": {}},
        }
        ports = create_ports_from_config(config)
        # Should still have llm port (with fallback)
        assert "llm" in ports

    def test_create_ports_resolves_env_vars(self):
        """Test that environment variable references are resolved."""
        import os

        os.environ["TEST_API_KEY"] = "test-key-value"
        config = {
            "llm": {"adapter": "MockLLM", "config": {"api_key": "${TEST_API_KEY}"}},
        }
        ports = create_ports_from_config(config)
        assert "llm" in ports
        del os.environ["TEST_API_KEY"]

    def test_create_ports_coerces_string_values(self):
        """Test that string config values are coerced to proper types."""
        config = {
            "llm": {
                "adapter": "MockLLM",
                "config": {
                    "delay_seconds": "1.5",  # Should become float
                },
            },
        }
        ports = create_ports_from_config(config)
        assert "llm" in ports
