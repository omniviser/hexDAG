"""Tests for the component instantiator module.

This module tests instantiation of adapters and policies from YAML specs.
"""

from __future__ import annotations

import pytest

from hexdag.core.pipeline_builder.component_instantiator import (
    ComponentInstantiationError,
    ComponentInstantiator,
    ComponentSpec,
)


class TestComponentSpec:
    """Tests for ComponentSpec namedtuple."""

    def test_component_spec_creation(self) -> None:
        """Test creating a ComponentSpec."""
        spec = ComponentSpec(
            module_path="hexdag.builtin.adapters.mock.MockLLM",
            params={"model": "gpt-4"},
        )
        assert spec.module_path == "hexdag.builtin.adapters.mock.MockLLM"
        assert spec.params == {"model": "gpt-4"}

    def test_component_spec_empty_params(self) -> None:
        """Test ComponentSpec with empty params."""
        spec = ComponentSpec(module_path="some.module.Class", params={})
        assert spec.params == {}


class TestComponentInstantiationError:
    """Tests for ComponentInstantiationError."""

    def test_error_message(self) -> None:
        """Test error message format."""
        error = ComponentInstantiationError("Failed to instantiate component")
        assert "Failed to instantiate component" in str(error)


class TestParseComponentSpec:
    """Tests for _parse_component_spec method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.instantiator = ComponentInstantiator()

    def test_parse_adapter_format(self) -> None:
        """Test parsing spec with 'adapter' key."""
        spec = {
            "adapter": "hexdag.builtin.adapters.mock.MockLLM",
            "config": {"model": "gpt-4"},
        }
        result = self.instantiator._parse_component_spec(spec)
        assert result.module_path == "hexdag.builtin.adapters.mock.MockLLM"
        assert result.params == {"model": "gpt-4"}

    def test_parse_name_format(self) -> None:
        """Test parsing spec with 'name' key."""
        spec = {
            "name": "hexdag.builtin.policies.execution_policies.RetryPolicy",
            "params": {"max_retries": 3},
        }
        result = self.instantiator._parse_component_spec(spec)
        assert result.module_path == "hexdag.builtin.policies.execution_policies.RetryPolicy"
        assert result.params == {"max_retries": 3}

    def test_parse_adapter_with_params_key(self) -> None:
        """Test parsing adapter spec with 'params' instead of 'config'."""
        spec = {
            "adapter": "hexdag.builtin.adapters.mock.MockLLM",
            "params": {"model": "gpt-4"},
        }
        result = self.instantiator._parse_component_spec(spec)
        assert result.params == {"model": "gpt-4"}

    def test_parse_with_no_params(self) -> None:
        """Test parsing spec without params."""
        spec = {"adapter": "hexdag.builtin.adapters.mock.MockLLM"}
        result = self.instantiator._parse_component_spec(spec)
        assert result.params == {}

    def test_parse_non_dict_raises_error(self) -> None:
        """Test that non-dict spec raises error."""
        with pytest.raises(ComponentInstantiationError) as exc_info:
            self.instantiator._parse_component_spec("not a dict")  # type: ignore[arg-type]
        assert "must be a dict" in str(exc_info.value)

    def test_parse_missing_module_path_raises_error(self) -> None:
        """Test that missing adapter/name raises error."""
        spec = {"config": {"model": "gpt-4"}}
        with pytest.raises(ComponentInstantiationError) as exc_info:
            self.instantiator._parse_component_spec(spec)
        assert "requires 'adapter' or 'name' field" in str(exc_info.value)


class TestInstantiateAdapter:
    """Tests for instantiate_adapter method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.instantiator = ComponentInstantiator()

    def test_instantiate_mock_llm_adapter(self) -> None:
        """Test instantiating MockLLM adapter."""
        spec = {"adapter": "hexdag.builtin.adapters.mock.MockLLM"}
        adapter = self.instantiator.instantiate_adapter(spec, port_name="llm")
        assert adapter is not None
        # Check it has the expected interface
        assert hasattr(adapter, "aresponse")

    def test_instantiate_mock_database_adapter(self) -> None:
        """Test instantiating MockDatabaseAdapter."""
        spec = {"adapter": "hexdag.builtin.adapters.mock.MockDatabaseAdapter"}
        adapter = self.instantiator.instantiate_adapter(spec, port_name="database")
        assert adapter is not None

    def test_instantiate_with_config_params(self) -> None:
        """Test instantiating adapter with configuration parameters."""
        spec = {
            "adapter": "hexdag.builtin.adapters.mock.MockLLM",
            "config": {"default_response": "test response"},
        }
        adapter = self.instantiator.instantiate_adapter(spec, port_name="llm")
        assert adapter is not None

    def test_instantiate_nonexistent_module_raises_error(self) -> None:
        """Test that nonexistent module raises error."""
        spec = {"adapter": "nonexistent.module.Adapter"}
        with pytest.raises(ComponentInstantiationError) as exc_info:
            self.instantiator.instantiate_adapter(spec, port_name="test")
        assert "could not be resolved" in str(exc_info.value)

    def test_instantiate_nonexistent_class_raises_error(self) -> None:
        """Test that nonexistent class raises error."""
        spec = {"adapter": "hexdag.builtin.adapters.mock.NonExistentAdapter"}
        with pytest.raises(ComponentInstantiationError) as exc_info:
            self.instantiator.instantiate_adapter(spec, port_name="test")
        assert "could not be resolved" in str(exc_info.value)

    def test_instantiate_with_extra_params_accepted(self) -> None:
        """Test that extra params are accepted (passed as kwargs)."""
        spec = {
            "adapter": "hexdag.builtin.adapters.mock.MockLLM",
            "config": {"extra_param": True},
        }
        # MockLLM accepts **kwargs, so extra params don't raise
        adapter = self.instantiator.instantiate_adapter(spec, port_name="llm")
        assert adapter is not None

    def test_instantiate_adapter_without_port_name(self) -> None:
        """Test instantiating adapter without port_name."""
        spec = {"adapter": "hexdag.builtin.adapters.mock.MockLLM"}
        adapter = self.instantiator.instantiate_adapter(spec)
        assert adapter is not None


class TestInstantiatePolicy:
    """Tests for instantiate_policy method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.instantiator = ComponentInstantiator()

    def test_instantiate_retry_policy(self) -> None:
        """Test instantiating RetryPolicy."""
        spec = {
            "name": "hexdag.builtin.policies.execution_policies.RetryPolicy",
            "params": {"max_retries": 5},
        }
        policy = self.instantiator.instantiate_policy(spec, policy_name="retry")
        assert policy is not None
        assert policy.max_retries == 5

    def test_instantiate_timeout_policy(self) -> None:
        """Test instantiating TimeoutPolicy."""
        spec = {
            "name": "hexdag.builtin.policies.execution_policies.TimeoutPolicy",
            "params": {"timeout_seconds": 60.0},
        }
        policy = self.instantiator.instantiate_policy(spec, policy_name="timeout")
        assert policy is not None
        assert policy.timeout_seconds == 60.0

    def test_instantiate_policy_with_defaults(self) -> None:
        """Test instantiating policy without custom params."""
        spec = {"name": "hexdag.builtin.policies.execution_policies.RetryPolicy"}
        policy = self.instantiator.instantiate_policy(spec, policy_name="retry")
        assert policy is not None

    def test_instantiate_nonexistent_policy_raises_error(self) -> None:
        """Test that nonexistent policy raises error."""
        spec = {"name": "nonexistent.module.Policy"}
        with pytest.raises(ComponentInstantiationError) as exc_info:
            self.instantiator.instantiate_policy(spec, policy_name="test")
        assert "could not be resolved" in str(exc_info.value)

    def test_instantiate_policy_with_extra_params_accepted(self) -> None:
        """Test that extra params are accepted (passed as kwargs)."""
        spec = {
            "name": "hexdag.builtin.policies.execution_policies.RetryPolicy",
            "params": {"extra_param": True},
        }
        # RetryPolicy accepts **kwargs, so extra params don't raise
        policy = self.instantiator.instantiate_policy(spec, policy_name="retry")
        assert policy is not None

    def test_instantiate_policy_without_policy_name(self) -> None:
        """Test instantiating policy without policy_name."""
        spec = {"name": "hexdag.builtin.policies.execution_policies.RetryPolicy"}
        policy = self.instantiator.instantiate_policy(spec)
        assert policy is not None


class TestInstantiatePorts:
    """Tests for instantiate_ports method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.instantiator = ComponentInstantiator()

    def test_instantiate_multiple_ports(self) -> None:
        """Test instantiating multiple ports."""
        ports_config = {
            "llm": {"adapter": "hexdag.builtin.adapters.mock.MockLLM"},
            "database": {"adapter": "hexdag.builtin.adapters.mock.MockDatabaseAdapter"},
        }
        ports = self.instantiator.instantiate_ports(ports_config)
        assert "llm" in ports
        assert "database" in ports
        assert ports["llm"] is not None
        assert ports["database"] is not None

    def test_instantiate_empty_ports(self) -> None:
        """Test instantiating empty ports config."""
        ports = self.instantiator.instantiate_ports({})
        assert ports == {}

    def test_instantiate_ports_with_one_failure_raises_error(self) -> None:
        """Test that one failed port raises error."""
        ports_config = {
            "llm": {"adapter": "hexdag.builtin.adapters.mock.MockLLM"},
            "bad": {"adapter": "nonexistent.Adapter"},
        }
        with pytest.raises(ComponentInstantiationError):
            self.instantiator.instantiate_ports(ports_config)


class TestInstantiatePolicies:
    """Tests for instantiate_policies method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.instantiator = ComponentInstantiator()

    def test_instantiate_multiple_policies(self) -> None:
        """Test instantiating multiple policies."""
        policies_config = {
            "retry": {
                "name": "hexdag.builtin.policies.execution_policies.RetryPolicy",
                "params": {"max_retries": 3},
            },
            "timeout": {
                "name": "hexdag.builtin.policies.execution_policies.TimeoutPolicy",
                "params": {"timeout_seconds": 60.0},
            },
        }
        policies = self.instantiator.instantiate_policies(policies_config)
        assert len(policies) == 2

    def test_instantiate_empty_policies(self) -> None:
        """Test instantiating empty policies config."""
        policies = self.instantiator.instantiate_policies({})
        assert policies == []

    def test_instantiate_policies_with_one_failure_raises_error(self) -> None:
        """Test that one failed policy raises error."""
        policies_config = {
            "good": {"name": "hexdag.builtin.policies.execution_policies.RetryPolicy"},
            "bad": {"name": "nonexistent.Policy"},
        }
        with pytest.raises(ComponentInstantiationError):
            self.instantiator.instantiate_policies(policies_config)


class TestComponentInstantiatorIntegration:
    """Integration tests for ComponentInstantiator."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.instantiator = ComponentInstantiator()

    def test_full_pipeline_config_instantiation(self) -> None:
        """Test instantiating a complete pipeline configuration."""
        ports_config = {
            "llm": {
                "adapter": "hexdag.builtin.adapters.mock.MockLLM",
                "config": {},
            },
            "memory": {
                "adapter": "hexdag.builtin.adapters.memory.InMemoryMemory",
                "config": {},
            },
        }
        policies_config = {
            "retry": {
                "name": "hexdag.builtin.policies.execution_policies.RetryPolicy",
                "params": {"max_retries": 3},
            },
        }

        ports = self.instantiator.instantiate_ports(ports_config)
        policies = self.instantiator.instantiate_policies(policies_config)

        assert len(ports) == 2
        assert len(policies) == 1
        assert "llm" in ports
        assert "memory" in ports

    def test_instantiate_all_mock_adapters(self) -> None:
        """Test that all mock adapters can be instantiated."""
        mock_adapters = [
            "hexdag.builtin.adapters.mock.MockLLM",
            "hexdag.builtin.adapters.mock.MockDatabaseAdapter",
            "hexdag.builtin.adapters.mock.MockEmbedding",
            "hexdag.builtin.adapters.mock.MockToolAdapter",
            "hexdag.builtin.adapters.mock.MockToolRouter",
        ]
        for adapter_path in mock_adapters:
            spec = {"adapter": adapter_path}
            adapter = self.instantiator.instantiate_adapter(spec)
            assert adapter is not None, f"Failed to instantiate {adapter_path}"
