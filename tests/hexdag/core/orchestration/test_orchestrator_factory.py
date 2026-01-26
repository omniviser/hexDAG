"""Tests for orchestrator_factory module.

This module tests the orchestrator factory that creates orchestrator instances
from pipeline configuration.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hexdag.core.orchestration.orchestrator_factory import OrchestratorFactory
from hexdag.core.pipeline_builder.pipeline_config import PipelineConfig


class TestOrchestratorFactory:
    """Tests for OrchestratorFactory."""

    def test_initialization(self) -> None:
        """Test factory initialization."""
        factory = OrchestratorFactory()
        assert factory.component_instantiator is not None

    def test_create_orchestrator_empty_config(self) -> None:
        """Test creating orchestrator with empty config."""
        factory = OrchestratorFactory()
        config = PipelineConfig(
            ports={},
            type_ports={},
            policies={},
            metadata={"name": "test"},
            nodes=[],
        )
        orchestrator = factory.create_orchestrator(config)
        assert orchestrator is not None

    def test_create_orchestrator_with_mock_llm(self) -> None:
        """Test creating orchestrator with mock LLM port."""
        factory = OrchestratorFactory()
        config = PipelineConfig(
            ports={
                "llm": {
                    "adapter": "hexdag.builtin.adapters.mock.MockLLM",
                }
            },
            type_ports={},
            policies={},
            metadata={"name": "test"},
            nodes=[],
        )
        orchestrator = factory.create_orchestrator(config)
        assert orchestrator is not None

    def test_create_orchestrator_with_policy(self) -> None:
        """Test creating orchestrator with policy."""
        factory = OrchestratorFactory()
        config = PipelineConfig(
            ports={},
            type_ports={},
            policies={
                "retry": {
                    "name": "hexdag.builtin.policies.execution_policies.RetryPolicy",
                    "params": {"max_retries": 3},
                }
            },
            metadata={"name": "test"},
            nodes=[],
        )
        orchestrator = factory.create_orchestrator(config)
        assert orchestrator is not None

    def test_create_orchestrator_with_custom_params(self) -> None:
        """Test creating orchestrator with custom parameters."""
        factory = OrchestratorFactory()
        config = PipelineConfig(
            ports={},
            type_ports={},
            policies={},
            metadata={"name": "test"},
            nodes=[],
        )
        orchestrator = factory.create_orchestrator(
            config,
            max_concurrent_nodes=5,
            strict_validation=True,
            default_node_timeout=30.0,
        )
        assert orchestrator is not None

    def test_create_orchestrator_with_additional_ports(self) -> None:
        """Test creating orchestrator with additional ports."""
        factory = OrchestratorFactory()
        config = PipelineConfig(
            ports={},
            type_ports={},
            policies={},
            metadata={"name": "test"},
            nodes=[],
        )

        # Create mock additional port
        mock_port = MagicMock()

        orchestrator = factory.create_orchestrator(config, additional_ports={"custom": mock_port})
        assert orchestrator is not None

    @pytest.mark.skip(reason="Type ports support is incomplete - known limitation")
    def test_create_orchestrator_with_type_ports(self) -> None:
        """Test creating orchestrator with type-specific ports.

        Note: This test is skipped because type_ports is a known Phase 4 limitation
        (see OrchestratorFactory.create_orchestrator docstring).
        """
        factory = OrchestratorFactory()
        config = PipelineConfig(
            ports={
                "llm": {"adapter": "hexdag.builtin.adapters.mock.MockLLM"},
            },
            type_ports={
                "agent": {
                    "llm": {"adapter": "hexdag.builtin.adapters.mock.MockLLM"},
                }
            },
            policies={},
            metadata={"name": "test"},
            nodes=[],
        )
        orchestrator = factory.create_orchestrator(config)
        assert orchestrator is not None

    def test_instantiate_ports(self) -> None:
        """Test instantiating ports from specs."""
        factory = OrchestratorFactory()
        port_specs = {
            "llm": {"adapter": "hexdag.builtin.adapters.mock.MockLLM"},
            "memory": {"adapter": "hexdag.builtin.adapters.memory.InMemoryMemory"},
        }
        ports = factory._instantiate_ports(port_specs)

        assert "llm" in ports
        assert "memory" in ports
        assert ports["llm"] is not None
        assert ports["memory"] is not None

    def test_instantiate_ports_error(self) -> None:
        """Test error handling when port instantiation fails."""
        from hexdag.core.pipeline_builder.component_instantiator import (
            ComponentInstantiationError,
        )

        factory = OrchestratorFactory()
        port_specs = {"bad": {"adapter": "nonexistent.module.Adapter"}}

        with pytest.raises(ComponentInstantiationError):
            factory._instantiate_ports(port_specs)

    def test_instantiate_policies(self) -> None:
        """Test instantiating policies from specs."""
        factory = OrchestratorFactory()
        policy_specs = {
            "retry": {
                "name": "hexdag.builtin.policies.execution_policies.RetryPolicy",
                "params": {"max_retries": 5},
            },
            "timeout": {
                "name": "hexdag.builtin.policies.execution_policies.TimeoutPolicy",
                "params": {"timeout_seconds": 60.0},
            },
        }
        policies = factory._instantiate_policies(policy_specs)

        assert "retry" in policies
        assert "timeout" in policies

    def test_instantiate_policies_error(self) -> None:
        """Test error handling when policy instantiation fails."""
        from hexdag.core.pipeline_builder.component_instantiator import (
            ComponentInstantiationError,
        )

        factory = OrchestratorFactory()
        policy_specs = {"bad": {"name": "nonexistent.module.Policy"}}

        with pytest.raises(ComponentInstantiationError):
            factory._instantiate_policies(policy_specs)

    def test_subscribe_policies_to_manager(self) -> None:
        """Test subscribing policies to manager."""
        factory = OrchestratorFactory()

        # Create mock policy manager
        policy_manager = MagicMock()
        policy_manager.subscribe = MagicMock()

        # Create mock policy
        policy = MagicMock()
        policies = {"retry": policy}

        factory._subscribe_policies_to_manager(policy_manager, policies)

        policy_manager.subscribe.assert_called_once_with(policy)

    def test_subscribe_policies_no_subscribe_method(self) -> None:
        """Test handling policy manager without subscribe method."""
        factory = OrchestratorFactory()

        # Create mock without subscribe method
        policy_manager = MagicMock(spec=[])  # Empty spec = no methods

        policy = MagicMock()
        policies = {"retry": policy}

        # Should not raise, just log warning
        factory._subscribe_policies_to_manager(policy_manager, policies)

    def test_subscribe_policies_error_handling(self) -> None:
        """Test error handling during policy subscription."""
        factory = OrchestratorFactory()

        # Create mock that raises on subscribe
        policy_manager = MagicMock()
        policy_manager.subscribe = MagicMock(side_effect=Exception("Subscribe failed"))

        policy = MagicMock()
        policies = {"retry": policy}

        # Should not raise, just log warning
        factory._subscribe_policies_to_manager(policy_manager, policies)

    def test_create_ports_builder(self) -> None:
        """Test creating a PortsBuilder from config."""
        factory = OrchestratorFactory()
        config = PipelineConfig(
            ports={
                "llm": {"adapter": "hexdag.builtin.adapters.mock.MockLLM"},
            },
            type_ports={},
            policies={},
            metadata={"name": "test"},
            nodes=[],
        )

        builder = factory.create_ports_builder(config)
        assert builder is not None

    def test_create_ports_builder_multiple_ports(self) -> None:
        """Test creating PortsBuilder with multiple ports."""
        factory = OrchestratorFactory()
        config = PipelineConfig(
            ports={
                "llm": {"adapter": "hexdag.builtin.adapters.mock.MockLLM"},
                "memory": {"adapter": "hexdag.builtin.adapters.memory.InMemoryMemory"},
            },
            type_ports={},
            policies={},
            metadata={"name": "test"},
            nodes=[],
        )

        builder = factory.create_ports_builder(config)
        ports = builder.build()

        assert "llm" in ports
        assert "memory" in ports


class TestOrchestratorFactoryIntegration:
    """Integration tests for OrchestratorFactory."""

    def test_full_pipeline_config(self) -> None:
        """Test creating orchestrator from full pipeline config."""
        factory = OrchestratorFactory()
        config = PipelineConfig(
            ports={
                "llm": {
                    "adapter": "hexdag.builtin.adapters.mock.MockLLM",
                    "config": {},
                },
                "memory": {
                    "adapter": "hexdag.builtin.adapters.memory.InMemoryMemory",
                    "config": {},
                },
            },
            type_ports={},
            policies={
                "retry": {
                    "name": "hexdag.builtin.policies.execution_policies.RetryPolicy",
                    "params": {"max_retries": 3},
                },
            },
            metadata={"name": "integration-test", "version": "1.0.0"},
            nodes=[],
        )

        orchestrator = factory.create_orchestrator(
            config,
            max_concurrent_nodes=5,
            default_node_timeout=60.0,
        )

        assert orchestrator is not None

    def test_create_orchestrator_with_policy_manager(self) -> None:
        """Test orchestrator creation with policy manager port."""
        factory = OrchestratorFactory()

        # Create mock policy manager
        mock_policy_manager = MagicMock()
        mock_policy_manager.subscribe = MagicMock()

        config = PipelineConfig(
            ports={},
            type_ports={},
            policies={
                "retry": {
                    "name": "hexdag.builtin.policies.execution_policies.RetryPolicy",
                },
            },
            metadata={"name": "test"},
            nodes=[],
        )

        orchestrator = factory.create_orchestrator(
            config, additional_ports={"policy_manager": mock_policy_manager}
        )

        assert orchestrator is not None
        # Verify policies were subscribed
        mock_policy_manager.subscribe.assert_called()
