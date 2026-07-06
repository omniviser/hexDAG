"""Tests for orchestrator_factory module.

This module tests the orchestrator factory that creates orchestrator instances
from pipeline configuration.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hexdag.kernel.domain.pipeline_config import PipelineConfig
from hexdag.kernel.orchestration.orchestrator_factory import OrchestratorFactory


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
                    "adapter": "hexdag.stdlib.adapters.mock.MockLLM",
                }
            },
            type_ports={},
            policies={},
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
                "llm": {"adapter": "hexdag.stdlib.adapters.mock.MockLLM"},
            },
            type_ports={
                "agent": {
                    "llm": {"adapter": "hexdag.stdlib.adapters.mock.MockLLM"},
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
            "llm": {"adapter": "hexdag.stdlib.adapters.mock.MockLLM"},
            "memory": {"adapter": "hexdag.stdlib.adapters.memory.InMemoryMemory"},
        }
        ports = factory._instantiate_ports(port_specs)

        assert "llm" in ports
        assert "memory" in ports
        assert ports["llm"] is not None
        assert ports["memory"] is not None

    def test_instantiate_ports_error(self) -> None:
        """Test error handling when port instantiation fails."""
        from hexdag.compiler.component_instantiator import (
            ComponentInstantiationError,
        )

        factory = OrchestratorFactory()
        port_specs = {"bad": {"adapter": "nonexistent.module.Adapter"}}

        with pytest.raises(ComponentInstantiationError):
            factory._instantiate_ports(port_specs)

    def test_create_ports_builder(self) -> None:
        """Test creating a PortsBuilder from config."""
        factory = OrchestratorFactory()
        config = PipelineConfig(
            ports={
                "llm": {"adapter": "hexdag.stdlib.adapters.mock.MockLLM"},
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
                "llm": {"adapter": "hexdag.stdlib.adapters.mock.MockLLM"},
                "memory": {"adapter": "hexdag.stdlib.adapters.memory.InMemoryMemory"},
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


class TestOrchestratorFactoryServiceOverrides:
    """Tests for service_overrides parameter."""

    def test_service_overrides_merged_into_services(self) -> None:
        """Overrides appear in the orchestrator's services dict."""
        factory = OrchestratorFactory()
        config = PipelineConfig(
            ports={},
            type_ports={},
            policies={},
            metadata={"name": "test"},
            nodes=[],
        )
        sentinel = object()
        orchestrator = factory.create_orchestrator(
            config,
            service_overrides={"my_service": sentinel},
        )
        services = orchestrator.ports["_hexdag_services"]
        assert services["my_service"] is sentinel

    def test_service_overrides_prevents_fresh_entity_state(self) -> None:
        """When entity_state is in overrides, factory skips auto-registration."""
        from hexdag.stdlib.lib.entity_state import EntityState

        factory = OrchestratorFactory()
        shared_es = EntityState()
        config = PipelineConfig(
            ports={},
            type_ports={},
            policies={},
            metadata={"name": "test"},
            nodes=[],
            state_machines={
                "ticket": {
                    "initial": "OPEN",
                    "transitions": {"OPEN": {"CLOSED"}},
                },
            },
        )
        orchestrator = factory.create_orchestrator(
            config,
            service_overrides={"entity_state": shared_es},
        )
        services = orchestrator.ports["_hexdag_services"]
        assert services["entity_state"] is shared_es

    def test_none_service_overrides_preserves_existing_behaviour(self) -> None:
        """Without overrides, EntityState is auto-created from state_machines."""
        from hexdag.stdlib.lib.entity_state import EntityState

        factory = OrchestratorFactory()
        config = PipelineConfig(
            ports={},
            type_ports={},
            policies={},
            metadata={"name": "test"},
            nodes=[],
            state_machines={
                "ticket": {
                    "initial": "OPEN",
                    "transitions": {"OPEN": {"CLOSED"}},
                },
            },
        )
        orchestrator = factory.create_orchestrator(config)
        services = orchestrator.ports["_hexdag_services"]
        assert isinstance(services["entity_state"], EntityState)

    def test_pipeline_memory_still_auto_registered_with_overrides(self) -> None:
        """PipelineMemory auto-registration is not blocked by service_overrides."""
        from hexdag.stdlib.lib.pipeline_memory import PipelineMemory

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
            service_overrides={"custom": "value"},
        )
        services = orchestrator.ports["_hexdag_services"]
        assert isinstance(services["pipeline_memory"], PipelineMemory)


class CountingService:
    """Module-level service class, instantiated via resolver in tests."""

    instantiations = 0

    def __init__(self) -> None:
        type(self).instantiations += 1


class TestServicesAccessor:
    def test_services_property_exposes_resolved_instances(self) -> None:
        """Host code reaches the same instances pipeline runs use."""
        factory = OrchestratorFactory()
        config = PipelineConfig(
            ports={},
            type_ports={},
            policies={},
            metadata={"name": "test"},
            nodes=[],
        )
        sentinel = object()
        orchestrator = factory.create_orchestrator(
            config,
            service_overrides={"my_service": sentinel},
        )
        assert orchestrator.services["my_service"] is sentinel
        assert "pipeline_memory" in orchestrator.services

    def test_services_property_empty_without_services(self) -> None:
        from hexdag.kernel.orchestration.orchestrator import Orchestrator

        orchestrator = Orchestrator(ports={})
        assert orchestrator.services == {}

    def test_override_skips_yaml_instantiation(self) -> None:
        """A YAML-declared service covered by an override is never built."""
        factory = OrchestratorFactory()
        config = PipelineConfig(
            ports={},
            type_ports={},
            policies={},
            metadata={"name": "test"},
            nodes=[],
            services={
                "counter": {
                    "class": (
                        "tests.hexdag.kernel.orchestration"
                        ".test_orchestrator_factory.CountingService"
                    ),
                },
            },
        )
        CountingService.instantiations = 0
        injected = CountingService()
        assert CountingService.instantiations == 1

        orchestrator = factory.create_orchestrator(
            config,
            service_overrides={"counter": injected},
        )
        assert orchestrator.services["counter"] is injected
        # The YAML spec was NOT instantiated on top of the override
        assert CountingService.instantiations == 1


class TestOrchestratorFactoryIntegration:
    """Integration tests for OrchestratorFactory."""

    def test_full_pipeline_config(self) -> None:
        """Test creating orchestrator from full pipeline config."""
        factory = OrchestratorFactory()
        config = PipelineConfig(
            ports={
                "llm": {
                    "adapter": "hexdag.stdlib.adapters.mock.MockLLM",
                    "config": {},
                },
                "memory": {
                    "adapter": "hexdag.stdlib.adapters.memory.InMemoryMemory",
                    "config": {},
                },
            },
            type_ports={},
            policies={},
            metadata={"name": "integration-test", "version": "1.0.0"},
            nodes=[],
        )

        orchestrator = factory.create_orchestrator(
            config,
            max_concurrent_nodes=5,
            default_node_timeout=60.0,
        )

        assert orchestrator is not None


class TestAdapterPools:
    """Tests for multi-adapter pool ports (``adapters:`` list + ``strategy``)."""

    def test_pool_instantiation(self) -> None:
        from hexdag.stdlib.middleware.round_robin import RoundRobin

        factory = OrchestratorFactory()
        ports = factory._instantiate_ports({
            "llm": {
                "adapters": [
                    {"adapter": "llm:mock"},
                    {"adapter": "hexdag.stdlib.adapters.mock.MockLLM"},
                ],
                "strategy": "failover",
            }
        })

        pool = ports["llm"]
        assert isinstance(pool, RoundRobin)
        assert len(pool._adapters) == 2
        assert pool._strategy == "failover"

    def test_pool_default_strategy_is_round_robin(self) -> None:
        from hexdag.stdlib.middleware.round_robin import RoundRobin

        factory = OrchestratorFactory()
        ports = factory._instantiate_ports({"llm": {"adapters": [{"adapter": "llm:mock"}]}})

        pool = ports["llm"]
        assert isinstance(pool, RoundRobin)
        assert pool._strategy == "round_robin"

    def test_pool_member_config_applied(self) -> None:
        factory = OrchestratorFactory()
        ports = factory._instantiate_ports({
            "llm": {
                "adapters": [
                    {"adapter": "llm:mock", "config": {"responses": ["one"]}},
                    {"adapter": "llm:mock", "config": {"responses": ["two"]}},
                ]
            }
        })

        members = ports["llm"]._adapters
        assert members[0].responses == ["one"]
        assert members[1].responses == ["two"]

    def test_pool_conflicting_keys_error(self) -> None:
        from hexdag.kernel.exceptions import ComponentInstantiationError

        factory = OrchestratorFactory()
        with pytest.raises(ComponentInstantiationError, match="mutually exclusive"):
            factory._instantiate_ports({
                "llm": {
                    "adapter": "llm:mock",
                    "adapters": [{"adapter": "llm:mock"}],
                }
            })

    def test_pool_empty_list_error(self) -> None:
        from hexdag.kernel.exceptions import ComponentInstantiationError

        factory = OrchestratorFactory()
        with pytest.raises(ComponentInstantiationError, match="non-empty list"):
            factory._instantiate_ports({"llm": {"adapters": []}})

    def test_pool_invalid_strategy_error(self) -> None:
        from hexdag.kernel.exceptions import ComponentInstantiationError

        factory = OrchestratorFactory()
        with pytest.raises(ComponentInstantiationError, match=r"(?i)invalid strategy"):
            factory._instantiate_ports({
                "llm": {
                    "adapters": [{"adapter": "llm:mock"}],
                    "strategy": "primary",
                }
            })

    def test_strategy_without_adapters_error(self) -> None:
        from hexdag.kernel.exceptions import ComponentInstantiationError

        factory = OrchestratorFactory()
        with pytest.raises(ComponentInstantiationError, match="without 'adapters'"):
            factory._instantiate_ports({"llm": {"adapter": "llm:mock", "strategy": "failover"}})

    def test_create_orchestrator_with_pool_and_middleware(self) -> None:
        """Pool spec composes with the existing middleware key."""
        factory = OrchestratorFactory()
        config = PipelineConfig(
            ports={
                "llm": {
                    "adapters": [
                        {"adapter": "llm:mock"},
                        {"adapter": "llm:mock"},
                    ],
                    "strategy": "round_robin",
                    "middleware": [],
                }
            },
            type_ports={},
            policies={},
            metadata={"name": "pool-test"},
            nodes=[],
        )

        orchestrator = factory.create_orchestrator(config)
        assert orchestrator is not None


class TestStateMachineHandlerLists:
    """YAML handlers.on_transition accepts a single path or a list of paths."""

    @staticmethod
    def _install_handler_module():
        import sys

        module = type(sys)("handler_test_module")

        async def handler_one(**kwargs):
            return None

        async def handler_two(**kwargs):
            return None

        module.handler_one = handler_one
        module.handler_two = handler_two
        sys.modules["handler_test_module"] = module
        return module

    @staticmethod
    def _machine(on_transition) -> dict:
        return {
            "ticket": {
                "initial": "OPEN",
                "transitions": {"OPEN": ["CLOSED"]},
                "handlers": {"on_transition": on_transition},
            }
        }

    def test_single_string_registers_one_handler(self) -> None:
        module = self._install_handler_module()
        services: dict = {}
        OrchestratorFactory._register_state_machines(
            self._machine("handler_test_module.handler_one"), services
        )
        entity_state = services["entity_state"]
        handlers = entity_state._transition_handlers["ticket"]
        assert len(handlers) == 1
        assert handlers[0][0] is module.handler_one

    def test_list_registers_handlers_in_order(self) -> None:
        module = self._install_handler_module()
        services: dict = {}
        OrchestratorFactory._register_state_machines(
            self._machine(["handler_test_module.handler_one", "handler_test_module.handler_two"]),
            services,
        )
        entity_state = services["entity_state"]
        handlers = entity_state._transition_handlers["ticket"]
        assert len(handlers) == 2
        assert handlers[0][0] is module.handler_one
        assert handlers[1][0] is module.handler_two
