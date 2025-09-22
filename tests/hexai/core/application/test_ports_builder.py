"""Tests for PortsBuilder - fluent builder for orchestrator ports."""

from unittest.mock import Mock

import pytest

from hexai.adapters.local import InMemoryMemory, LocalObserverManager, LocalPolicyManager
from hexai.adapters.mock import MockDatabaseAdapter, MockLLM, MockToolRouter
from hexai.core.application.orchestrator import Orchestrator
from hexai.core.application.ports_builder import PortsBuilder
from hexai.core.domain.dag import DirectedGraph, NodeSpec
from hexai.core.ports import ObserverManagerPort


class TestPortsBuilder:
    """Test suite for PortsBuilder functionality."""

    def test_basic_port_addition(self):
        """Test adding individual ports to the builder."""
        builder = PortsBuilder()
        mock_llm = MockLLM()
        mock_db = MockDatabaseAdapter()

        builder.with_llm(mock_llm).with_database(mock_db)

        ports = builder.build()
        assert ports["llm"] is mock_llm
        assert ports["database"] is mock_db
        assert len(ports) == 2

    def test_fluent_interface_chaining(self):
        """Test method chaining returns self."""
        builder = PortsBuilder()
        result = builder.with_llm(MockLLM()).with_database(MockDatabaseAdapter())

        assert result is builder
        assert isinstance(result, PortsBuilder)

    def test_all_port_categories(self):
        """Test adding ports from all categories."""
        builder = (
            PortsBuilder()
            .with_llm(MockLLM())
            .with_tool_router(MockToolRouter())
            .with_database(MockDatabaseAdapter())
            .with_memory(InMemoryMemory())
            .with_policy_manager(LocalPolicyManager())
            .with_observer_manager(LocalObserverManager())
        )

        ports = builder.build()
        assert len(ports) == 6
        assert all(
            key in ports
            for key in [
                "llm",
                "tool_router",
                "database",
                "memory",
                "policy_manager",
                "observer_manager",
            ]
        )

    def test_custom_port_keys(self):
        """Test using custom keys for ports."""
        builder = PortsBuilder()
        mock_llm_1 = MockLLM()
        mock_llm_2 = MockLLM()

        builder.with_llm(mock_llm_1, key="primary_llm")
        builder.with_llm(mock_llm_2, key="fallback_llm")

        ports = builder.build()
        assert ports["primary_llm"] is mock_llm_1
        assert ports["fallback_llm"] is mock_llm_2
        assert "llm" not in ports

    def test_custom_ports(self):
        """Test adding custom ports."""
        builder = PortsBuilder()
        custom_service = {"type": "custom", "config": "test"}

        builder.with_custom_port("my_service", custom_service)

        ports = builder.build()
        assert ports["my_service"] is custom_service

    def test_custom_ports_bulk(self):
        """Test adding multiple custom ports at once."""
        builder = PortsBuilder()
        custom_ports = {
            "service_1": {"type": "s1"},
            "service_2": {"type": "s2"},
            "service_3": {"type": "s3"},
        }

        builder.with_custom_ports(custom_ports)

        ports = builder.build()
        assert len(ports) == 3
        for key, value in custom_ports.items():
            assert ports[key] is value

    def test_with_defaults_without_policy(self):
        """Test with_defaults is now a no-op placeholder."""
        builder = PortsBuilder()
        builder.with_defaults()

        ports = builder.build()
        # with_defaults no longer creates managers automatically
        assert "observer_manager" not in ports
        assert "policy_manager" not in ports

    def test_with_defaults_with_policy(self):
        """Test with_defaults is now a no-op placeholder."""
        builder = PortsBuilder()
        # with_defaults no longer accepts parameters
        builder.with_defaults()

        ports = builder.build()
        # with_defaults no longer creates managers automatically
        assert "observer_manager" not in ports
        assert "policy_manager" not in ports

    def test_with_defaults_preserves_existing(self):
        """Test with_defaults doesn't override existing ports."""
        builder = PortsBuilder()
        custom_observer = LocalObserverManager(max_concurrent_observers=100)

        builder.with_observer_manager(custom_observer)
        builder.with_defaults()  # Should be a no-op

        ports = builder.build()
        assert ports["observer_manager"] is custom_observer  # Still preserved

    def test_with_test_defaults(self):
        """Test with_test_defaults is now a no-op placeholder."""
        builder = PortsBuilder()
        builder.with_test_defaults()

        ports = builder.build()
        # with_test_defaults no longer creates managers automatically
        assert "observer_manager" not in ports
        assert "policy_manager" not in ports

    def test_has_port(self):
        """Test checking if a port exists."""
        builder = PortsBuilder().with_llm(MockLLM())

        assert builder.has_port("llm") is True
        assert builder.has_port("database") is False

    def test_get_port(self):
        """Test retrieving a configured port."""
        mock_llm = MockLLM()
        builder = PortsBuilder().with_llm(mock_llm)

        assert builder.get_port("llm") is mock_llm
        assert builder.get_port("nonexistent") is None

    def test_remove_port(self):
        """Test removing a port."""
        builder = PortsBuilder().with_llm(MockLLM()).with_database(MockDatabaseAdapter())

        assert builder.has_port("llm")
        builder.remove_port("llm")
        assert not builder.has_port("llm")
        assert builder.has_port("database")  # Other ports unaffected

    def test_clear(self):
        """Test clearing all ports."""
        builder = (
            PortsBuilder()
            .with_llm(MockLLM())
            .with_database(MockDatabaseAdapter())
            .with_memory(InMemoryMemory())
        )

        assert len(builder.build()) == 3
        builder.clear()
        assert len(builder.build()) == 0

    def test_from_dict(self):
        """Test creating builder from existing dictionary."""
        existing_ports = {
            "llm": MockLLM(),
            "database": MockDatabaseAdapter(),
            "custom": {"type": "custom"},
        }

        builder = PortsBuilder.from_dict(existing_ports)
        ports = builder.build()

        assert len(ports) == 3
        for key in existing_ports:
            assert key in ports

    def test_from_dict_isolation(self):
        """Test from_dict creates isolated copy."""
        original = {"llm": MockLLM()}
        builder = PortsBuilder.from_dict(original)

        builder.with_database(MockDatabaseAdapter())

        # Original should be unchanged
        assert len(original) == 1
        assert len(builder.build()) == 2

    def test_build_returns_copy(self):
        """Test build() returns a copy, not the internal dict."""
        builder = PortsBuilder().with_llm(MockLLM())
        ports1 = builder.build()
        ports2 = builder.build()

        assert ports1 is not ports2  # Different objects
        assert ports1 == ports2  # Same content

    def test_repr_empty(self):
        """Test string representation of empty builder."""
        builder = PortsBuilder()
        assert "empty" in str(builder).lower()

    def test_repr_with_ports(self):
        """Test string representation with configured ports."""
        builder = (
            PortsBuilder()
            .with_llm(MockLLM())
            .with_database(MockDatabaseAdapter())
            .with_observer_manager(LocalObserverManager())
        )

        repr_str = str(builder)
        assert "AI Services" in repr_str
        assert "llm" in repr_str
        assert "Data" in repr_str
        assert "database" in repr_str
        assert "Control" in repr_str
        assert "observer_manager" in repr_str

    def test_repr_with_custom_ports(self):
        """Test string representation includes custom ports."""
        builder = PortsBuilder().with_custom_port("my_custom", {})

        repr_str = str(builder)
        assert "Custom" in repr_str
        assert "my_custom" in repr_str


class TestPortsBuilderObserverManager:
    """Test PortsBuilder observer manager configuration."""

    def test_with_observer_manager_provided_instance(self):
        """Test providing an existing observer manager instance."""
        mock_manager = Mock(spec=ObserverManagerPort)
        builder = PortsBuilder()

        result = builder.with_observer_manager(mock_manager)

        assert result is builder  # Fluent interface
        ports = builder.build()
        assert ports["observer_manager"] is mock_manager

        # Test providing a configured LocalObserverManager
        configured_manager = LocalObserverManager(
            max_concurrent_observers=20,
            observer_timeout=10.0,
            max_sync_workers=8,
            use_weak_refs=False,
        )
        builder2 = PortsBuilder()
        result = builder2.with_observer_manager(configured_manager)

        assert result is builder2
        ports = builder2.build()
        assert ports["observer_manager"] is configured_manager

    def test_with_local_observer_manager(self):
        """Test explicitly creating a LocalObserverManager."""
        # with_local_observer_manager no longer exists
        manager = LocalObserverManager(
            max_concurrent_observers=25,
            observer_timeout=12.0,
            max_sync_workers=10,
            use_weak_refs=False,
        )

        builder = PortsBuilder()
        result = builder.with_observer_manager(manager)

        assert result is builder
        ports = builder.build()

        assert ports["observer_manager"] is manager
        assert manager._max_concurrent == 25
        assert manager._timeout == 12.0
        assert manager._executor._max_workers == 10
        assert manager._use_weak_refs is False

    def test_with_local_observer_manager_defaults(self):
        """Test LocalObserverManager with default values."""
        # Create manager with defaults
        manager = LocalObserverManager()

        builder = PortsBuilder()
        builder.with_observer_manager(manager)

        ports = builder.build()
        retrieved = ports["observer_manager"]

        assert retrieved is manager
        # Check defaults
        assert manager._max_concurrent == 10
        assert manager._timeout == 5.0
        assert manager._executor._max_workers == 4
        assert manager._use_weak_refs is True

    def test_with_observer_config(self):
        """Test configuring observer manager from a dictionary."""
        config = {
            "max_concurrent_observers": 30,
            "observer_timeout": 15.0,
            "max_sync_workers": 12,
            "use_weak_refs": True,
        }

        # with_observer_config no longer exists, create manager directly
        manager = LocalObserverManager(**config)

        builder = PortsBuilder()
        result = builder.with_observer_manager(manager)

        assert result is builder
        ports = builder.build()

        assert ports["observer_manager"] is manager
        assert manager._max_concurrent == 30
        assert manager._timeout == 15.0
        assert manager._executor._max_workers == 12
        assert manager._use_weak_refs is True

    def test_with_observer_config_invalid_keys(self):
        """Test that invalid configuration keys raise an error."""
        config = {
            "max_concurrent_observers": 10,
            "invalid_key": "value",
            "another_bad_key": 123,
        }

        # Error should come from LocalObserverManager constructor
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            LocalObserverManager(**config)

    def test_with_defaults_creates_local_observer_manager(self):
        """Test that with_defaults no longer creates managers."""
        builder = PortsBuilder()

        builder.with_defaults()

        ports = builder.build()
        # with_defaults no longer creates managers automatically
        assert "observer_manager" not in ports
        assert "policy_manager" not in ports

    def test_with_defaults_preserves_existing_observer_manager(self):
        """Test that with_defaults doesn't overwrite existing observer manager."""
        mock_manager = Mock(spec=ObserverManagerPort)
        builder = PortsBuilder()

        builder.with_observer_manager(mock_manager)
        builder.with_defaults()

        ports = builder.build()
        assert ports["observer_manager"] is mock_manager

    def test_with_test_defaults_creates_minimal_observer_manager(self):
        """Test that with_test_defaults no longer creates managers."""
        builder = PortsBuilder()

        builder.with_test_defaults()

        ports = builder.build()
        # with_test_defaults no longer creates managers automatically
        assert "observer_manager" not in ports
        assert "policy_manager" not in ports

    def test_backward_compatibility_with_existing_code(self):
        """Test that existing code patterns still work."""
        # Old pattern: directly passing an ObserverManager instance
        old_manager = LocalObserverManager()
        builder = PortsBuilder()

        builder.with_observer_manager(old_manager)

        ports = builder.build()
        assert ports["observer_manager"] is old_manager


class TestOrchestratorIntegration:
    """Test PortsBuilder integration with Orchestrator."""

    def test_orchestrator_with_builder(self):
        """Test passing PortsBuilder to Orchestrator."""
        builder = (
            PortsBuilder()
            .with_llm(MockLLM())
            .with_observer_manager(LocalObserverManager())
            .with_policy_manager(LocalPolicyManager())
        )
        orchestrator = Orchestrator(ports=builder)

        assert "llm" in orchestrator.ports
        assert "observer_manager" in orchestrator.ports
        assert "policy_manager" in orchestrator.ports
        assert isinstance(orchestrator.ports["llm"], MockLLM)

    def test_orchestrator_with_dict(self):
        """Test backward compatibility with dictionary."""
        ports_dict = {"llm": MockLLM(), "database": MockDatabaseAdapter()}
        orchestrator = Orchestrator(ports=ports_dict)

        assert orchestrator.ports == ports_dict

    def test_orchestrator_with_none(self):
        """Test Orchestrator requires both managers when ports=None."""
        # Since we can't create defaults, Orchestrator will need explicit managers
        # Create minimal ports for testing
        orchestrator = Orchestrator(
            ports={
                "observer_manager": LocalObserverManager(),
                "policy_manager": LocalPolicyManager(),
            }
        )

        assert "observer_manager" in orchestrator.ports
        assert "policy_manager" in orchestrator.ports
        assert isinstance(orchestrator.ports["observer_manager"], LocalObserverManager)

    def test_orchestrator_default_no_args(self):
        """Test Orchestrator requires explicit managers."""
        # Provide required managers explicitly
        orchestrator = Orchestrator(
            ports={
                "observer_manager": LocalObserverManager(),
                "policy_manager": LocalPolicyManager(),
            }
        )

        assert "observer_manager" in orchestrator.ports
        assert "policy_manager" in orchestrator.ports

    @pytest.mark.asyncio
    async def test_end_to_end_with_builder(self):
        """Test end-to-end execution using PortsBuilder."""

        async def test_node(data: dict, **_kwargs) -> dict:
            return {"result": f"processed: {data}"}

        graph = DirectedGraph()
        graph.add(NodeSpec("test", test_node))

        # Using PortsBuilder with explicit managers
        builder = (
            PortsBuilder()
            .with_observer_manager(LocalObserverManager())
            .with_policy_manager(LocalPolicyManager())
        )
        orchestrator = Orchestrator(ports=builder)

        result = await orchestrator.run(graph, {"input": "test"})
        assert "test" in result
        assert "processed" in str(result["test"])

    @pytest.mark.asyncio
    async def test_end_to_end_with_dict(self):
        """Test end-to-end execution with traditional dictionary."""

        async def test_node(data: dict, **_kwargs) -> dict:
            return {"result": f"processed: {data}"}

        graph = DirectedGraph()
        graph.add(NodeSpec("test", test_node))

        # Using traditional dictionary (needs policy_manager for current orchestrator)
        ports_dict = {
            "observer_manager": LocalObserverManager(),
            "policy_manager": LocalPolicyManager(),
        }
        orchestrator = Orchestrator(ports=ports_dict)

        result = await orchestrator.run(graph, {"input": "test"})
        assert "test" in result

    @pytest.mark.asyncio
    async def test_builder_with_llm_node(self):
        """Test using builder with LLM nodes."""
        from hexai.core.application.nodes import LLMNode

        mock_llm = MockLLM(responses=["Test response"])

        graph = DirectedGraph()
        llm_node_factory = LLMNode()
        node_spec = llm_node_factory(
            "analyzer",
            template="Analyze: {{input}}",
            output_key="analysis",
        )
        graph.add(node_spec)

        builder = (
            PortsBuilder()
            .with_llm(mock_llm)
            .with_observer_manager(LocalObserverManager())
            .with_policy_manager(LocalPolicyManager())
        )
        orchestrator = Orchestrator(ports=builder)

        result = await orchestrator.run(graph, {"input": "test data"})
        assert "analyzer" in result
        # The result should be the string directly from the LLM
        assert "Test response" in str(result["analyzer"])


class TestPortsBuilderPatterns:
    """Test common usage patterns."""

    def test_migration_pattern(self):
        """Test migrating from dictionary to builder."""
        # Old pattern
        old_ports = {
            "llm": MockLLM(),
            "database": MockDatabaseAdapter(),
            "observer_manager": LocalObserverManager(),
        }

        # Migration path
        builder = PortsBuilder.from_dict(old_ports)

        new_ports = builder.build()
        assert len(new_ports) == 3
        assert all(k in new_ports for k in old_ports)

    def test_multi_llm_pattern(self):
        """Test configuring multiple LLMs for different purposes."""
        builder = (
            PortsBuilder()
            .with_llm(MockLLM(responses=["fast"]), key="fast_llm")
            .with_llm(MockLLM(responses=["accurate"]), key="accurate_llm")
            .with_llm(MockLLM(responses=["creative"]), key="creative_llm")
        )

        ports = builder.build()
        assert len(ports) == 3
        assert all(k in ports for k in ["fast_llm", "accurate_llm", "creative_llm"])

    def test_environment_specific_pattern(self):
        """Test building environment-specific configurations."""

        def build_dev_ports():
            # Create minimal test configuration manually
            return (
                PortsBuilder()
                .with_observer_manager(
                    LocalObserverManager(
                        max_concurrent_observers=1,
                        observer_timeout=1.0,
                        max_sync_workers=1,
                        use_weak_refs=False,
                    )
                )
                .with_llm(MockLLM())
                .build()
            )

        def build_prod_ports():
            return (
                PortsBuilder()
                .with_observer_manager(LocalObserverManager())
                .with_policy_manager(LocalPolicyManager())
                .with_llm(MockLLM())  # Would be real LLM in production
                .with_database(MockDatabaseAdapter())  # Would be real DB
                .build()
            )

        dev_ports = build_dev_ports()
        prod_ports = build_prod_ports()

        # Dev has minimal config
        dev_manager = dev_ports["observer_manager"]
        assert isinstance(dev_manager, LocalObserverManager)
        assert dev_manager._max_concurrent == 1

        # Prod has full config
        assert "policy_manager" in prod_ports

    def test_override_pattern(self):
        """Test overriding specific ports while keeping defaults."""
        custom_observer = LocalObserverManager(max_concurrent_observers=50)

        builder = (
            PortsBuilder()
            .with_policy_manager(LocalPolicyManager())
            .with_observer_manager(custom_observer)  # Use custom observer
        )

        ports = builder.build()
        assert ports["observer_manager"] is custom_observer
        assert "policy_manager" in ports  # Policy manager preserved from defaults
