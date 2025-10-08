"""Tests for the PortsBuilder fluent interface."""

from unittest.mock import Mock

from hexdag.core.ports_builder import PortsBuilder


class TestPortsBuilder:
    """Test cases for PortsBuilder."""

    def test_empty_builder(self):
        """Test empty builder initialization."""
        builder = PortsBuilder()
        ports = builder.build()

        assert ports == {}
        assert len(builder) == 0
        assert builder.keys() == []

    def test_with_llm(self):
        """Test adding LLM port."""
        mock_llm = Mock()
        builder = PortsBuilder()

        result = builder.with_llm(mock_llm)

        assert result is builder  # Fluent interface
        assert builder.has("llm")
        assert builder.get("llm") is mock_llm

        ports = builder.build()
        assert ports["llm"] is mock_llm

    def test_with_database(self):
        """Test adding database port."""
        mock_db = Mock()
        builder = PortsBuilder()

        result = builder.with_database(mock_db)

        assert result is builder
        assert builder.has("database")
        assert builder.get("database") is mock_db

    def test_with_observer_manager(self):
        """Test adding observer manager."""
        mock_observer = Mock()
        builder = PortsBuilder()

        result = builder.with_observer_manager(mock_observer)

        assert result is builder
        assert builder.has("observer_manager")
        assert builder.get("observer_manager") is mock_observer

    def test_with_policy_manager(self):
        """Test adding policy manager."""
        mock_policy = Mock()
        builder = PortsBuilder()

        result = builder.with_policy_manager(mock_policy)

        assert result is builder
        assert builder.has("policy_manager")
        assert builder.get("policy_manager") is mock_policy

    def test_with_memory(self):
        """Test adding memory port."""
        mock_memory = Mock()
        builder = PortsBuilder()

        result = builder.with_memory(mock_memory)

        assert result is builder
        assert builder.has("memory")
        assert builder.get("memory") is mock_memory

    def test_with_tool_router(self):
        """Test adding tool router."""
        mock_router = Mock()
        builder = PortsBuilder()

        result = builder.with_tool_router(mock_router)

        assert result is builder
        assert builder.has("tool_router")
        assert builder.get("tool_router") is mock_router

    def test_with_api_call(self):
        """Test adding API call port."""
        mock_api = Mock()
        builder = PortsBuilder()

        result = builder.with_api_call(mock_api)

        assert result is builder
        assert builder.has("api_call")
        assert builder.get("api_call") is mock_api

    def test_with_custom(self):
        """Test adding custom port."""
        custom_port = Mock()
        builder = PortsBuilder()

        result = builder.with_custom("my_custom_port", custom_port)

        assert result is builder
        assert builder.has("my_custom_port")
        assert builder.get("my_custom_port") is custom_port

    def test_fluent_chaining(self):
        """Test fluent interface with method chaining."""
        mock_llm = Mock()
        mock_db = Mock()
        mock_observer = Mock()

        ports = (
            PortsBuilder()
            .with_llm(mock_llm)
            .with_database(mock_db)
            .with_observer_manager(mock_observer)
            .build()
        )

        assert len(ports) == 3
        assert ports["llm"] is mock_llm
        assert ports["database"] is mock_db
        assert ports["observer_manager"] is mock_observer

    def test_update_from_dict(self):
        """Test updating from dictionary."""
        existing_ports = {
            "llm": Mock(),
            "database": Mock(),
            "custom_port": Mock(),
        }

        builder = PortsBuilder()
        result = builder.update(existing_ports)

        assert result is builder
        assert len(builder) == 3

        ports = builder.build()
        assert ports == existing_ports

    def test_clear(self):
        """Test clearing all ports."""
        builder = PortsBuilder().with_llm(Mock()).with_database(Mock())

        assert len(builder) == 2

        result = builder.clear()

        assert result is builder
        assert len(builder) == 0
        assert builder.build() == {}

    def test_build_creates_copy(self):
        """Test that build() returns a copy, not the original."""
        builder = PortsBuilder().with_llm(Mock())

        ports1 = builder.build()
        ports2 = builder.build()

        assert ports1 == ports2
        assert ports1 is not ports2  # Different objects

        # Modifying built dict doesn't affect builder
        ports1["new_key"] = "new_value"
        assert not builder.has("new_key")

    def test_get_with_default(self):
        """Test getting port with default value."""
        builder = PortsBuilder()

        assert builder.get("nonexistent") is None
        assert builder.get("nonexistent", "default") == "default"

        builder.with_llm(Mock())
        assert builder.get("llm") is not None
        assert builder.get("llm", "default") != "default"

    def test_keys_method(self):
        """Test getting list of configured keys."""
        builder = (
            PortsBuilder().with_llm(Mock()).with_database(Mock()).with_custom("my_port", Mock())
        )

        keys = builder.keys()
        assert set(keys) == {"llm", "database", "my_port"}

    def test_repr(self):
        """Test string representation."""
        builder = PortsBuilder()
        assert "none" in repr(builder)

        builder.with_llm(Mock()).with_database(Mock())
        repr_str = repr(builder)
        assert "llm" in repr_str
        assert "database" in repr_str

    def test_with_defaults(self):
        """Test adding default implementations."""
        builder = PortsBuilder()

        # This should add defaults without error
        # (actual implementations may not be available in test env)
        result = builder.with_defaults()

        assert result is builder

        # If implementations are available, they should be added
        # Otherwise, builder remains empty (graceful degradation)
        ports = builder.build()
        assert isinstance(ports, dict)

    def test_with_defaults_doesnt_override(self):
        """Test that defaults don't override existing ports."""
        mock_observer = Mock()

        builder = PortsBuilder().with_observer_manager(mock_observer).with_defaults()

        # The mock should still be there, not replaced by default
        assert builder.get("observer_manager") is mock_observer

    def test_complex_scenario(self):
        """Test a complex real-world usage scenario."""
        # Start with some existing config
        existing_config = {
            "database": Mock(),
            "legacy_port": Mock(),
        }

        # Build new config using builder
        mock_llm = Mock()
        mock_observer = Mock()

        ports = (
            PortsBuilder()
            .update(existing_config)  # Import existing
            .with_llm(mock_llm)  # Add new ports
            .with_observer_manager(mock_observer)
            .with_custom("feature_flag", True)  # Custom config
            .build()
        )

        # Verify final configuration
        assert len(ports) == 5
        assert ports["database"] == existing_config["database"]
        assert ports["legacy_port"] == existing_config["legacy_port"]
        assert ports["llm"] is mock_llm
        assert ports["observer_manager"] is mock_observer
        assert ports["feature_flag"] is True


class TestPortsBuilderEnhanced:
    """Test enhanced PortsBuilder with per-node and per-type configuration."""

    def test_for_type_configuration(self):
        """Test configuring ports for a specific node type."""
        mock_llm = Mock()
        openai_llm = Mock()
        openai_llm.name = "openai"

        config = (
            PortsBuilder()
            .with_llm(mock_llm)
            .for_type("agent", llm=openai_llm)
            .build_configuration()
        )

        # Agent nodes get OpenAI
        agent_ports = config.resolve_ports("my_agent", "agent")
        assert agent_ports["llm"].port is openai_llm

        # Other nodes get MockLLM
        function_ports = config.resolve_ports("my_function", "function")
        assert function_ports["llm"].port is mock_llm

    def test_for_node_configuration(self):
        """Test configuring ports for a specific node."""
        mock_llm = Mock()
        openai_llm = Mock()
        anthropic_llm = Mock()

        config = (
            PortsBuilder()
            .with_llm(mock_llm)
            .for_type("agent", llm=openai_llm)
            .for_node("researcher", llm=anthropic_llm)
            .build_configuration()
        )

        # Researcher gets Anthropic
        researcher_ports = config.resolve_ports("researcher", "agent")
        assert researcher_ports["llm"].port is anthropic_llm

        # Other agents get OpenAI
        analyzer_ports = config.resolve_ports("analyzer", "agent")
        assert analyzer_ports["llm"].port is openai_llm

    def test_multiple_type_configurations(self):
        """Test configuring multiple node types."""
        mock_llm = Mock()
        openai_llm = Mock()
        anthropic_llm = Mock()

        config = (
            PortsBuilder()
            .with_llm(mock_llm)
            .for_type("agent", llm=openai_llm)
            .for_type("llm", llm=anthropic_llm)
            .build_configuration()
        )

        agent_ports = config.resolve_ports("my_agent", "agent")
        assert agent_ports["llm"].port is openai_llm

        llm_ports = config.resolve_ports("my_llm_node", "llm")
        assert llm_ports["llm"].port is anthropic_llm

    def test_multiple_ports_per_type(self):
        """Test configuring multiple ports for a type."""
        mock_llm = Mock()
        mock_db = Mock()
        openai_llm = Mock()
        postgres_db = Mock()

        config = (
            PortsBuilder()
            .with_llm(mock_llm)
            .with_database(mock_db)
            .for_type("agent", llm=openai_llm, database=postgres_db)
            .build_configuration()
        )

        agent_ports = config.resolve_ports("my_agent", "agent")
        assert agent_ports["llm"].port is openai_llm
        assert agent_ports["database"].port is postgres_db

    def test_backward_compatibility_with_build(self):
        """Test that build() still works for backward compatibility."""
        mock_llm = Mock()
        mock_db = Mock()

        ports_dict = PortsBuilder().with_llm(mock_llm).with_database(mock_db).build()

        assert "llm" in ports_dict
        assert "database" in ports_dict
        assert ports_dict["llm"] is mock_llm
        assert ports_dict["database"] is mock_db

    def test_chaining_all_methods(self):
        """Test method chaining with all configuration methods."""
        mock_llm = Mock()
        openai_llm = Mock()
        anthropic_llm = Mock()
        mock_db = Mock()

        config = (
            PortsBuilder()
            .with_llm(mock_llm)
            .with_database(mock_db)
            .for_type("agent", llm=openai_llm)
            .for_type("llm", llm=anthropic_llm)
            .for_node("researcher", llm=Mock())
            .build_configuration()
        )

        # Verify all levels work
        researcher_ports = config.resolve_ports("researcher", "agent")
        assert researcher_ports["llm"].port is not openai_llm

        agent_ports = config.resolve_ports("analyzer", "agent")
        assert agent_ports["llm"].port is openai_llm

        function_ports = config.resolve_ports("transformer", "function")
        assert function_ports["llm"].port is mock_llm

    def test_with_defaults_and_overrides(self):
        """Test using with_defaults() alongside type/node overrides."""
        openai_llm = Mock()

        builder = PortsBuilder().with_defaults().for_type("agent", llm=openai_llm)

        # Should have default observer_manager and policy_manager from with_defaults()
        ports_dict = builder.build()
        assert "observer_manager" in ports_dict or "policy_manager" in ports_dict

        # Agent nodes should get OpenAI override
        config = builder.build_configuration()
        agent_ports = config.resolve_ports("my_agent", "agent")
        assert agent_ports["llm"].port is openai_llm
