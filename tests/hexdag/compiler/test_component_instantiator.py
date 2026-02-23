"""Tests for the component instantiator module.

This module tests instantiation of adapters and policies from YAML specs.
"""

from __future__ import annotations

import pytest

from hexdag.compiler.component_instantiator import (
    ComponentInstantiationError,
    ComponentInstantiator,
    ComponentSpec,
    _resolve_deferred_env_vars,
    _resolve_string_value,
)


class TestComponentSpec:
    """Tests for ComponentSpec namedtuple."""

    def test_component_spec_creation(self) -> None:
        """Test creating a ComponentSpec."""
        spec = ComponentSpec(
            module_path="hexdag.stdlib.adapters.mock.MockLLM",
            params={"model": "gpt-4"},
        )
        assert spec.module_path == "hexdag.stdlib.adapters.mock.MockLLM"
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
            "adapter": "hexdag.stdlib.adapters.mock.MockLLM",
            "config": {"model": "gpt-4"},
        }
        result = self.instantiator._parse_component_spec(spec)
        assert result.module_path == "hexdag.stdlib.adapters.mock.MockLLM"
        assert result.params == {"model": "gpt-4"}

    def test_parse_name_format(self) -> None:
        """Test parsing spec with 'name' key."""
        spec = {
            "name": "hexdag.stdlib.policies.execution_policies.RetryPolicy",
            "params": {"max_retries": 3},
        }
        result = self.instantiator._parse_component_spec(spec)
        assert result.module_path == "hexdag.stdlib.policies.execution_policies.RetryPolicy"
        assert result.params == {"max_retries": 3}

    def test_parse_adapter_with_params_key(self) -> None:
        """Test parsing adapter spec with 'params' instead of 'config'."""
        spec = {
            "adapter": "hexdag.stdlib.adapters.mock.MockLLM",
            "params": {"model": "gpt-4"},
        }
        result = self.instantiator._parse_component_spec(spec)
        assert result.params == {"model": "gpt-4"}

    def test_parse_with_no_params(self) -> None:
        """Test parsing spec without params."""
        spec = {"adapter": "hexdag.stdlib.adapters.mock.MockLLM"}
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
        spec = {"adapter": "hexdag.stdlib.adapters.mock.MockLLM"}
        adapter = self.instantiator.instantiate_adapter(spec, port_name="llm")
        assert adapter is not None
        # Check it has the expected interface
        assert hasattr(adapter, "aresponse")

    def test_instantiate_mock_database_adapter(self) -> None:
        """Test instantiating MockDatabaseAdapter."""
        spec = {"adapter": "hexdag.stdlib.adapters.mock.MockDatabaseAdapter"}
        adapter = self.instantiator.instantiate_adapter(spec, port_name="database")
        assert adapter is not None

    def test_instantiate_with_config_params(self) -> None:
        """Test instantiating adapter with configuration parameters."""
        spec = {
            "adapter": "hexdag.stdlib.adapters.mock.MockLLM",
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
        spec = {"adapter": "hexdag.stdlib.adapters.mock.NonExistentAdapter"}
        with pytest.raises(ComponentInstantiationError) as exc_info:
            self.instantiator.instantiate_adapter(spec, port_name="test")
        assert "could not be resolved" in str(exc_info.value)

    def test_instantiate_with_extra_params_accepted(self) -> None:
        """Test that extra params are accepted (passed as kwargs)."""
        spec = {
            "adapter": "hexdag.stdlib.adapters.mock.MockLLM",
            "config": {"extra_param": True},
        }
        # MockLLM accepts **kwargs, so extra params don't raise
        adapter = self.instantiator.instantiate_adapter(spec, port_name="llm")
        assert adapter is not None

    def test_instantiate_adapter_without_port_name(self) -> None:
        """Test instantiating adapter without port_name."""
        spec = {"adapter": "hexdag.stdlib.adapters.mock.MockLLM"}
        adapter = self.instantiator.instantiate_adapter(spec)
        assert adapter is not None


class TestInstantiatePorts:
    """Tests for instantiate_ports method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.instantiator = ComponentInstantiator()

    def test_instantiate_multiple_ports(self) -> None:
        """Test instantiating multiple ports."""
        ports_config = {
            "llm": {"adapter": "hexdag.stdlib.adapters.mock.MockLLM"},
            "database": {"adapter": "hexdag.stdlib.adapters.mock.MockDatabaseAdapter"},
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
            "llm": {"adapter": "hexdag.stdlib.adapters.mock.MockLLM"},
            "bad": {"adapter": "nonexistent.Adapter"},
        }
        with pytest.raises(ComponentInstantiationError):
            self.instantiator.instantiate_ports(ports_config)


class TestComponentInstantiatorIntegration:
    """Integration tests for ComponentInstantiator."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.instantiator = ComponentInstantiator()

    def test_full_pipeline_config_instantiation(self) -> None:
        """Test instantiating a complete pipeline configuration."""
        ports_config = {
            "llm": {
                "adapter": "hexdag.stdlib.adapters.mock.MockLLM",
                "config": {},
            },
            "memory": {
                "adapter": "hexdag.stdlib.adapters.memory.InMemoryMemory",
                "config": {},
            },
        }

        ports = self.instantiator.instantiate_ports(ports_config)

        assert len(ports) == 2
        assert "llm" in ports
        assert "memory" in ports

    def test_instantiate_all_mock_adapters(self) -> None:
        """Test that all mock adapters can be instantiated."""
        mock_adapters = [
            "hexdag.stdlib.adapters.mock.MockLLM",
            "hexdag.stdlib.adapters.mock.MockDatabaseAdapter",
            "hexdag.stdlib.adapters.mock.MockEmbedding",
        ]
        for adapter_path in mock_adapters:
            spec = {"adapter": adapter_path}
            adapter = self.instantiator.instantiate_adapter(spec)
            assert adapter is not None, f"Failed to instantiate {adapter_path}"


class TestDeferredEnvVarResolution:
    """Tests for runtime environment variable resolution."""

    def test_resolve_simple_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolving a simple ${VAR} pattern."""
        monkeypatch.setenv("TEST_API_KEY", "secret123")
        result = _resolve_string_value("${TEST_API_KEY}")
        assert result == "secret123"

    def test_resolve_env_var_with_default_uses_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that env var takes precedence over default."""
        monkeypatch.setenv("TEST_MODEL", "gpt-4-turbo")
        result = _resolve_string_value("${TEST_MODEL:gpt-4}")
        assert result == "gpt-4-turbo"

    def test_resolve_env_var_with_default_uses_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that default is used when env var not set."""
        monkeypatch.delenv("UNSET_VAR", raising=False)
        result = _resolve_string_value("${UNSET_VAR:default_value}")
        assert result == "default_value"

    def test_resolve_missing_env_var_raises_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that missing required env var raises error."""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        with pytest.raises(ComponentInstantiationError) as exc_info:
            _resolve_string_value("${MISSING_VAR}")
        assert "MISSING_VAR" in str(exc_info.value)
        assert "not set" in str(exc_info.value)

    def test_missing_env_var_error_includes_phase_label(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that deferred env var error identifies Phase 3b."""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        with pytest.raises(ComponentInstantiationError, match=r"\[Phase 3b"):
            _resolve_string_value("${MISSING_VAR}")

    def test_missing_env_var_error_includes_secret_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that deferred env var error explains secret deferral."""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        with pytest.raises(ComponentInstantiationError, match=r"secret|deferred"):
            _resolve_string_value("${MISSING_VAR}")

    def test_resolve_env_var_in_middle_of_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolving ${VAR} in the middle of a string."""
        monkeypatch.setenv("HOST", "localhost")
        result = _resolve_string_value("http://${HOST}:8080/api")
        assert result == "http://localhost:8080/api"

    def test_resolve_multiple_env_vars_in_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolving multiple ${VAR} patterns in one string."""
        monkeypatch.setenv("DB_HOST", "localhost")
        monkeypatch.setenv("DB_PORT", "5432")
        result = _resolve_string_value("postgresql://${DB_HOST}:${DB_PORT}/mydb")
        assert result == "postgresql://localhost:5432/mydb"

    def test_resolve_dict_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolving env vars in dict parameters."""
        monkeypatch.setenv("API_KEY", "key123")
        monkeypatch.setenv("MODEL", "gpt-4")
        params = {
            "api_key": "${API_KEY}",
            "model": "${MODEL}",
            "timeout": 30,
        }
        resolved = _resolve_deferred_env_vars(params)
        assert resolved["api_key"] == "key123"
        assert resolved["model"] == "gpt-4"
        assert resolved["timeout"] == 30

    def test_resolve_nested_dict_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolving env vars in nested dict parameters."""
        monkeypatch.setenv("SECRET_TOKEN", "tok123")
        params = {
            "auth": {
                "token": "${SECRET_TOKEN}",
                "type": "bearer",
            },
            "timeout": 60,
        }
        resolved = _resolve_deferred_env_vars(params)
        assert resolved["auth"]["token"] == "tok123"
        assert resolved["auth"]["type"] == "bearer"
        assert resolved["timeout"] == 60

    def test_resolve_list_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolving env vars in list parameters."""
        monkeypatch.setenv("ENDPOINT_1", "http://api1.example.com")
        monkeypatch.setenv("ENDPOINT_2", "http://api2.example.com")
        params = {
            "endpoints": ["${ENDPOINT_1}", "${ENDPOINT_2}", "http://static.example.com"],
        }
        resolved = _resolve_deferred_env_vars(params)
        assert resolved["endpoints"][0] == "http://api1.example.com"
        assert resolved["endpoints"][1] == "http://api2.example.com"
        assert resolved["endpoints"][2] == "http://static.example.com"

    def test_resolve_empty_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that empty default is allowed."""
        monkeypatch.delenv("OPTIONAL_VAR", raising=False)
        result = _resolve_string_value("${OPTIONAL_VAR:}")
        assert result == ""

    def test_no_resolution_for_non_pattern_strings(self) -> None:
        """Test that strings without ${} are unchanged."""
        result = _resolve_string_value("regular string without vars")
        assert result == "regular string without vars"

    def test_instantiate_adapter_with_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that adapter instantiation resolves env vars."""
        monkeypatch.setenv("TEST_RESPONSE", "Hello from env!")
        instantiator = ComponentInstantiator()
        spec = {
            "adapter": "hexdag.stdlib.adapters.mock.MockLLM",
            "config": {"default_response": "${TEST_RESPONSE}"},
        }
        adapter = instantiator.instantiate_adapter(spec, port_name="llm")
        assert adapter is not None
        # The MockLLM should have received the resolved value

    def test_instantiate_adapter_missing_env_var_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that missing env var raises error during instantiation."""
        monkeypatch.delenv("REQUIRED_API_KEY", raising=False)
        instantiator = ComponentInstantiator()
        spec = {
            "adapter": "hexdag.stdlib.adapters.mock.MockLLM",
            "config": {"api_key": "${REQUIRED_API_KEY}"},
        }
        with pytest.raises(ComponentInstantiationError) as exc_info:
            instantiator.instantiate_adapter(spec, port_name="llm")
        assert "REQUIRED_API_KEY" in str(exc_info.value)
