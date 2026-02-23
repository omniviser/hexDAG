"""Tests for the config models module.

This module tests the configuration data models for HexDAG.
"""

from __future__ import annotations

import pytest

from hexdag.kernel.config.models import (
    DefaultCaps,
    DefaultLimits,
    HexDAGConfig,
    LoggingConfig,
    ManifestEntry,
)
from hexdag.kernel.exceptions import ValidationError
from hexdag.kernel.orchestration.models import OrchestratorConfig


class TestLoggingConfig:
    """Tests for LoggingConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values for LoggingConfig."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.format == "structured"
        assert config.output_file is None
        assert config.use_color is True
        assert config.include_timestamp is True
        assert config.use_rich is False
        assert config.dual_sink is False
        assert config.enable_stdlib_bridge is False
        assert config.backtrace is True
        assert config.diagnose is True

    def test_custom_values(self) -> None:
        """Test creating LoggingConfig with custom values."""
        config = LoggingConfig(
            level="DEBUG",
            format="json",
            output_file="/var/log/test.log",
            use_color=False,
            include_timestamp=False,
            use_rich=True,
            dual_sink=True,
            enable_stdlib_bridge=True,
            backtrace=False,
            diagnose=False,
        )
        assert config.level == "DEBUG"
        assert config.format == "json"
        assert config.output_file == "/var/log/test.log"
        assert config.use_color is False
        assert config.include_timestamp is False
        assert config.use_rich is True
        assert config.dual_sink is True
        assert config.enable_stdlib_bridge is True
        assert config.backtrace is False
        assert config.diagnose is False

    def test_frozen_immutability(self) -> None:
        """Test that LoggingConfig is frozen (immutable)."""
        config = LoggingConfig()
        with pytest.raises(AttributeError):
            config.level = "DEBUG"  # type: ignore[misc]

    def test_all_log_levels(self) -> None:
        """Test all valid log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = LoggingConfig(level=level)  # type: ignore[arg-type]
            assert config.level == level

    def test_all_format_types(self) -> None:
        """Test all valid format types."""
        for fmt in ["console", "json", "structured", "dual", "rich"]:
            config = LoggingConfig(format=fmt)  # type: ignore[arg-type]
            assert config.format == fmt


class TestManifestEntry:
    """Tests for ManifestEntry dataclass."""

    def test_valid_entry(self) -> None:
        """Test creating a valid manifest entry."""
        entry = ManifestEntry(namespace="core", module="hexdag.stdlib.nodes")
        assert entry.namespace == "core"
        assert entry.module == "hexdag.stdlib.nodes"

    def test_empty_namespace_raises_error(self) -> None:
        """Test that empty namespace raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ManifestEntry(namespace="", module="hexdag.stdlib.nodes")
        assert "namespace" in str(exc_info.value)
        assert "cannot be empty" in str(exc_info.value)

    def test_empty_module_raises_error(self) -> None:
        """Test that empty module raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ManifestEntry(namespace="core", module="")
        assert "module" in str(exc_info.value)
        assert "cannot be empty" in str(exc_info.value)

    def test_namespace_with_colon_raises_error(self) -> None:
        """Test that namespace containing ':' raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ManifestEntry(namespace="core:test", module="hexdag.stdlib.nodes")
        assert "namespace" in str(exc_info.value)
        assert "cannot contain ':'" in str(exc_info.value)

    def test_frozen_immutability(self) -> None:
        """Test that ManifestEntry is frozen (immutable)."""
        entry = ManifestEntry(namespace="core", module="hexdag.stdlib.nodes")
        with pytest.raises(AttributeError):
            entry.namespace = "user"  # type: ignore[misc]

    def test_various_namespaces(self) -> None:
        """Test various valid namespace names."""
        namespaces = ["core", "user", "plugin", "custom", "my_namespace"]
        for ns in namespaces:
            entry = ManifestEntry(namespace=ns, module="some.module")
            assert entry.namespace == ns


class TestHexDAGConfig:
    """Tests for HexDAGConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values for HexDAGConfig."""
        config = HexDAGConfig()
        assert config.modules == []
        assert config.plugins == []
        assert config.dev_mode is False
        assert isinstance(config.logging, LoggingConfig)
        assert config.settings == {}

    def test_custom_modules(self) -> None:
        """Test setting custom modules."""
        modules = ["myapp.adapters", "myapp.nodes"]
        config = HexDAGConfig(modules=modules)
        assert config.modules == modules

    def test_custom_plugins(self) -> None:
        """Test setting custom plugins."""
        plugins = ["hexdag-openai", "hexdag-postgres"]
        config = HexDAGConfig(plugins=plugins)
        assert config.plugins == plugins

    def test_dev_mode(self) -> None:
        """Test dev_mode setting."""
        config = HexDAGConfig(dev_mode=True)
        assert config.dev_mode is True

    def test_custom_logging(self) -> None:
        """Test custom logging configuration."""
        logging_config = LoggingConfig(level="DEBUG", format="json")
        config = HexDAGConfig(logging=logging_config)
        assert config.logging.level == "DEBUG"
        assert config.logging.format == "json"

    def test_custom_settings(self) -> None:
        """Test custom settings dict."""
        settings = {"log_level": "DEBUG", "enable_metrics": True, "max_retries": 5}
        config = HexDAGConfig(settings=settings)
        assert config.settings == settings

    def test_full_configuration(self) -> None:
        """Test creating a fully configured HexDAGConfig."""
        config = HexDAGConfig(
            modules=["myapp.adapters", "myapp.nodes"],
            plugins=["hexdag-openai"],
            dev_mode=True,
            logging=LoggingConfig(level="DEBUG"),
            settings={"custom_key": "custom_value"},
        )
        assert len(config.modules) == 2
        assert len(config.plugins) == 1
        assert config.dev_mode is True
        assert config.logging.level == "DEBUG"
        assert config.settings["custom_key"] == "custom_value"

    def test_mutability(self) -> None:
        """Test that HexDAGConfig is mutable (not frozen)."""
        config = HexDAGConfig()
        config.modules = ["new.module"]
        assert config.modules == ["new.module"]

        config.dev_mode = True
        assert config.dev_mode is True

    def test_independent_default_factories(self) -> None:
        """Test that default factories create independent instances."""
        config1 = HexDAGConfig()
        config2 = HexDAGConfig()

        config1.modules.append("test.module")
        assert "test.module" not in config2.modules

        config1.settings["key"] = "value"
        assert "key" not in config2.settings

    def test_default_orchestrator(self) -> None:
        """Test default orchestrator config."""
        config = HexDAGConfig()
        assert isinstance(config.orchestrator, OrchestratorConfig)
        assert config.orchestrator.max_concurrent_nodes == 10
        assert config.orchestrator.strict_validation is False
        assert config.orchestrator.default_node_timeout is None

    def test_custom_orchestrator(self) -> None:
        """Test custom orchestrator config."""
        orch = OrchestratorConfig(max_concurrent_nodes=5, default_node_timeout=60.0)
        config = HexDAGConfig(orchestrator=orch)
        assert config.orchestrator.max_concurrent_nodes == 5
        assert config.orchestrator.default_node_timeout == 60.0

    def test_default_limits(self) -> None:
        """Test default limits config."""
        config = HexDAGConfig()
        assert isinstance(config.limits, DefaultLimits)
        assert config.limits.max_total_tokens is None
        assert config.limits.max_llm_calls is None
        assert config.limits.max_tool_calls is None
        assert config.limits.max_cost_usd is None
        assert config.limits.warning_threshold == 0.8

    def test_custom_limits(self) -> None:
        """Test custom limits config."""
        limits = DefaultLimits(max_llm_calls=100, max_cost_usd=10.0)
        config = HexDAGConfig(limits=limits)
        assert config.limits.max_llm_calls == 100
        assert config.limits.max_cost_usd == 10.0

    def test_default_caps(self) -> None:
        """Test default caps config."""
        config = HexDAGConfig()
        assert isinstance(config.caps, DefaultCaps)
        assert config.caps.default_set is None
        assert config.caps.deny is None

    def test_custom_caps(self) -> None:
        """Test custom caps config."""
        caps = DefaultCaps(default_set=["llm", "memory"], deny=["secret"])
        config = HexDAGConfig(caps=caps)
        assert config.caps.default_set == ["llm", "memory"]
        assert config.caps.deny == ["secret"]


class TestDefaultLimits:
    """Tests for DefaultLimits dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        limits = DefaultLimits()
        assert limits.max_total_tokens is None
        assert limits.max_llm_calls is None
        assert limits.max_tool_calls is None
        assert limits.max_cost_usd is None
        assert limits.warning_threshold == 0.8

    def test_custom_values(self) -> None:
        """Test setting custom values."""
        limits = DefaultLimits(
            max_total_tokens=100000,
            max_llm_calls=50,
            max_tool_calls=200,
            max_cost_usd=5.0,
            warning_threshold=0.9,
        )
        assert limits.max_total_tokens == 100000
        assert limits.max_llm_calls == 50
        assert limits.max_tool_calls == 200
        assert limits.max_cost_usd == 5.0
        assert limits.warning_threshold == 0.9

    def test_frozen_immutability(self) -> None:
        """Test that DefaultLimits is frozen."""
        limits = DefaultLimits()
        with pytest.raises(AttributeError):
            limits.max_llm_calls = 10  # type: ignore[misc]

    def test_partial_limits(self) -> None:
        """Test setting only some limits."""
        limits = DefaultLimits(max_llm_calls=100)
        assert limits.max_llm_calls == 100
        assert limits.max_cost_usd is None
        assert limits.max_tool_calls is None


class TestDefaultCaps:
    """Tests for DefaultCaps dataclass."""

    def test_default_values(self) -> None:
        """Test default values (unrestricted)."""
        caps = DefaultCaps()
        assert caps.default_set is None
        assert caps.deny is None

    def test_custom_default_set(self) -> None:
        """Test custom default set."""
        caps = DefaultCaps(default_set=["llm", "memory", "datastore.read"])
        assert caps.default_set == ["llm", "memory", "datastore.read"]

    def test_custom_deny(self) -> None:
        """Test custom deny list."""
        caps = DefaultCaps(deny=["secret", "spawner"])
        assert caps.deny == ["secret", "spawner"]

    def test_frozen_immutability(self) -> None:
        """Test that DefaultCaps is frozen."""
        caps = DefaultCaps()
        with pytest.raises(AttributeError):
            caps.deny = ["secret"]  # type: ignore[misc]

    def test_full_caps(self) -> None:
        """Test full caps configuration."""
        caps = DefaultCaps(
            default_set=["llm", "memory"],
            deny=["secret"],
        )
        assert caps.default_set == ["llm", "memory"]
        assert caps.deny == ["secret"]
