"""Tests for centralized logging configuration using Loguru."""

import json
from pathlib import Path

from loguru import logger

from hexdag.kernel.logging import (
    configure_logging,
    enable_stdlib_logging_bridge,
    get_logger,
    get_logger_for_component,
)


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_returns_bound_logger(self):
        """Test that get_logger returns a bound logger instance."""
        test_logger = get_logger("test.module")
        # Loguru returns the same logger instance, just bound with context
        assert test_logger is not None

    def test_get_logger_with_name(self):
        """Test that __name__ style usage works."""
        test_logger = get_logger(__name__)
        assert test_logger is not None

    def test_get_logger_caches_results(self):
        """Test that get_logger caches logger instances."""
        logger1 = get_logger("test.cache")
        logger2 = get_logger("test.cache")
        # Should return same bound logger (cached)
        assert logger1 is logger2


class TestGetLoggerForComponent:
    """Test get_logger_for_component function."""

    def test_get_logger_for_component(self):
        """Test component-specific logger creation."""
        comp_logger = get_logger_for_component("adapter", "openai_llm")
        assert comp_logger is not None

    def test_logger_for_component_caches(self):
        """Test that component loggers are cached."""
        logger1 = get_logger_for_component("adapter", "test_adapter")
        logger2 = get_logger_for_component("adapter", "test_adapter")
        assert logger1 is logger2


class TestConfigureLogging:
    """Test configure_logging function."""

    def setup_method(self):
        """Reset logging configuration before each test."""
        # Loguru cleanup - remove all handlers
        logger.remove()

        # Reset the configuration tracking
        import hexdag.kernel.logging as logging_module

        logging_module._CURRENT_CONFIG = None

    def test_configures_json_format(self):
        """Test JSON format configuration."""
        configure_logging(level="INFO", format="json")

        # Loguru configured successfully (no exception)
        # We can verify by checking handlers exist
        assert len(logger._core.handlers) > 0

    def test_configures_structured_format(self):
        """Test structured format configuration."""
        configure_logging(level="INFO", format="structured")

        # Loguru configured successfully
        assert len(logger._core.handlers) > 0

    def test_configures_console_format(self):
        """Test console format configuration."""
        configure_logging(level="INFO", format="console")

        # Loguru configured successfully
        assert len(logger._core.handlers) > 0

    def test_configures_file_output(self, tmp_path: Path):
        """Test file output configuration."""
        log_file = tmp_path / "test.log"
        configure_logging(level="INFO", format="json", output_file=log_file)

        # Should have console + file handler
        assert len(logger._core.handlers) == 2

        # Write a log message
        test_logger = get_logger("test")
        test_logger.info("Test message")

        # File should exist after logging
        assert log_file.exists()

    def test_idempotent_reconfiguration(self):
        """Test that configure_logging is idempotent with same config."""
        configure_logging(level="DEBUG", format="console")
        handler_count_1 = len(logger._core.handlers)

        # Call again with same config - should not add handlers
        configure_logging(level="DEBUG", format="console")
        handler_count_2 = len(logger._core.handlers)

        assert handler_count_1 == handler_count_2

    def test_force_reconfigure(self):
        """Test force_reconfigure parameter."""
        configure_logging(level="INFO", format="console")

        # Force reconfigure with different settings
        configure_logging(level="DEBUG", format="structured", force_reconfigure=True)

        # Should have reconfigured
        assert len(logger._core.handlers) > 0

    def test_multiple_formats(self):
        """Test switching between formats."""
        # JSON format
        configure_logging(level="INFO", format="json", force_reconfigure=True)
        assert len(logger._core.handlers) > 0

        # Structured format
        configure_logging(level="INFO", format="structured", force_reconfigure=True)
        assert len(logger._core.handlers) > 0

        # Console format
        configure_logging(level="INFO", format="console", force_reconfigure=True)
        assert len(logger._core.handlers) > 0


class TestStdlibLoggingBridge:
    """Test stdlib logging bridge functionality."""

    def test_enable_stdlib_logging_bridge(self):
        """Test that stdlib logging bridge can be enabled."""
        # Should not raise
        enable_stdlib_logging_bridge()

        # Test that stdlib logging works
        import logging as stdlib_logging

        stdlib_logger = stdlib_logging.getLogger("test_bridge")
        # This should work without error
        stdlib_logger.info("Test message through bridge")


class TestLoggingHandlerIsolation:
    """Test that logging handler cleanup is isolated and safe."""

    def test_handler_cleanup_removes_all_handlers(self):
        """Test that reconfiguring logging removes ALL handlers (including default).

        This prevents duplicate log output from Loguru's default handler (ID=0)
        co-existing with hexDAG's custom handler.
        """
        from loguru import logger as loguru_logger

        # Add an external handler
        external_messages = []

        def external_sink(message):
            external_messages.append(message)

        loguru_logger.add(external_sink, format="{message}")

        # Configure logging — should remove ALL handlers (including external)
        # and add only its own. This is intentional to prevent duplicate output.
        configure_logging(level="INFO", format="console", force_reconfigure=True)

        test_logger = get_logger("test")
        test_logger.info("Test message")

        # External handler should have been removed by configure_logging()
        assert len(external_messages) == 0

    def test_multiple_reconfigurations_dont_leak_handlers(self):
        """Test that multiple reconfigurations don't accumulate handlers."""
        from loguru import logger as loguru_logger

        # Get initial handler count
        initial_handlers = len(loguru_logger._core.handlers)

        # Configure multiple times
        for _ in range(5):
            configure_logging(level="INFO", format="console", force_reconfigure=True)

        # Handler count should not grow unbounded (should only have our handlers + initial)
        final_handlers = len(loguru_logger._core.handlers)

        # We should have at most initial + 1 (our console handler)
        # Allow some tolerance for test framework handlers
        assert final_handlers <= initial_handlers + 2, (
            f"Handler leak detected: started with {initial_handlers}, ended with {final_handlers}"
        )


class TestConfigureLoggingExactHandlerCount:
    """Test that configure_logging produces exactly the expected number of handlers."""

    def setup_method(self):
        """Reset logging configuration before each test."""
        logger.remove()
        import hexdag.kernel.logging as logging_module

        logging_module._CURRENT_CONFIG = None

    def test_console_format_produces_one_handler(self):
        """Console format should produce exactly 1 handler."""
        configure_logging(level="INFO", format="console")
        assert len(logger._core.handlers) == 1

    def test_json_format_produces_one_handler(self):
        """JSON format should produce exactly 1 handler."""
        configure_logging(level="INFO", format="json")
        assert len(logger._core.handlers) == 1

    def test_structured_format_produces_one_handler(self):
        """Structured format should produce exactly 1 handler."""
        configure_logging(level="INFO", format="structured")
        assert len(logger._core.handlers) == 1

    def test_file_output_produces_two_handlers(self, tmp_path: Path):
        """Console + file should produce exactly 2 handlers."""
        configure_logging(level="INFO", format="console", output_file=tmp_path / "test.log")
        assert len(logger._core.handlers) == 2

    def test_reconfigure_removes_previous_handlers(self):
        """Reconfiguring should not accumulate handlers."""
        configure_logging(level="INFO", format="console")
        assert len(logger._core.handlers) == 1

        configure_logging(level="DEBUG", format="json", force_reconfigure=True)
        assert len(logger._core.handlers) == 1


class TestLogFileJsonOutput:
    """Test that log file output is valid JSON-lines format."""

    def setup_method(self):
        logger.remove()
        import hexdag.kernel.logging as logging_module

        logging_module._CURRENT_CONFIG = None

    def test_file_output_is_json_lines(self, tmp_path: Path):
        """Log file output should be parseable JSON-lines."""
        log_file = tmp_path / "test.log"
        configure_logging(level="DEBUG", format="console", output_file=log_file)

        test_logger = get_logger("test.json_output")
        test_logger.info("First message")
        test_logger.warning("Second message")

        # Force flush by removing handlers
        logger.remove()

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2

        for line in lines:
            record = json.loads(line)
            assert "record" in record
            assert "text" in record
            assert "message" in record["record"]

    def test_file_output_contains_bound_context(self, tmp_path: Path):
        """Log file records should contain bound context (module name)."""
        log_file = tmp_path / "test.log"
        configure_logging(level="DEBUG", format="console", output_file=log_file)

        test_logger = get_logger("test.context_check")
        test_logger.info("Context test")

        logger.remove()

        line = log_file.read_text().strip().split("\n")[0]
        record = json.loads(line)
        assert record["record"]["extra"]["module"] == "test.context_check"
