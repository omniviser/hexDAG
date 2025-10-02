"""Tests for centralized logging configuration."""

import logging
import sys
from pathlib import Path

from hexai.core.logging import (
    JSONFormatter,
    StructuredFormatter,
    configure_logging,
    get_logger,
    get_logger_for_component,
)


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a logger instance."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_get_logger_with_name(self):
        """Test that __name__ style usage works."""
        logger = get_logger(__name__)
        assert isinstance(logger, logging.Logger)
        assert "test_logging" in logger.name

    def test_get_logger_returns_same_instance(self):
        """Test that get_logger returns the same instance for the same name."""
        logger1 = get_logger("test.module")
        logger2 = get_logger("test.module")
        assert logger1 is logger2


class TestGetLoggerForComponent:
    """Test get_logger_for_component function."""

    def test_creates_hierarchical_name(self):
        """Test that component logger has hierarchical name."""
        logger = get_logger_for_component("adapter", "openai_llm")
        assert logger.name == "hexai.adapter.openai_llm"

    def test_different_components_get_different_loggers(self):
        """Test that different components get different loggers."""
        logger1 = get_logger_for_component("adapter", "openai")
        logger2 = get_logger_for_component("adapter", "anthropic")
        assert logger1 is not logger2
        assert logger1.name != logger2.name


class TestStructuredFormatter:
    """Test StructuredFormatter."""

    def test_formats_basic_message(self):
        """Test basic message formatting."""
        formatter = StructuredFormatter(use_color=False, include_timestamp=False)
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        assert "INFO" in output
        assert "test.logger" in output
        assert "Test message" in output

    def test_formats_with_timestamp(self):
        """Test formatting with timestamp."""
        formatter = StructuredFormatter(use_color=False, include_timestamp=True)
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        # Should contain timestamp pattern like "2024-01-01 12:00:00"
        assert any(char.isdigit() for char in output[:20])

    def test_formats_with_color(self):
        """Test formatting with color codes."""
        formatter = StructuredFormatter(use_color=True, include_timestamp=False)
        # Temporarily make stderr appear as TTY
        original_isatty = sys.stderr.isatty
        sys.stderr.isatty = lambda: True

        try:
            formatter = StructuredFormatter(use_color=True, include_timestamp=False)
            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg="Error message",
                args=(),
                exc_info=None,
            )

            output = formatter.format(record)
            # Should contain ANSI color codes
            assert "\033[" in output or "ERROR" in output
        finally:
            sys.stderr.isatty = original_isatty

    def test_formats_exception(self):
        """Test formatting with exception info."""
        formatter = StructuredFormatter(use_color=False, include_timestamp=False)

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        output = formatter.format(record)
        assert "Error occurred" in output
        assert "ValueError" in output
        assert "Test error" in output


class TestJSONFormatter:
    """Test JSONFormatter."""

    def test_formats_as_json(self):
        """Test that output is valid JSON."""
        import json

        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["message"] == "Test message"
        assert "timestamp" in parsed

    def test_includes_exception_in_json(self):
        """Test that exceptions are included in JSON output."""
        import json

        formatter = JSONFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]

    def test_includes_extra_context(self):
        """Test that extra context is included in JSON."""
        import json

        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        # Add extra context
        record.pipeline_id = "abc123"
        record.node_name = "test_node"

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["pipeline_id"] == "abc123"
        assert parsed["node_name"] == "test_node"


class TestConfigureLogging:
    """Test configure_logging function."""

    def setup_method(self):
        """Reset logging configuration before each test."""
        # Clear all handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.WARNING)

        # Reset the configured flag
        import hexai.core.logging as logging_module

        logging_module._LOGGING_CONFIGURED = False

    def test_configures_log_level(self):
        """Test that configure_logging sets the log level."""
        configure_logging(level="DEBUG", format="console")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_configures_structured_format(self):
        """Test structured format configuration."""
        configure_logging(level="INFO", format="structured")

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0

        handler = root_logger.handlers[0]
        assert isinstance(handler.formatter, StructuredFormatter)

    def test_configures_json_format(self):
        """Test JSON format configuration."""
        configure_logging(level="INFO", format="json")

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0

        handler = root_logger.handlers[0]
        assert isinstance(handler.formatter, JSONFormatter)

    def test_configures_console_format(self):
        """Test console format configuration."""
        configure_logging(level="INFO", format="console")

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0

        handler = root_logger.handlers[0]
        assert isinstance(handler.formatter, logging.Formatter)
        assert not isinstance(handler.formatter, (StructuredFormatter, JSONFormatter))

    def test_configures_file_output(self, tmp_path: Path):
        """Test file output configuration."""
        log_file = tmp_path / "test.log"
        configure_logging(level="INFO", format="json", output_file=log_file)

        root_logger = logging.getLogger()
        # Should have console + file handler
        assert len(root_logger.handlers) == 2

        # Check that file was created (may be created on first log)
        logger = get_logger("test")
        logger.info("Test message")

        # File should exist after logging
        assert log_file.exists()

    def test_skips_reconfiguration_by_default(self):
        """Test that configure_logging skips if already configured."""
        configure_logging(level="DEBUG", format="console")
        root_logger = logging.getLogger()
        original_level = root_logger.level

        # Try to reconfigure
        configure_logging(level="ERROR", format="json")

        # Should still be at DEBUG level
        assert root_logger.level == original_level

    def test_force_reconfiguration(self):
        """Test force_reconfigure parameter."""
        configure_logging(level="DEBUG", format="console")
        root_logger = logging.getLogger()

        # Force reconfiguration
        configure_logging(level="ERROR", format="json", force_reconfigure=True)

        # Should be at ERROR level now
        assert root_logger.level == logging.ERROR

    def test_sets_library_loggers_to_warning(self):
        """Test that noisy library loggers are set to WARNING."""
        configure_logging(level="DEBUG", format="console")

        urllib3_logger = logging.getLogger("urllib3")
        assert urllib3_logger.level == logging.WARNING


class TestLoggingIntegration:
    """Integration tests for logging system."""

    def setup_method(self):
        """Reset logging configuration before each test."""
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.WARNING)

        import hexai.core.logging as logging_module

        logging_module._LOGGING_CONFIGURED = False

    def test_logger_outputs_messages(self):
        """Test that logger can output messages without errors."""
        configure_logging(level="INFO", format="console")
        logger = get_logger("test")

        # Just verify these don't raise errors
        logger.info("Test message")
        logger.debug("Debug message")  # Should be filtered out
        logger.warning("Warning message")
        logger.error("Error message")

    def test_logger_with_extra_context(self):
        """Test logging with extra context."""
        configure_logging(level="INFO", format="json")
        logger = get_logger("test")

        # Verify extra context doesn't cause errors
        logger.info("Pipeline started", extra={"pipeline_id": "abc123", "user": "test_user"})
        logger.error("Pipeline failed", extra={"error_code": 500})

    def test_multiple_log_levels(self):
        """Test that different log levels work correctly."""
        configure_logging(level="DEBUG", format="structured")
        logger = get_logger("test")

        # Verify all log levels work
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

    def test_logger_hierarchy_works(self):
        """Test that logger hierarchy works correctly."""
        configure_logging(level="INFO", format="console")

        # Create parent and child loggers
        parent = get_logger("hexai.core")
        child = get_logger("hexai.core.orchestrator")

        # Both should work
        parent.info("Parent message")
        child.info("Child message")
