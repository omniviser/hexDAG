"""Example: Centralized Logging Configuration

Demonstrates how to configure and use the centralized logging system in hexDAG.

The logging system provides:
- Multiple output formats (console, JSON, structured)
- Environment-based configuration
- Structured context logging
- Easy integration with observability systems
"""

import tempfile

from hexdag.core.logging import configure_logging, get_logger, get_logger_for_component


def example_basic_logging():
    """Basic logging usage."""
    print("\n=== Basic Logging ===")

    # Configure logging for development
    configure_logging(level="DEBUG", format="structured", use_color=True)

    # Get a logger for your module
    logger = get_logger(__name__)

    # Log at different levels
    logger.debug("This is a debug message")
    logger.info("Pipeline started successfully")
    logger.warning("Resource usage is high")
    logger.error("Failed to connect to database")


def example_json_logging():
    """JSON logging for production."""
    print("\n\n=== JSON Logging for Production ===")

    # Use temporary file for demo (in production, use proper log directory)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as tmp_log:
        log_file = tmp_log.name

    # Configure for production with JSON format
    configure_logging(
        level="INFO",
        format="json",
        output_file=log_file,  # Also write to file
        force_reconfigure=True,
    )

    logger = get_logger(__name__)

    # JSON format is ideal for log aggregation systems
    logger.info("Production pipeline started")
    logger.error("Production error occurred")

    print(f"Logs written to: {log_file}")


def example_structured_context():
    """Logging with structured context."""
    print("\n\n=== Structured Context Logging ===")

    configure_logging(level="INFO", format="structured", force_reconfigure=True)

    logger = get_logger(__name__)

    # Add context via 'extra' parameter
    logger.info(
        "Processing pipeline",
        extra={
            "pipeline_id": "abc-123",
            "user_id": "user-456",
            "environment": "production",
        },
    )

    logger.error(
        "Pipeline failed",
        extra={
            "pipeline_id": "abc-123",
            "error_code": "DB_CONNECTION_FAILED",
            "retry_count": 3,
        },
    )


def example_component_logger():
    """Component-specific loggers."""
    print("\n\n=== Component-Specific Loggers ===")

    configure_logging(level="INFO", format="structured", force_reconfigure=True)

    # Get loggers for specific components
    llm_logger = get_logger_for_component("adapter", "openai_llm")
    db_logger = get_logger_for_component("adapter", "postgres_db")
    node_logger = get_logger_for_component("node", "data_processor")

    # Each component gets its own hierarchical logger name
    llm_logger.info("LLM request sent")
    db_logger.info("Database query executed")
    node_logger.info("Node processing completed")


def example_console_logging():
    """Simple console logging."""
    print("\n\n=== Simple Console Logging ===")

    # Simple console format (like standard Python logging)
    configure_logging(level="INFO", format="console", force_reconfigure=True)

    logger = get_logger(__name__)

    logger.info("Simple console output")
    logger.warning("Simple warning message")


def example_with_exceptions():
    """Logging with exception information."""
    print("\n\n=== Exception Logging ===")

    configure_logging(level="ERROR", format="structured", force_reconfigure=True)

    logger = get_logger(__name__)

    try:
        # Simulate an error
        pass
    except ZeroDivisionError:
        # Log with exception info automatically included
        logger.error("Division by zero occurred", exc_info=True)


def main():
    """Run all logging examples."""
    print("=" * 60)
    print("hexDAG Centralized Logging Examples")
    print("=" * 60)

    # Run examples
    example_basic_logging()
    example_json_logging()
    example_structured_context()
    example_component_logger()
    example_console_logging()
    example_with_exceptions()

    print("\n\n" + "=" * 60)
    print("Logging Configuration Guide:")
    print("=" * 60)
    print(
        """
1. Development Setup:
   configure_logging(level="DEBUG", format="structured", use_color=True)

2. Production Setup:
   configure_logging(
       level="INFO",
       format="json",
       output_file="/var/log/hexdag/app.log"
   )

3. Testing Setup:
   configure_logging(level="WARNING", format="console")

4. Usage in Code:
   from hexdag.core.logging import get_logger
   logger = get_logger(__name__)
   logger.info("Message", extra={"context": "value"})
"""
    )


if __name__ == "__main__":
    main()
