"""Example 20: Integration Testing

This example demonstrates comprehensive integration testing patterns
for hexAI pipelines and components.
"""

import asyncio
import time
from typing import Any, Dict, List

from hexai.core.application.orchestrator import Orchestrator
from hexai.core.domain.dag import DirectedGraph, NodeSpec
from hexai.core.validation import coerce_validator


class MockEmailAdapter:
    """Mock email adapter for testing email functionality."""

    def __init__(self):
        """Initialize the mock email adapter."""
        self.sent_emails = []
        self.fail_rate = 0.0  # 0.0 = never fail, 1.0 = always fail

    async def send_email(
        self,
        to_address: str,
        subject: str,
        body: str,
        from_address: str = "noreply@example.com",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Send a mock email."""
        await asyncio.sleep(0.1)  # Simulate network delay

        # Simulate failure based on fail_rate
        if asyncio.get_event_loop().time() % 10 < self.fail_rate * 10:
            raise Exception("Mock email service temporarily unavailable")

        email_data = {
            "to": to_address,
            "from": from_address,
            "subject": subject,
            "body": body,
            "timestamp": asyncio.get_event_loop().time(),
            "status": "sent",
        }

        self.sent_emails.append(email_data)

        return {
            "message_id": f"mock_{len(self.sent_emails)}",
            "status": "sent",
            "to": to_address,
            "subject": subject,
        }

    async def send_bulk_email(
        self,
        recipients: List[str],
        subject: str,
        body: str,
        from_address: str = "noreply@example.com",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Send bulk mock emails."""
        await asyncio.sleep(0.2)  # Simulate bulk processing delay

        results = []
        for recipient in recipients:
            try:
                result = await self.send_email(recipient, subject, body, from_address)
                results.append(result)
            except Exception as e:
                results.append({"to": recipient, "status": "failed", "error": str(e)})

        return {
            "total_sent": len([r for r in results if r["status"] == "sent"]),
            "total_failed": len([r for r in results if r["status"] == "failed"]),
            "results": results,
        }

    def get_sent_emails(self) -> List[Dict[str, Any]]:
        """Get all sent emails for testing."""
        return self.sent_emails.copy()

    def clear_sent_emails(self) -> None:
        """Clear the sent emails list."""
        self.sent_emails.clear()

    def set_fail_rate(self, rate: float) -> None:
        """Set the failure rate for testing (0.0 to 1.0)."""
        self.fail_rate = max(0.0, min(1.0, rate))


class MockDatabaseAdapter:
    """Mock database adapter for testing database operations."""

    def __init__(self):
        """Initialize the mock database adapter."""
        self.data = {}
        self.fail_rate = 0.0

    async def save_data(self, key: str, value: Any) -> Dict[str, Any]:
        """Save data to mock database."""
        await asyncio.sleep(0.05)  # Simulate database write

        if asyncio.get_event_loop().time() % 10 < self.fail_rate * 10:
            raise Exception("Mock database temporarily unavailable")

        self.data[key] = value
        return {"status": "saved", "key": key}

    async def get_data(self, key: str) -> Dict[str, Any]:
        """Get data from mock database."""
        await asyncio.sleep(0.02)  # Simulate database read

        if asyncio.get_event_loop().time() % 10 < self.fail_rate * 10:
            raise Exception("Mock database temporarily unavailable")

        return {"status": "found", "data": self.data.get(key)}

    def set_fail_rate(self, rate: float) -> None:
        """Set the failure rate for testing."""
        self.fail_rate = max(0.0, min(1.0, rate))


class MockLoggerAdapter:
    """Mock logger adapter for testing logging functionality."""

    def __init__(self):
        """Initialize the mock logger adapter."""
        self.logs = []

    async def log_info(self, message: str, **kwargs: Any) -> Dict[str, Any]:
        """Log info message."""
        await asyncio.sleep(0.01)  # Simulate logging delay
        self.logs.append({"level": "info", "message": message, **kwargs})
        return {"status": "logged", "level": "info"}

    async def log_error(self, message: str, **kwargs: Any) -> Dict[str, Any]:
        """Log error message."""
        await asyncio.sleep(0.01)  # Simulate logging delay
        self.logs.append({"level": "error", "message": message, **kwargs})
        return {"status": "logged", "level": "error"}

    def get_logs(self) -> List[Dict[str, Any]]:
        """Get all logs for testing."""
        return self.logs.copy()

    def clear_logs(self) -> None:
        """Clear the logs list."""
        self.logs.clear()


async def test_basic_integration():
    """Test basic integration with mock services."""

    print("\n🧪 Basic Integration Test")
    print("=" * 30)

    # Initialize mock services
    email_service = MockEmailAdapter()
    database_service = MockDatabaseAdapter()
    logger_service = MockLoggerAdapter()

    # Create test functions
    async def data_processor(input_data: Any, **kwargs) -> dict:
        """Process data and save to database."""
        db = kwargs.get("database")
        logger = kwargs.get("logger")

        # Process data
        processed_data = {"original": input_data, "processed": True, "timestamp": time.time()}

        # Save to database
        await db.save_data("processed_data", processed_data)

        # Log the operation
        await logger.log_info("Data processed successfully", data_id="test_001")

        return processed_data

    async def notification_sender(input_data: Any, **kwargs) -> dict:
        """Send notification email."""
        email = kwargs.get("email")
        logger = kwargs.get("logger")

        # Send email
        email_result = await email.send_email(
            to_address="test@example.com",
            subject="Data Processing Complete",
            body=f"Data processing completed at {time.time()}",
        )

        # Log the notification
        await logger.log_info("Notification sent", email_id=email_result["message_id"])

        return {"notification_sent": True, "email_result": email_result}

    # Create workflow
    graph = DirectedGraph()
    graph.add(NodeSpec("data_processor", data_processor))
    graph.add(NodeSpec("notification_sender", notification_sender).after("data_processor"))

    # Set up orchestrator with mock services
    orchestrator = Orchestrator(validator=coerce_validator())

    # Execute with mock services
    ports = {"email": email_service, "database": database_service, "logger": logger_service}

    start_time = time.time()
    result = await orchestrator.run(graph, {"test": "data"}, additional_ports=ports)
    end_time = time.time()

    # Verify results
    print(f"   ⏱️  Execution time: {end_time - start_time:.3f}s")
    print(f"   📧 Emails sent: {len(email_service.get_sent_emails())}")
    print(f"   💾 Database operations: {len(database_service.data)}")
    print(f"   📝 Log entries: {len(logger_service.get_logs())}")

    return result


async def test_error_handling():
    """Test error handling in integration scenarios."""

    print("\n🛡️ Error Handling Integration Test")
    print("=" * 40)

    # Initialize services with high failure rate
    email_service = MockEmailAdapter()
    email_service.set_fail_rate(0.8)  # 80% failure rate

    database_service = MockDatabaseAdapter()
    database_service.set_fail_rate(0.5)  # 50% failure rate

    logger_service = MockLoggerAdapter()

    async def unreliable_processor(input_data: Any, **kwargs) -> dict:
        """Process data with potential failures."""
        db = kwargs.get("database")
        logger = kwargs.get("logger")

        try:
            # Try to save data
            await db.save_data("test_key", input_data)
            await logger.log_info("Data saved successfully")

            return {"status": "success", "data": input_data}

        except Exception as e:
            # Log the error
            await logger.log_error(f"Failed to save data: {str(e)}")

            # Return fallback result
            return {"status": "fallback", "error": str(e)}

    async def notification_with_retry(input_data: Any, **kwargs) -> dict:
        """Send notification with retry logic."""
        email = kwargs.get("email")
        logger = kwargs.get("logger")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                await email.send_email(
                    to_address="test@example.com",
                    subject="Processing Update",
                    body=f"Attempt {attempt + 1}: {input_data}",
                )
                await logger.log_info("Email sent successfully", attempt=attempt + 1)
                return {"status": "sent", "attempts": attempt + 1}

            except Exception as e:
                await logger.log_error(f"Email attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    return {"status": "failed", "error": str(e), "attempts": max_retries}
                await asyncio.sleep(0.1)  # Brief delay before retry

    # Create workflow
    graph = DirectedGraph()
    graph.add(NodeSpec("unreliable_processor", unreliable_processor))
    graph.add(
        NodeSpec("notification_with_retry", notification_with_retry).after("unreliable_processor")
    )

    # Execute
    orchestrator = Orchestrator(validator=coerce_validator())
    ports = {"email": email_service, "database": database_service, "logger": logger_service}

    result = await orchestrator.run(graph, {"test": "error_handling"}, additional_ports=ports)

    # Analyze results
    logs = logger_service.get_logs()
    error_logs = [log for log in logs if log["level"] == "error"]
    info_logs = [log for log in logs if log["level"] == "info"]

    print(f"   📊 Total operations: {len(logs)}")
    print(f"   ✅ Successful operations: {len(info_logs)}")
    print(f"   ❌ Failed operations: {len(error_logs)}")
    print(f"   📧 Email attempts: {result.get('notification_with_retry', {}).get('attempts', 0)}")

    return result


async def test_performance_monitoring():
    """Test performance monitoring in integration scenarios."""

    print("\n📊 Performance Monitoring Integration Test")
    print("=" * 45)

    email_service = MockEmailAdapter()
    database_service = MockDatabaseAdapter()
    logger_service = MockLoggerAdapter()

    async def performance_monitored_operation(input_data: Any, **kwargs) -> dict:
        """Operation with performance monitoring."""
        logger = kwargs.get("logger")

        start_time = time.time()

        # Simulate work
        await asyncio.sleep(0.2)

        # Save to database
        db = kwargs.get("database")
        await db.save_data("performance_data", {"timestamp": start_time})

        # Send notification
        email = kwargs.get("email")
        await email.send_email(
            to_address="monitoring@example.com",
            subject="Performance Report",
            body=f"Operation completed in {time.time() - start_time:.3f}s",
        )

        # Log performance metrics
        await logger.log_info(
            "Performance monitored operation completed",
            duration=time.time() - start_time,
            data_size=len(str(input_data)),
        )

        return {
            "performance_metrics": {
                "duration": time.time() - start_time,
                "data_size": len(str(input_data)),
                "timestamp": start_time,
            }
        }

    # Create workflow with multiple operations
    graph = DirectedGraph()
    graph.add(NodeSpec("op1", performance_monitored_operation))
    graph.add(NodeSpec("op2", performance_monitored_operation).after("op1"))
    graph.add(NodeSpec("op3", performance_monitored_operation).after("op2"))

    # Execute
    orchestrator = Orchestrator(validator=coerce_validator())
    ports = {"email": email_service, "database": database_service, "logger": logger_service}

    start_time = time.time()
    result = await orchestrator.run(graph, {"test": "performance"}, additional_ports=ports)
    total_time = time.time() - start_time

    # Analyze performance
    logs = logger_service.get_logs()
    performance_logs = [log for log in logs if "duration" in log]

    total_duration = sum(log.get("duration", 0) for log in performance_logs)
    avg_duration = total_duration / len(performance_logs) if performance_logs else 0

    print(f"   ⏱️  Total execution time: {total_time:.3f}s")
    print(f"   📊 Operations monitored: {len(performance_logs)}")
    print(f"   📈 Average operation time: {avg_duration:.3f}s")
    print(f"   📧 Performance notifications: {len(email_service.get_sent_emails())}")
    print(f"   💾 Performance data saved: {len(database_service.data)}")

    return result


async def main():
    """Run comprehensive integration tests."""

    print("🚀 Example 20: Integration Testing")
    print("=" * 35)

    print("\n🎯 This example demonstrates:")
    print("   • Integration testing patterns")
    print("   • Mock service implementations")
    print("   • Error handling in real scenarios")
    print("   • Performance monitoring - integration")
    print("   • Service orchestration testing")

    # Run integration tests
    await test_basic_integration()
    await test_error_handling()
    await test_performance_monitoring()

    print("\n🎯 Key Concepts Learned:")
    print("   ✅ Integration Testing - End-to-end workflow testing")
    print("   ✅ Mock Services - Simulating external dependencies")
    print("   ✅ Error Handling - Graceful degradation in real scenarios")
    print("   ✅ Performance Monitoring - Tracking operational metrics")
    print("   ✅ Service Orchestration - Coordinating multiple services")

    print("\n💡 Best Practices:")
    print("   • Use mock services for reliable testing")
    print("   • Implement comprehensive error handling")
    print("   • Monitor performance in integration scenarios")
    print("   • Test service interactions thoroughly")
    print("   • Validate end-to-end workflows")

    print("\n🔗 Next: Explore the hexAI framework documentation!")


if __name__ == "__main__":
    asyncio.run(main())
