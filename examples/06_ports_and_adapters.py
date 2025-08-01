"""
Example 06: Ports and Adapters Pattern

This example demonstrates the hexagonal architecture pattern with ports and adapters:
- Ports: Define interfaces for external dependencies
- Adapters: Implement the interfaces for different technologies
- Mock adapters for testing
- Real adapters for production
"""

import asyncio
from typing import Any, Protocol

from hexai.app.application.orchestrator import Orchestrator
from hexai.app.domain.dag import DirectedGraph, NodeSpec


# Ports (Interfaces)
class DatabasePort(Protocol):
    """Port for database operations."""

    async def save(self, data: dict) -> bool:
        """Save data to database."""
        ...

    async def load(self, key: str) -> dict:
        """Load data from database."""
        ...


class EmailPort(Protocol):
    """Port for email operations."""

    async def send(self, to: str, subject: str, body: str) -> bool:
        """Send email."""
        ...


class LoggerPort(Protocol):
    """Port for logging operations."""

    async def log(self, level: str, message: str) -> None:
        """Log message."""
        ...


# Mock Adapters (for testing)
class MockDatabaseAdapter:
    """Mock database adapter for testing."""

    def __init__(self):
        self.storage = {}

    async def save(self, data: dict) -> bool:
        """Mock save operation."""
        key = data.get("id", "default")
        self.storage[key] = data
        return True

    async def load(self, key: str) -> dict:
        """Mock load operation."""
        return self.storage.get(key, {})


class MockEmailAdapter:
    """Mock email adapter for testing."""

    def __init__(self):
        self.sent_emails = []

    async def send(self, to: str, subject: str, body: str) -> bool:
        """Mock send email operation."""
        self.sent_emails.append({"to": to, "subject": subject, "body": body})
        return True


class MockLoggerAdapter:
    """Mock logger adapter for testing."""

    def __init__(self):
        self.logs = []

    async def log(self, level: str, message: str) -> None:
        """Mock log operation."""
        self.logs.append({"level": level, "message": message})


# Business Logic Functions
async def process_user_data(input_data: dict, **ports) -> dict:
    """Process user data using ports."""
    # Get ports from kwargs
    db = ports.get("database")
    email = ports.get("email")
    logger = ports.get("logger")

    # Business logic
    user_id = input_data.get("user_id", "unknown")
    user_name = input_data.get("name", "Unknown User")

    # Use database port
    await db.save({"id": user_id, "name": user_name, "processed": True})

    # Use logger port
    await logger.log("INFO", f"Processed user: {user_name}")

    # Use email port
    await email.send(
        to=f"{user_name}@example.com",
        subject="Processing Complete",
        body=f"Hello {user_name}, your data has been processed successfully!",
    )

    return {
        "user_id": user_id,
        "user_name": user_name,
        "status": "processed",
        "message": "User data processed with all ports",
    }


async def generate_report(input_data: Any, **kwargs) -> dict:
    """Generate report using database port."""
    db = kwargs.get("database")
    logger = kwargs.get("logger")

    # Load data from database
    user_data = await db.load("default")

    # Generate report
    report = {
        "total_users": len(user_data),
        "processed_users": sum(1 for user in user_data.values() if user.get("processed")),
        "timestamp": "2024-01-01",
    }

    # Log report generation
    await logger.log("INFO", f"Generated report: {report}")

    return {"report": report, "status": "completed"}


async def main():
    """Demonstrate ports and adapters pattern."""

    print("🔌 Example 06: Ports and Adapters Pattern")
    print("=" * 50)

    print("\n🎯 This example demonstrates:")
    print("   • Ports (interfaces) for external dependencies")
    print("   • Mock adapters for testing")
    print("   • Dependency injection through orchestrator")
    print("   • Hexagonal architecture principles")

    # Create mock adapters
    mock_db = MockDatabaseAdapter()
    mock_email = MockEmailAdapter()
    mock_logger = MockLoggerAdapter()

    # Create orchestrator with ports
    orchestrator = Orchestrator(
        ports={"database": mock_db, "email": mock_email, "logger": mock_logger}
    )

    # Create DAG
    graph = DirectedGraph()

    # Add nodes
    processor = NodeSpec("process_user", process_user_data)
    reporter = NodeSpec("generate_report", generate_report).after("process_user")

    graph.add(processor)
    graph.add(reporter)

    # Test data
    test_user = {"user_id": "user123", "name": "John Doe", "email": "john@example.com"}

    print(f"\n🚀 Executing pipeline with test data:")
    print(f"   User: {test_user['name']} (ID: {test_user['user_id']})")

    # Execute pipeline
    results = await orchestrator.run(graph, test_user)

    # Show results
    print(f"\n📋 Pipeline Results:")
    print(f"   Process Result: {results['process_user']['status']}")
    print(f"   Report Status: {results['generate_report']['status']}")

    # Show adapter activity
    print(f"\n🔌 Adapter Activity:")
    print(f"   Database saves: {len(mock_db.storage)}")
    print(f"   Emails sent: {len(mock_email.sent_emails)}")
    print(f"   Log entries: {len(mock_logger.logs)}")

    # Show detailed adapter results
    print(f"\n📊 Detailed Adapter Results:")

    print(f"   Database Storage:")
    for key, value in mock_db.storage.items():
        print(f"     {key}: {value}")

    print(f"   Sent Emails:")
    for email in mock_email.sent_emails:
        print(f"     To: {email['to']}")
        print(f"     Subject: {email['subject']}")

    print(f"   Log Entries:")
    for log in mock_logger.logs:
        print(f"     [{log['level']}] {log['message']}")

    print(f"\n🎯 Key Concepts Learned:")
    print("   ✅ Ports - Define interfaces for external dependencies")
    print("   ✅ Adapters - Implement interfaces for different technologies")
    print("   ✅ Dependency Injection - Pass adapters through orchestrator")
    print("   ✅ Mock Adapters - Test business logic without real dependencies")
    print("   ✅ Hexagonal Architecture - Separate business logic from infrastructure")

    print(f"\n🔗 Next: Run example 07 to learn about error handling!")


if __name__ == "__main__":
    asyncio.run(main())
