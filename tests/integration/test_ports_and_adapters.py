"""Integration tests for hexagonal architecture ports and adapters pattern.

Tests demonstrate:
- Port interfaces for external dependencies
- Mock adapter implementations for testing
- Dependency injection through orchestrator
- Hexagonal architecture principles
"""

import pytest

from hexdag.kernel.domain.dag import DirectedGraph, NodeSpec
from hexdag.kernel.orchestration.orchestrator import Orchestrator


# Mock Adapters for testing
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
async def process_user_data(input_data: dict, **kwargs) -> dict:
    """Process user data using ports."""
    from hexdag.kernel.context import get_port

    db = get_port("database")
    email = get_port("email")
    logger = get_port("logger")

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


async def generate_report(input_data: dict, **kwargs) -> dict:
    """Generate report using database port."""
    from hexdag.kernel.context import get_port

    db = get_port("database")
    logger = get_port("logger")

    # Load data from database
    user_data = await db.load("default")

    # Generate report
    report = {
        "total_users": len(user_data),
        "processed_users": sum(1 for user in [user_data] if user.get("processed")),
        "timestamp": "2024-01-01",
    }

    # Log report generation
    await logger.log("INFO", f"Generated report: {report}")

    return {"report": report, "status": "completed"}


@pytest.fixture
def mock_adapters():
    """Provide mock adapters for testing."""
    return {
        "database": MockDatabaseAdapter(),
        "email": MockEmailAdapter(),
        "logger": MockLoggerAdapter(),
    }


@pytest.fixture
def orchestrator(mock_adapters):
    """Provide orchestrator with mock ports."""
    return Orchestrator(ports=mock_adapters)


class TestPortsAndAdapters:
    """Test suite for ports and adapters pattern."""

    @pytest.mark.asyncio
    async def test_basic_port_usage(self, orchestrator, mock_adapters):
        """Test basic usage of ports through orchestrator."""
        # Create simple pipeline
        graph = DirectedGraph()
        processor = NodeSpec("process_user", process_user_data)
        graph.add(processor)

        # Test data
        test_user = {
            "user_id": "user123",
            "name": "John Doe",
            "email": "john@example.com",
        }

        # Execute pipeline
        results = await orchestrator.run(graph, test_user)

        # Verify results
        assert results["process_user"]["status"] == "processed"
        assert results["process_user"]["user_id"] == "user123"
        assert results["process_user"]["user_name"] == "John Doe"

        # Verify adapter interactions
        db = mock_adapters["database"]
        email = mock_adapters["email"]
        logger = mock_adapters["logger"]

        assert len(db.storage) == 1
        assert len(email.sent_emails) == 1
        assert len(logger.logs) == 1

    @pytest.mark.asyncio
    async def test_multiple_nodes_with_ports(self, orchestrator, mock_adapters):
        """Test multiple nodes using the same ports."""
        # Create DAG with dependencies
        graph = DirectedGraph()
        processor = NodeSpec("process_user", process_user_data)
        reporter = NodeSpec("generate_report", generate_report).after("process_user")

        graph.add(processor)
        graph.add(reporter)

        # Test data
        test_user = {
            "user_id": "user456",
            "name": "Jane Smith",
        }

        # Execute pipeline
        results = await orchestrator.run(graph, test_user)

        # Verify both nodes executed
        assert "process_user" in results
        assert "generate_report" in results
        assert results["process_user"]["status"] == "processed"
        assert results["generate_report"]["status"] == "completed"

        # Verify all adapters were used
        logger = mock_adapters["logger"]
        assert len(logger.logs) >= 2  # At least one log per node

    @pytest.mark.asyncio
    async def test_database_adapter_functionality(self, orchestrator, mock_adapters):
        """Test database adapter save and load operations."""
        db = mock_adapters["database"]

        # Test save
        test_data = {"id": "test1", "value": "data"}
        saved = await db.save(test_data)
        assert saved is True
        assert "test1" in db.storage

        # Test load
        loaded = await db.load("test1")
        assert loaded == test_data

        # Test load non-existent
        empty = await db.load("nonexistent")
        assert empty == {}

    @pytest.mark.asyncio
    async def test_email_adapter_functionality(self, orchestrator, mock_adapters):
        """Test email adapter send operations."""
        email = mock_adapters["email"]

        # Test send
        result = await email.send(to="test@example.com", subject="Test Subject", body="Test Body")
        assert result is True
        assert len(email.sent_emails) == 1

        # Verify email details
        sent_email = email.sent_emails[0]
        assert sent_email["to"] == "test@example.com"
        assert sent_email["subject"] == "Test Subject"
        assert sent_email["body"] == "Test Body"

    @pytest.mark.asyncio
    async def test_logger_adapter_functionality(self, orchestrator, mock_adapters):
        """Test logger adapter log operations."""
        logger = mock_adapters["logger"]

        # Test different log levels
        await logger.log("INFO", "Info message")
        await logger.log("WARNING", "Warning message")
        await logger.log("ERROR", "Error message")

        assert len(logger.logs) == 3
        assert logger.logs[0]["level"] == "INFO"
        assert logger.logs[1]["level"] == "WARNING"
        assert logger.logs[2]["level"] == "ERROR"

    @pytest.mark.asyncio
    async def test_dependency_injection(self, mock_adapters):
        """Test that ports are properly injected through orchestrator."""
        # Create orchestrator with specific ports
        orchestrator = Orchestrator(ports=mock_adapters)

        graph = DirectedGraph()
        processor = NodeSpec("process_user", process_user_data)
        graph.add(processor)

        test_user = {"user_id": "test", "name": "Test User"}

        # Execute and verify port injection worked
        await orchestrator.run(graph, test_user)

        # All adapters should have been accessed
        assert len(mock_adapters["database"].storage) > 0
        assert len(mock_adapters["email"].sent_emails) > 0
        assert len(mock_adapters["logger"].logs) > 0

    @pytest.mark.asyncio
    async def test_adapter_isolation(self):
        """Test that different orchestrators have isolated adapters."""
        # Create two separate orchestrators with different adapters
        db1 = MockDatabaseAdapter()
        db2 = MockDatabaseAdapter()

        Orchestrator(ports={"database": db1})
        Orchestrator(ports={"database": db2})

        # Save to first database
        await db1.save({"id": "test1", "data": "value1"})

        # Verify isolation
        assert len(db1.storage) == 1
        assert len(db2.storage) == 0

        # Save to second database
        await db2.save({"id": "test2", "data": "value2"})

        assert len(db1.storage) == 1
        assert len(db2.storage) == 1
        assert "test1" in db1.storage
        assert "test2" in db2.storage

    @pytest.mark.asyncio
    async def test_port_reusability(self, orchestrator, mock_adapters):
        """Test that ports can be reused across multiple executions."""
        graph = DirectedGraph()
        processor = NodeSpec("process_user", process_user_data)
        graph.add(processor)

        # Execute multiple times
        users = [
            {"user_id": "user1", "name": "User One"},
            {"user_id": "user2", "name": "User Two"},
            {"user_id": "user3", "name": "User Three"},
        ]

        for user in users:
            results = await orchestrator.run(graph, user)
            assert results["process_user"]["status"] == "processed"

        # Verify all executions used the same adapters
        db = mock_adapters["database"]
        email = mock_adapters["email"]
        logger = mock_adapters["logger"]

        assert len(db.storage) == 3
        assert len(email.sent_emails) == 3
        assert len(logger.logs) == 3
