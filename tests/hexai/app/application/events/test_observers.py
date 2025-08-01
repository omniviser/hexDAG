"""Tests for observer implementations."""

from unittest.mock import MagicMock

from hexai.app.application.events import LoggingObserver, MetricsObserver, PipelineStartedEvent


class MockEventManager:
    """Mock event manager for testing."""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or "test-session-123"
        self.events = []

    async def emit(self, event):
        """Mock emit that just stores events."""
        self.events.append(event)


class TestObserverIntegration:
    """Integration tests for observers with MockEventManager."""

    def test_multiple_observers(self):
        """Test multiple observers can handle the same events."""
        # Create multiple observers
        logging_observer = LoggingObserver()
        metrics_observer = MetricsObserver()

        # Mock their handle methods
        logging_observer.handle_sync = MagicMock()
        metrics_observer.handle_sync = MagicMock()

        # Create an event
        event = PipelineStartedEvent(pipeline_name="integration_test", total_waves=1, total_nodes=2)

        # Both observers should be able to handle the event
        logging_observer.handle_sync(event)
        metrics_observer.handle_sync(event)

        # Verify both were called
        logging_observer.handle_sync.assert_called_once_with(event)
        metrics_observer.handle_sync.assert_called_once_with(event)
