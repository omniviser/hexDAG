"""Example observer implementations for the event system.

These are sample implementations showing how to create observers.
They belong in Tier 2 or external systems, not in the core event system.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from hexai.core.application.events import (
    Event,
    LLMPromptSent,
    NodeCompleted,
    NodeFailed,
    PipelineCompleted,
    PipelineStarted,
    ToolCalled,
    WaveCompleted,
    WaveStarted,
)

logger = logging.getLogger(__name__)


class BaseObserver(ABC):
    """Abstract base class for all observers.

    Provides a common interface and optional helper methods for observer implementations.
    """

    @abstractmethod
    async def handle(self, event: Event) -> None:
        """Handle an event.

        Args
        ----
            event: The event to handle

        Note
        ----
            Implementations should be fault-tolerant and not raise exceptions.
            Any errors should be logged internally.
        """
        pass

    def should_handle(self, event: Event) -> bool:
        """Check if this observer should handle the given event.

        Override this method to filter events at the observer level.

        Args
        ----
            event: The event to check

        Returns
        -------
            True if the observer should handle this event, False otherwise
        """
        _ = event  # Mark as intentionally unused
        return True

    async def initialize(self) -> None:
        """Initialize the observer.

        Called once when the observer is registered.
        Override to perform any necessary setup.
        """
        return  # Default implementation does nothing

    async def cleanup(self) -> None:
        """Clean up resources.

        Called when the observer is unregistered or the system shuts down.
        Override to perform any necessary cleanup.
        """
        return  # Default implementation does nothing


class LoggingObserver(BaseObserver):
    """Simple logging observer."""

    def __init__(self, log_level: int = logging.INFO):
        self.log_level = log_level
        self.logger = logging.getLogger("pipeline.events")

    async def handle(self, event: Event) -> None:
        """Log events using their built-in log messages."""
        # Run logging in thread to not block
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._log_event, event)

    def _log_event(self, event: Event) -> None:
        """Log the event using its log_message method."""
        # NodeFailed events should always use ERROR level
        if isinstance(event, NodeFailed):
            self.logger.error(event.log_message(logging.ERROR))
        else:
            self.logger.log(self.log_level, event.log_message(self.log_level))


class MetricsObserver(BaseObserver):
    """Collects execution metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all metrics."""
        self.pipeline_start_time: float | None = None
        self.total_nodes = 0
        self.completed_nodes = 0
        self.failed_nodes = 0
        self.node_durations: list[float] = []
        self.wave_durations: list[float] = []
        self.llm_calls = 0
        self.tool_calls = 0

    async def handle(self, event: Event) -> None:
        """Update metrics based on event."""
        match event:
            case PipelineStarted(total_nodes=nodes):
                self.reset()
                self.pipeline_start_time = time.time()
                self.total_nodes = nodes

            case NodeCompleted(duration_ms=ms):
                self.completed_nodes += 1
                self.node_durations.append(ms / 1000)

            case NodeFailed():
                self.failed_nodes += 1

            case WaveCompleted(duration_ms=ms):
                self.wave_durations.append(ms / 1000)

            case LLMPromptSent():
                self.llm_calls += 1

            case ToolCalled():
                self.tool_calls += 1

            case PipelineCompleted():
                # Log final metrics
                if self.node_durations:
                    avg_node = sum(self.node_durations) / len(self.node_durations)
                else:
                    avg_node = 0

                logger.info(
                    f"📊 Metrics: {self.completed_nodes}/{self.total_nodes} nodes, "
                    f"{self.failed_nodes} failures, avg node time: {avg_node:.2f}s, "
                    f"{self.llm_calls} LLM calls, {self.tool_calls} tool calls"
                )

    def get_summary(self) -> dict[str, Any]:
        """Get current metrics summary."""
        return {
            "total_nodes": self.total_nodes,
            "completed_nodes": self.completed_nodes,
            "failed_nodes": self.failed_nodes,
            "average_node_time": (
                sum(self.node_durations) / len(self.node_durations) if self.node_durations else 0
            ),
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
        }


class FileObserver(BaseObserver):
    """Writes events to a file for audit/replay."""

    def __init__(self, output_path: Path | str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.events: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()  # Thread-safe lock for async operations

    async def handle(self, event: Event) -> None:
        """Save event to file."""
        event_data = {
            "type": event.__class__.__name__,
            "timestamp": event.timestamp.isoformat(),
            "data": {k: v for k, v in event.__dict__.items() if k != "timestamp"},
        }

        # Protect both list append and file write with lock
        async with self._lock:
            self.events.append(event_data)

            # Write to file (could be optimized to batch writes)
            # Run file I/O in executor to avoid blocking
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._write_to_file)

    def _write_to_file(self) -> None:
        """Write events to file (called within lock)."""
        with open(self.output_path, "w") as f:
            json.dump(self.events, f, indent=2, default=str)


class ConsoleProgressObserver(BaseObserver):
    """Shows real-time progress in the console."""

    def __init__(self) -> None:
        self.total_nodes = 0
        self.completed_nodes = 0
        self.current_wave = 0
        self.console_logger = logging.getLogger("console.progress")
        # Set up a console handler specifically for progress output
        if not self.console_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.console_logger.addHandler(handler)
            self.console_logger.setLevel(logging.INFO)

    async def handle(self, event: Event) -> None:
        """Update console progress."""
        match event:
            case PipelineStarted(total_nodes=nodes):
                self.total_nodes = nodes
                self.completed_nodes = 0
                self._output(f"\n[Pipeline] Starting with {nodes} nodes...")

            case WaveStarted(wave_index=idx):
                self.current_wave = idx
                self._output(f"\n[Wave {idx}] Starting...")

            case NodeCompleted(name=name):
                self.completed_nodes += 1
                progress = (self.completed_nodes / self.total_nodes) * 100
                self._output(
                    f"  ✓ {name} ({self.completed_nodes}/{self.total_nodes} - {progress:.0f}%)"
                )

            case NodeFailed(name=name):
                self._output(f"  ✗ {name} FAILED")

            case PipelineCompleted():
                self._output(
                    f"\n[Pipeline] Complete! {self.completed_nodes}/{self.total_nodes} succeeded\n"
                )

    def _output(self, message: str) -> None:
        """Output message to console using logger to avoid flake8 print warnings."""
        self.console_logger.info(message)
