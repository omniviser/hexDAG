"""Example observer implementations for the event system.

These are sample implementations showing how to create observers.
They belong in Tier 2 or external systems, not in the core event system.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from hexai.core.application.events import (
    Event,
    LLMPromptSent,
    LLMResponseReceived,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    PipelineCompleted,
    PipelineStarted,
    ToolCalled,
    ToolCompleted,
    WaveCompleted,
    WaveStarted,
)

logger = logging.getLogger(__name__)


class LoggingObserver:
    """Simple logging observer."""

    def __init__(self, log_level: int = logging.INFO):
        self.log_level = log_level
        self.logger = logging.getLogger("pipeline.events")

    async def handle(self, event: Event) -> None:
        """Log events with nice formatting."""
        # Run logging in thread to not block
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._log_event, event)

    def _log_event(self, event: Event) -> None:
        """Actually log the event."""
        match event:
            case PipelineStarted(name=name, total_nodes=nodes, total_waves=waves):
                self.logger.log(
                    self.log_level, f"🎬 Pipeline '{name}' started ({nodes} nodes, {waves} waves)"
                )

            case PipelineCompleted(name=name, duration_ms=ms):
                self.logger.log(
                    self.log_level, f"🎉 Pipeline '{name}' completed in {ms / 1000:.2f}s"
                )

            case WaveStarted(wave_index=idx, nodes=nodes):
                self.logger.log(self.log_level, f"🌊 Wave {idx} started with {len(nodes)} nodes")

            case WaveCompleted(wave_index=idx, duration_ms=ms):
                self.logger.log(self.log_level, f"✅ Wave {idx} completed in {ms / 1000:.2f}s")

            case NodeStarted(name=name, wave_index=wave, dependencies=deps):
                deps_str = f" (deps: {deps})" if deps else ""
                self.logger.log(
                    self.log_level, f"🚀 Node '{name}' started in wave {wave}{deps_str}"
                )

            case NodeCompleted(name=name, duration_ms=ms):
                self.logger.log(self.log_level, f"✅ Node '{name}' completed in {ms / 1000:.2f}s")

            case NodeFailed(name=name, error=error):
                self.logger.error(f"❌ Node '{name}' failed: {error}")

            case LLMPromptSent(node_name=node, messages=msgs):
                self.logger.log(
                    self.log_level, f"💭 LLM prompt sent from '{node}' ({len(msgs)} messages)"
                )

            case LLMResponseReceived(node_name=node, duration_ms=ms):
                self.logger.log(self.log_level, f"🤖 LLM response for '{node}' in {ms / 1000:.2f}s")

            case ToolCalled(node_name=node, tool_name=tool):
                self.logger.log(self.log_level, f"🔧 Tool '{tool}' called from '{node}'")

            case ToolCompleted(tool_name=tool, duration_ms=ms):
                self.logger.log(self.log_level, f"✅ Tool '{tool}' completed in {ms / 1000:.2f}s")


class MetricsObserver:
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


class FileObserver:
    """Writes events to a file for audit/replay."""

    def __init__(self, output_path: Path | str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.events: list[dict[str, Any]] = []

    async def handle(self, event: Event) -> None:
        """Save event to file."""
        event_data = {
            "type": event.__class__.__name__,
            "timestamp": event.timestamp.isoformat(),
            "data": {k: v for k, v in event.__dict__.items() if k != "timestamp"},
        }
        self.events.append(event_data)

        # Write to file (could be optimized to batch writes)
        with open(self.output_path, "w") as f:
            json.dump(self.events, f, indent=2, default=str)


class ConsoleProgressObserver:
    """Shows real-time progress in the console."""

    def __init__(self) -> None:
        self.total_nodes = 0
        self.completed_nodes = 0
        self.current_wave = 0

    async def handle(self, event: Event) -> None:
        """Update console progress."""
        match event:
            case PipelineStarted(total_nodes=nodes):
                self.total_nodes = nodes
                self.completed_nodes = 0
                print(f"\n[Pipeline] Starting with {nodes} nodes...")

            case WaveStarted(wave_index=idx):
                self.current_wave = idx
                print(f"\n[Wave {idx}] Starting...")

            case NodeCompleted(name=name):
                self.completed_nodes += 1
                progress = (self.completed_nodes / self.total_nodes) * 100
                print(f"  ✓ {name} ({self.completed_nodes}/{self.total_nodes} - {progress:.0f}%)")

            case NodeFailed(name=name):
                print(f"  ✗ {name} FAILED")

            case PipelineCompleted():
                print(
                    f"\n[Pipeline] Complete! {self.completed_nodes}/{self.total_nodes} succeeded\n"
                )
