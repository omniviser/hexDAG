"""Simple observer implementations for the clean event system."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from .events import (
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
    """Logs events to a JSON file."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    async def handle(self, event: Event) -> None:
        """Write event to file as JSON."""
        # Run file I/O in thread
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._write_event, event)

    def _write_event(self, event: Event) -> None:
        """Actually write the event."""
        try:
            # Convert event to dict
            event_data: dict[str, Any] = {"type": event.__class__.__name__, "timestamp": None}

            # Safe timestamp extraction
            if hasattr(event, "timestamp"):
                ts = getattr(event, "timestamp", None)
                if ts is not None:
                    event_data["timestamp"] = ts.isoformat()

            # Add all other fields
            for key, value in event.__dict__.items():
                if key != "timestamp":
                    # Handle exceptions specially
                    if isinstance(value, Exception):
                        event_data[key] = str(value)
                    else:
                        event_data[key] = value

            # Append to file
            with open(self.file_path, "a") as f:
                json.dump(event_data, f)
                f.write("\n")
        except Exception as e:
            logger.error(f"Failed to write event to file: {e}")


class NodeStateObserver:
    """Tracks detailed node execution state."""

    def __init__(self) -> None:
        self.node_states: dict[str, dict[str, Any]] = {}
        self.node_history: list[dict[str, Any]] = []

    async def handle(self, event: Event) -> None:
        """Track node state changes."""
        match event:
            case NodeStarted(name=name, wave_index=wave, dependencies=deps, timestamp=ts):
                self.node_states[name] = {
                    "status": "running",
                    "started_at": ts,
                    "wave_index": wave,
                    "dependencies": deps,
                    "llm_prompts": [],
                    "tool_calls": [],
                }
                self.node_history.append(
                    {
                        "node": name,
                        "event": "started",
                        "timestamp": ts,
                        "wave_index": wave,
                    }
                )

            case NodeCompleted(name=name, result=result, duration_ms=ms, timestamp=ts):
                if name in self.node_states:
                    self.node_states[name].update(
                        {
                            "status": "completed",
                            "completed_at": ts,
                            "duration_ms": ms,
                            "result": result,
                        }
                    )
                    self.node_history.append(
                        {
                            "node": name,
                            "event": "completed",
                            "timestamp": ts,
                            "duration_ms": ms,
                        }
                    )

            case NodeFailed(name=name, error=error, timestamp=ts):
                if name in self.node_states:
                    self.node_states[name].update(
                        {
                            "status": "failed",
                            "failed_at": ts,
                            "error": str(error),
                        }
                    )
                    self.node_history.append(
                        {
                            "node": name,
                            "event": "failed",
                            "timestamp": ts,
                            "error": str(error),
                        }
                    )

            case LLMPromptSent(node_name=name, messages=msgs, timestamp=ts):
                if name in self.node_states:
                    self.node_states[name]["llm_prompts"].append(
                        {
                            "timestamp": ts,
                            "messages": msgs,
                        }
                    )

            case ToolCalled(node_name=name, tool_name=tool, params=params, timestamp=ts):
                if name in self.node_states:
                    self.node_states[name]["tool_calls"].append(
                        {
                            "timestamp": ts,
                            "tool": tool,
                            "params": params,
                        }
                    )

    def get_node_state(self, node_name: str) -> dict[str, Any] | None:
        """Get current state of a specific node."""
        return self.node_states.get(node_name)

    def get_running_nodes(self) -> list[str]:
        """Get currently running nodes."""
        return [name for name, state in self.node_states.items() if state["status"] == "running"]

    def get_failed_nodes(self) -> list[str]:
        """Get failed nodes."""
        return [name for name, state in self.node_states.items() if state["status"] == "failed"]


class WebSocketObserver:
    """Sends events over WebSocket for real-time monitoring."""

    def __init__(self, websocket_handler: Any):
        self.websocket_handler = websocket_handler

    async def handle(self, event: Event) -> None:
        """Send event over WebSocket."""
        try:
            # Convert event to dict
            event_data: dict[str, Any] = {"type": event.__class__.__name__, "timestamp": None}

            # Safe timestamp extraction
            if hasattr(event, "timestamp"):
                ts = getattr(event, "timestamp", None)
                if ts is not None:
                    event_data["timestamp"] = ts.isoformat()

            # Add all fields
            for key, value in event.__dict__.items():
                if key != "timestamp":
                    if isinstance(value, Exception):
                        event_data[key] = str(value)
                    else:
                        event_data[key] = value

            # Add progress info for certain events
            match event:
                case NodeCompleted(name=name):
                    event_data["progress"] = {"type": "node_completed", "node": name}
                case WaveCompleted(wave_index=idx):
                    event_data["progress"] = {"type": "wave_completed", "wave": idx}

            # Send via WebSocket
            self.websocket_handler.send_json(event_data)
        except Exception as e:
            logger.error(f"Failed to send event via WebSocket: {e}")
