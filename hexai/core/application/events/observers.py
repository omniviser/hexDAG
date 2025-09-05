"""Built-in observer implementations for logging and metrics."""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from typing import Any

from .base import PipelineEvent, SyncObserver
from .events import ExecutionEvent, ExecutionLevel, ExecutionPhase, HookEvent, LLMEvent, MetaEvent

logger = logging.getLogger(__name__)


class LoggingObserver(SyncObserver):
    """Observer that logs pipeline events with structured logging."""

    def __init__(self, log_level: int = logging.INFO) -> None:
        self.log_level = log_level
        self.logger = logging.getLogger("pipeline.events")

    def handle_sync(self, event: PipelineEvent) -> None:
        """Log the event with appropriate formatting."""
        if isinstance(event, ExecutionEvent):
            self._handle_execution_event(event)
        elif isinstance(event, LLMEvent):
            self._handle_llm_event(event)
        elif isinstance(event, HookEvent):
            self._handle_hook_event(event)
        elif isinstance(event, MetaEvent):
            self._handle_meta_event(event)

    def _handle_execution_event(self, event: ExecutionEvent) -> None:
        """Handle execution events."""
        if event.level == ExecutionLevel.NODE:
            if event.phase == ExecutionPhase.STARTED:
                self.logger.log(
                    self.log_level,
                    f"🚀 Node '{event.name}' started "
                    f"(wave {event.wave_index}, deps: {event.dependencies})",
                )
            elif event.phase == ExecutionPhase.COMPLETED:
                time_str = (
                    f" in {event.execution_time_ms / 1000:.2f}s" if event.execution_time_ms else ""
                )
                self.logger.log(self.log_level, f"✅ Node '{event.name}' completed{time_str}")
            elif event.phase == ExecutionPhase.FAILED:
                self.logger.error(f"❌ Node '{event.name}' failed: {event.error}")
        elif event.level == ExecutionLevel.DAG:
            if event.phase == ExecutionPhase.STARTED:
                self.logger.log(
                    self.log_level,
                    f"🎬 Pipeline '{event.name}' started "
                    f"({event.total_nodes} nodes, {event.total_waves} waves)",
                )
            elif event.phase == ExecutionPhase.COMPLETED:
                time_str = (
                    f" in {event.execution_time_ms / 1000:.2f}s" if event.execution_time_ms else ""
                )
                self.logger.log(self.log_level, f"🎉 Pipeline '{event.name}' completed{time_str}")

    def _handle_llm_event(self, event: LLMEvent) -> None:
        """Handle LLM and tool events."""
        if event.event_class == "llm":
            if event.action == "prompt":
                messages_len = len(event.messages) if event.messages else 0
                self.logger.log(
                    self.log_level,
                    f"💭 LLM prompt generated for '{event.node_name}' ({messages_len} messages)",
                )
            elif event.action == "response":
                response_preview = (
                    str(event.output_data)[:200] + "..."
                    if len(str(event.output_data)) > 200
                    else str(event.output_data)
                )
                self.logger.log(
                    self.log_level,
                    f"🤖 LLM response received for '{event.node_name}': {response_preview}",
                )
        elif event.event_class == "tool":
            if event.action == "called":
                self.logger.log(
                    self.log_level, f"🔧 Tool '{event.tool_name}' called from '{event.node_name}'"
                )
            elif event.action == "completed":
                time_str = (
                    f" in {event.execution_time_ms / 1000:.2f}s" if event.execution_time_ms else ""
                )
                self.logger.log(self.log_level, f"✅ Tool '{event.tool_name}' completed{time_str}")

    def _handle_hook_event(self, event: HookEvent) -> None:
        """Handle hook events."""
        self.logger.debug(f"🪝 Hook '{event.hook_type}' triggered for '{event.target_name}'")

    def _handle_meta_event(self, event: MetaEvent) -> None:
        """Handle meta events."""
        if event.category == "validation":
            self.logger.warning(
                f"⚠️  Pipeline '{event.pipeline_name}' has {len(event.warnings)} warnings"
            )
        elif event.category == "build":
            self.logger.info(f"🔨 Pipeline '{event.pipeline_name}': {event.message}")
        elif event.category == "diagnostic":
            self.logger.debug(f"📊 Pipeline '{event.pipeline_name}': {event.message}")


class MetricsObserver(SyncObserver):
    """Observer that collects metrics about pipeline execution."""

    def __init__(self) -> None:
        self.start_time: float | None = None
        self.total_nodes = 0
        self.completed_nodes = 0
        self.error_count = 0
        self.node_start_times: dict[str, float] = {}
        self.node_execution_times: dict[str, list[float]] = defaultdict(list)
        self.llm_call_count = 0
        self.tool_call_count = 0

    def handle_sync(self, event: PipelineEvent) -> None:
        """Update metrics based on the event."""
        if isinstance(event, ExecutionEvent):
            self._handle_execution_event(event)
        elif isinstance(event, LLMEvent):
            self._handle_llm_event(event)

    def _handle_execution_event(self, event: ExecutionEvent) -> None:
        """Handle execution events for metrics."""
        if event.level == ExecutionLevel.DAG and event.phase == ExecutionPhase.STARTED:
            self.start_time = time.time()
            self.total_nodes = event.total_nodes or 0
        elif event.level == ExecutionLevel.NODE:
            if event.phase == ExecutionPhase.STARTED:
                self.node_start_times[event.name] = time.time()
            elif event.phase == ExecutionPhase.COMPLETED:
                self.completed_nodes += 1
                if event.execution_time_ms:
                    self.node_execution_times[event.name].append(event.execution_time_ms / 1000)
            elif event.phase == ExecutionPhase.FAILED:
                self.error_count += 1
                self.node_start_times.pop(event.name, None)
        elif event.level == ExecutionLevel.DAG and event.phase == ExecutionPhase.COMPLETED:
            if self.node_execution_times:
                all_times = [t for times in self.node_execution_times.values() for t in times]
                avg_time = sum(all_times) / len(all_times) if all_times else 0
                logger.info(
                    f"Pipeline metrics: {self.completed_nodes}/{self.total_nodes} nodes completed, "
                    f"{self.error_count} errors, avg node time: {avg_time:.2f}s"
                )

    def _handle_llm_event(self, event: LLMEvent) -> None:
        """Handle LLM and tool events for metrics."""
        if event.event_class == "llm" and event.action == "prompt":
            self.llm_call_count += 1
        elif event.event_class == "tool" and event.action == "called":
            self.tool_call_count += 1

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of collected metrics."""
        all_times = [t for times in self.node_execution_times.values() for t in times]
        return {
            "total_nodes": self.total_nodes,
            "completed_nodes": self.completed_nodes,
            "error_count": self.error_count,
            "average_node_time": sum(all_times) / len(all_times) if all_times else 0,
            "llm_calls": self.llm_call_count,
            "tool_calls": self.tool_call_count,
        }


class WebSocketObserver(SyncObserver):
    """Observer that sends events over WebSocket for real-time monitoring."""

    def __init__(self, websocket_handler: Any) -> None:
        """Initialize with a WebSocket handler."""
        self.websocket_handler = websocket_handler

    def handle_sync(self, event: PipelineEvent) -> None:
        """Send event data over WebSocket."""
        try:
            event_data = {
                "type": event.__class__.__name__,
                "timestamp": event.timestamp.isoformat() if hasattr(event, "timestamp") else None,
                "session_id": getattr(event, "session_id", None),
            }

            # Add event-specific data
            if isinstance(event, ExecutionEvent):
                event_data.update(
                    {
                        "level": event.level,
                        "phase": event.phase,
                        "name": event.name,
                        "wave_index": event.wave_index,
                        "execution_time_ms": event.execution_time_ms,
                    }
                )

                # Add progress info for node events
                if event.level == ExecutionLevel.NODE and event.phase == ExecutionPhase.COMPLETED:
                    event_data["progress"] = {"type": "node_completed", "node": event.name}
            elif isinstance(event, LLMEvent):
                event_data.update(
                    {
                        "event_class": event.event_class,
                        "action": event.action,
                        "node_name": event.node_name,
                        "tool_name": event.tool_name,
                    }
                )

            # Send via WebSocket
            self.websocket_handler.send_json(event_data)

        except Exception as e:
            logger.error(f"Failed to send event via WebSocket: {e}")


class NodeObserver(SyncObserver):
    """Observer that tracks detailed node execution state."""

    def __init__(self) -> None:
        self.node_states: dict[str, dict[str, Any]] = {}
        self.node_history: list[dict[str, Any]] = []

    def handle_sync(self, event: PipelineEvent) -> None:
        """Handle node-specific events with detailed tracking."""
        if isinstance(event, ExecutionEvent):
            self._handle_execution_event(event)
        elif isinstance(event, LLMEvent):
            self._handle_llm_event(event)

    def _handle_execution_event(self, event: ExecutionEvent) -> None:
        """Handle execution events for node tracking."""
        if event.level == ExecutionLevel.NODE:
            if event.phase == ExecutionPhase.STARTED:
                self.node_states[event.name] = {
                    "status": "running",
                    "started_at": event.timestamp,
                    "wave_index": event.wave_index,
                    "dependencies": event.dependencies,
                    "llm_prompts": [],
                    "llm_responses": [],
                    "tool_calls": [],
                }

                self.node_history.append(
                    {
                        "node": event.name,
                        "event": "started",
                        "timestamp": event.timestamp,
                        "wave_index": event.wave_index,
                    }
                )
            elif event.phase == ExecutionPhase.COMPLETED:
                if event.name in self.node_states:
                    self.node_states[event.name].update(
                        {
                            "status": "completed",
                            "completed_at": event.timestamp,
                            "execution_time_ms": event.execution_time_ms,
                            "result": event.result,
                        }
                    )

                    self.node_history.append(
                        {
                            "node": event.name,
                            "event": "completed",
                            "timestamp": event.timestamp,
                            "execution_time_ms": event.execution_time_ms,
                        }
                    )
            elif event.phase == ExecutionPhase.FAILED:
                if event.name in self.node_states:
                    self.node_states[event.name].update(
                        {
                            "status": "failed",
                            "failed_at": event.timestamp,
                            "error": str(event.error),
                        }
                    )

                    self.node_history.append(
                        {
                            "node": event.name,
                            "event": "failed",
                            "timestamp": event.timestamp,
                            "error": str(event.error),
                        }
                    )

    def _handle_llm_event(self, event: LLMEvent) -> None:
        """Handle LLM events for node tracking."""
        if event.node_name in self.node_states:
            if event.event_class == "llm":
                if event.action == "prompt" and event.messages:
                    self.node_states[event.node_name]["llm_prompts"].append(
                        {
                            "timestamp": event.timestamp,
                            "messages": event.messages,
                            "template_vars": event.template_vars,
                        }
                    )
                elif event.action == "response":
                    self.node_states[event.node_name]["llm_responses"].append(
                        {
                            "timestamp": event.timestamp,
                            "response": str(event.output_data)[:500],  # Truncate long responses
                        }
                    )
            elif event.event_class == "tool" and event.action == "called":
                self.node_states[event.node_name]["tool_calls"].append(
                    {
                        "timestamp": event.timestamp,
                        "tool": event.tool_name,
                        "params": event.input_data,
                    }
                )

    def get_node_state(self, node_name: str) -> dict[str, Any] | None:
        """Get the current state of a specific node."""
        return self.node_states.get(node_name)

    def get_running_nodes(self) -> list[str]:
        """Get list of currently running nodes."""
        return [name for name, state in self.node_states.items() if state["status"] == "running"]

    def get_failed_nodes(self) -> list[str]:
        """Get list of failed nodes."""
        return [name for name, state in self.node_states.items() if state["status"] == "failed"]


class FileObserver(SyncObserver):
    """Observer that logs events to a file in JSON format."""

    def __init__(self, file_path: str) -> None:
        """Initialize with output file path."""
        self.file_path = file_path
        self.file = open(file_path, "a")

    def handle_sync(self, event: PipelineEvent) -> None:
        """Write event to file as JSON."""
        try:
            event_data = {
                "timestamp": event.timestamp.isoformat() if hasattr(event, "timestamp") else None,
                "type": event.__class__.__name__,
                "session_id": getattr(event, "session_id", None),
            }

            # Add extra fields based on event type
            if hasattr(event, "_extra_fields"):
                event_data.update(event._extra_fields())

            # Write JSON line
            json.dump(event_data, self.file)
            self.file.write("\n")
            self.file.flush()

        except Exception as e:
            logger.error(f"Failed to write event to file: {e}")

    def close(self) -> None:
        """Close the file."""
        self.file.close()

    def __del__(self) -> None:
        """Ensure file is closed when observer is destroyed."""
        if hasattr(self, "file") and not self.file.closed:
            self.file.close()
