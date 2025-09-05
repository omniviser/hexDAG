"""Built-in observer implementations for logging and metrics."""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from typing import Any, Callable

from .base import PipelineEvent, SyncObserver
from .events import (
    LLMPromptGeneratedEvent,
    LLMResponseReceivedEvent,
    NodeCompletedEvent,
    NodeFailedEvent,
    NodeStartedEvent,
    PipelineCompletedEvent,
    PipelineStartedEvent,
    ToolCalledEvent,
    ToolCompletedEvent,
    ValidationWarningEvent,
)

logger = logging.getLogger(__name__)


class LoggingObserver(SyncObserver):
    """Observer that logs pipeline events with structured logging."""

    def __init__(self, log_level: int = logging.INFO) -> None:
        self.log_level = log_level
        self.logger = logging.getLogger("pipeline.events")

    def handle_sync(self, event: PipelineEvent) -> None:
        """Log the event with appropriate formatting."""
        handlers: dict[type, Callable[[Any], None]] = {
            NodeStartedEvent: lambda e: self.logger.log(
                self.log_level,
                f"🚀 Node '{e.node_name}' started (wave {e.wave_index}, deps: {e.dependencies})",
            ),
            NodeCompletedEvent: lambda e: self.logger.log(
                self.log_level, f"✅ Node '{e.node_name}' completed in {e.execution_time:.2f}s"
            ),
            NodeFailedEvent: lambda e: self.logger.error(
                f"❌ Node '{e.node_name}' failed: {e.error}"
            ),
            PipelineStartedEvent: lambda e: self.logger.log(
                self.log_level,
                f"🎬 Pipeline '{e.pipeline_name}' started "
                f"({e.total_nodes} nodes, {e.total_waves} waves)",
            ),
            PipelineCompletedEvent: lambda e: self.logger.log(
                self.log_level,
                f"🎉 Pipeline '{e.pipeline_name}' completed in {e.total_execution_time:.2f}s",
            ),
            ValidationWarningEvent: lambda e: self.logger.warning(
                f"⚠️  Pipeline '{e.pipeline_name}' has {len(e.warnings)} validation warnings"
            ),
        }

        handler = handlers.get(type(event))
        if handler is not None:
            handler(event)
        else:
            self.logger.log(self.log_level, f"Event: {event.event_type.value}")


class MetricsObserver(SyncObserver):
    """Observer that collects pipeline execution metrics."""

    def __init__(self) -> None:
        self.node_start_times: dict[str, float] = {}
        self.node_execution_times: dict[str, list[float]] = defaultdict(list)
        self.pipeline_start_time: float | None = None
        self.error_count = 0
        self.completed_nodes = 0
        self.total_nodes = 0

    def handle_sync(self, event: PipelineEvent) -> None:
        """Collect metrics based on event type."""
        handlers: dict[type, Callable[[PipelineEvent], None]] = {
            PipelineStartedEvent: self._handle_pipeline_started,
            NodeStartedEvent: self._handle_node_started,
            NodeCompletedEvent: self._handle_node_completed,
            NodeFailedEvent: self._handle_node_failed,
            PipelineCompletedEvent: self._handle_pipeline_completed,
        }

        handler = handlers.get(type(event))
        if handler is not None:
            handler(event)

    def _handle_pipeline_started(self, event: PipelineEvent) -> None:
        if isinstance(event, PipelineStartedEvent):
            self.total_nodes = event.total_nodes
            self.pipeline_start_time = time.time()
            self.error_count = 0
            self.completed_nodes = 0
            logger.debug(f"Started metrics collection for pipeline '{event.pipeline_name}'")

    def _handle_node_started(self, event: PipelineEvent) -> None:
        if isinstance(event, NodeStartedEvent):
            self.node_start_times[event.node_name] = time.time()

    def _handle_node_completed(self, event: PipelineEvent) -> None:
        if isinstance(event, NodeCompletedEvent):
            self.completed_nodes += 1
            self.node_execution_times[event.node_name].append(event.execution_time)
            self.node_start_times.pop(event.node_name, None)

    def _handle_node_failed(self, event: PipelineEvent) -> None:
        if isinstance(event, NodeFailedEvent):
            self.error_count += 1
            self.node_start_times.pop(event.node_name, None)

    def _handle_pipeline_completed(self, event: PipelineEvent) -> None:
        if isinstance(event, PipelineCompletedEvent):
            metrics = self.get_metrics()
            logger.info("📊 Pipeline Execution Metrics:")
            logger.info(f"   Total Time: {event.total_execution_time:.2f}s")
            logger.info(
                f"   Nodes Completed: {metrics['completed_nodes']}/{metrics['total_nodes']}"
            )
            logger.info(f"   Success Rate: {metrics['success_rate']:.1f}%")
            logger.info(f"   Errors: {metrics['error_count']}")

            if metrics["node_execution_times"]:
                logger.info("   Slowest Nodes:")
                slowest = sorted(
                    metrics["node_execution_times"].items(),
                    key=lambda x: sum(x[1]) / len(x[1]),
                    reverse=True,
                )[:3]
                for node, times in slowest:
                    logger.info(f"      - {node}: {sum(times) / len(times):.2f}s avg")

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics summary."""
        success_rate = (
            (self.completed_nodes / self.total_nodes * 100) if self.total_nodes > 0 else 0
        )
        return {
            "completed_nodes": self.completed_nodes,
            "total_nodes": self.total_nodes,
            "error_count": self.error_count,
            "success_rate": round(success_rate, 1),
            "node_execution_times": dict(self.node_execution_times),
        }


class WebSocketObserver(SyncObserver):
    """Observer that streams events to frontend via WebSocket."""

    def __init__(self, websocket_manager: Any) -> None:
        self.websocket_manager = websocket_manager

    def handle_sync(self, event: PipelineEvent) -> None:
        """Stream event to WebSocket clients."""
        event_data = {
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "session_id": event.session_id,
            "metadata": event.metadata or {},
        }

        # Add progress info for node events
        if isinstance(event, NodeCompletedEvent):
            event_data["progress"] = {"type": "node_completed", "node": event.node_name}

        self.websocket_manager.broadcast(json.dumps(event_data))


class FileObserver(SyncObserver):
    """Observer that logs events to a file."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.file_handle = open(filepath, "w")

    def handle_sync(self, event: PipelineEvent) -> None:
        """Write event to file as JSON."""
        event_data = json.dumps(
            {
                "event_type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "session_id": event.session_id,
                "metadata": event.metadata or {},
            }
        )
        self.file_handle.write(event_data + "\n")
        self.file_handle.flush()

    def close(self) -> None:
        """Close the file handle."""
        if not self.file_handle.closed:
            self.file_handle.close()


class NodeObserver(SyncObserver):
    """Observer that provides detailed tracking of node execution lifecycle and LLM operations."""

    def __init__(self, log_level: int = logging.INFO, track_llm_details: bool = True) -> None:
        self.log_level = log_level
        self.track_llm_details = track_llm_details
        self.logger = logging.getLogger("pipeline.nodes")
        self.llm_logger = logging.getLogger("pipeline.llm")
        self.node_states: dict[str, dict[str, Any]] = {}
        self.wave_nodes: dict[int, list[str]] = {}
        self.llm_interactions: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def handle_sync(self, event: PipelineEvent) -> None:
        """Handle node-specific events with detailed tracking."""
        handlers: dict[type, Callable[[PipelineEvent], None]] = {
            NodeStartedEvent: self._handle_node_started,
            NodeCompletedEvent: self._handle_node_completed,
            NodeFailedEvent: self._handle_node_failed,
            LLMPromptGeneratedEvent: self._handle_llm_prompt_generated,
            LLMResponseReceivedEvent: self._handle_llm_response_received,
            ToolCalledEvent: self._handle_tool_called,
            ToolCompletedEvent: self._handle_tool_completed,
        }

        handler = handlers.get(type(event))
        if handler is not None:
            handler(event)

    def _handle_node_started(self, event: PipelineEvent) -> None:
        if isinstance(event, NodeStartedEvent):
            self.node_states[event.node_name] = {
                "status": "running",
                "start_time": time.time(),
                "wave_index": event.wave_index,
                "dependencies": event.dependencies,
            }

            # Track nodes per wave
            if event.wave_index not in self.wave_nodes:
                self.wave_nodes[event.wave_index] = []
            self.wave_nodes[event.wave_index].append(event.node_name)

            deps_str = f" (deps: {event.dependencies})" if event.dependencies else ""
            self.logger.log(
                self.log_level,
                f"🔄 Node '{event.node_name}' started in wave {event.wave_index}{deps_str}",
            )

    def _handle_node_completed(self, event: PipelineEvent) -> None:
        if isinstance(event, NodeCompletedEvent):
            if event.node_name in self.node_states:
                self.node_states[event.node_name].update(
                    {
                        "status": "completed",
                        "end_time": time.time(),
                        "execution_time": event.execution_time,
                        "result_type": type(event.result).__name__,
                    }
                )

            result_info = f" -> {type(event.result).__name__}"
            self.logger.log(
                self.log_level,
                f"✅ Node '{event.node_name}' completed in {event.execution_time:.2f}s{result_info}",
            )

            # Log wave progress
            self._log_wave_progress(event.wave_index)

    def _handle_node_failed(self, event: PipelineEvent) -> None:
        if isinstance(event, NodeFailedEvent):
            if event.node_name in self.node_states:
                self.node_states[event.node_name].update(
                    {
                        "status": "failed",
                        "end_time": time.time(),
                        "error_type": type(event.error).__name__,
                        "error_message": str(event.error),
                    }
                )

            self.logger.error(
                f"❌ Node '{event.node_name}' failed in wave {event.wave_index}: "
                f"{type(event.error).__name__}: {event.error}"
            )

    def _handle_llm_prompt_generated(self, event: PipelineEvent) -> None:
        """Handle LLM prompt generation events."""
        if isinstance(event, LLMPromptGeneratedEvent) and self.track_llm_details:
            interaction = {
                "type": "prompt_generated",
                "timestamp": event.timestamp,
                "template": (
                    event.template[:50] + "..." if len(event.template) > 50 else event.template
                ),
                "message_count": len(event.messages),
                "total_prompt_length": sum(len(msg["content"]) for msg in event.messages),
                "messages": event.messages if self.log_level <= logging.DEBUG else None,
            }
            self.llm_interactions[event.node_name].append(interaction)

            template_info = f" (template: {event.template[:50]}...)" if event.template else ""
            self.llm_logger.log(
                self.log_level,
                f"📝 LLM prompt generated for node '{event.node_name}'{template_info} "
                f"({len(event.messages)} messages, {interaction['total_prompt_length']} chars)",
            )

            if self.log_level <= logging.DEBUG:
                for i, msg in enumerate(event.messages):
                    preview = (
                        msg["content"][:200] + "..."
                        if len(msg["content"]) > 200
                        else msg["content"]
                    )
                    self.llm_logger.debug(f"   Message {i + 1} ({msg['role']}): {preview}")

    def _handle_llm_response_received(self, event: PipelineEvent) -> None:
        """Handle LLM response events."""
        if isinstance(event, LLMResponseReceivedEvent) and self.track_llm_details:
            interaction = {
                "type": "response_received",
                "timestamp": event.timestamp,
                "response_length": event.response_length or len(event.response),
                "response_preview": (
                    event.response[:200] + "..." if len(event.response) > 200 else event.response
                ),
                "full_response": event.response if self.log_level <= logging.DEBUG else None,
            }
            self.llm_interactions[event.node_name].append(interaction)

            self.llm_logger.log(
                self.log_level,
                f"🤖 LLM response received for node '{event.node_name}' "
                f"({event.response_length or len(event.response)} chars)",
            )

            if self.log_level <= logging.DEBUG:
                self.llm_logger.debug(f"   Response: {interaction['response_preview']}")

    def _handle_tool_called(self, event: PipelineEvent) -> None:
        """Handle tool call events."""
        if isinstance(event, ToolCalledEvent) and self.track_llm_details:
            self.llm_logger.log(
                self.log_level,
                f"🔧 Tool '{event.tool_name}' called from node '{event.node_name}' "
                f"(params: {event.tool_params})",
            )

    def _handle_tool_completed(self, event: PipelineEvent) -> None:
        """Handle tool completion events."""
        if isinstance(event, ToolCompletedEvent) and self.track_llm_details:
            self.llm_logger.log(
                self.log_level,
                f"✅ Tool '{event.tool_name}' completed for node '{event.node_name}' "
                f"({type(event.result).__name__} in {event.execution_time:.2f}s)",
            )

    def _count_nodes_with_status(self, nodes: list[str], status: str) -> int:
        """Count nodes with specified status."""
        return sum(1 for node in nodes if self.node_states.get(node, {}).get("status") == status)

    def _log_wave_progress(self, wave_index: int) -> None:
        """Log progress of current wave."""
        if wave_index not in self.wave_nodes:
            return

        wave_nodes = self.wave_nodes[wave_index]
        completed = self._count_nodes_with_status(wave_nodes, "completed")
        failed = self._count_nodes_with_status(wave_nodes, "failed")
        total = len(wave_nodes)

        if completed + failed == total:
            success_rate = (completed / total * 100) if total > 0 else 0
            self.logger.log(
                self.log_level,
                f"🌊 Wave {wave_index} completed: {completed}/{total} nodes successful "
                f"({success_rate:.1f}% success rate)",
            )

    def get_node_summary(self) -> dict[str, Any]:
        """Get summary of all node executions."""
        completed = [n for n, s in self.node_states.items() if s.get("status") == "completed"]
        failed = [n for n, s in self.node_states.items() if s.get("status") == "failed"]
        running = [n for n, s in self.node_states.items() if s.get("status") == "running"]

        execution_times = {
            name: state.get("execution_time", 0)
            for name, state in self.node_states.items()
            if "execution_time" in state
        }

        # Calculate LLM interaction statistics
        llm_stats = self._calculate_llm_stats() if self.track_llm_details else {}

        return {
            "total_nodes": len(self.node_states),
            "completed": len(completed),
            "failed": len(failed),
            "running": len(running),
            "completed_nodes": completed,
            "failed_nodes": failed,
            "running_nodes": running,
            "execution_times": execution_times,
            "slowest_nodes": sorted(execution_times.items(), key=lambda x: x[1], reverse=True)[:5],
            "wave_distribution": dict(self.wave_nodes),
            "llm_statistics": llm_stats,
        }

    def _calculate_llm_stats(self) -> dict[str, Any]:
        """Calculate LLM interaction statistics."""
        total_prompts = 0
        total_responses = 0
        total_prompt_chars = 0
        total_response_chars = 0
        total_response_time = 0.0
        nodes_with_llm = set()

        for node_name, interactions in self.llm_interactions.items():
            nodes_with_llm.add(node_name)
            for interaction in interactions:
                if interaction["type"] == "prompt_generated":
                    total_prompts += 1
                    total_prompt_chars += interaction.get("total_prompt_length", 0)
                elif interaction["type"] == "response_received":
                    total_responses += 1
                    total_response_chars += interaction.get("response_length", 0)
                    total_response_time += interaction.get("response_time", 0.0)

        avg_response_time = total_response_time / total_responses if total_responses > 0 else 0.0

        return {
            "nodes_with_llm_interactions": len(nodes_with_llm),
            "total_prompts_generated": total_prompts,
            "total_responses_received": total_responses,
            "total_prompt_characters": total_prompt_chars,
            "total_response_characters": total_response_chars,
            "total_response_time": round(total_response_time, 2),
            "average_response_time": round(avg_response_time, 2),
            "nodes_by_llm_activity": sorted(
                [(node, len(interactions)) for node, interactions in self.llm_interactions.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:5],
        }

    def get_llm_interactions_for_node(self, node_name: str) -> list[dict[str, Any]]:
        """Get all LLM interactions for a specific node."""
        return self.llm_interactions.get(node_name, [])
