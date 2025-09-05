#!/usr/bin/env python3
"""
📊 Example 05: Event System and Monitoring.

This example teaches:
- Event observers and monitoring
- Custom event handlers
- Pipeline metrics collection
- Real-time execution tracking

Run: python examples/05_event_system.py
"""

import asyncio
import time
from typing import Any

from hexai.core.application.events.base import Observer, PipelineEvent
from hexai.core.application.events.events import (
    NodeCompletedEvent,
    NodeFailedEvent,
    NodeStartedEvent,
    PipelineStartedEvent,
)
from hexai.core.application.events.manager import PipelineEventManager
from hexai.core.application.events.observers import LoggingObserver
from hexai.core.application.orchestrator import Orchestrator
from hexai.core.domain.dag import DirectedGraph, NodeSpec
from hexai.core.validation import coerce_validator


class CustomMetricsObserver(Observer):
    """Custom observer that collects detailed execution metrics."""

    def __init__(self):
        """Initialize metrics collection."""
        self.metrics = {
            "nodes_started": 0,
            "nodes_completed": 0,
            "nodes_failed": 0,
            "total_execution_time": 0,
            "node_timings": {},
            "events_received": [],
        }
        self.node_start_times = {}

    def can_handle(self, event: PipelineEvent) -> bool:
        """Check if this observer can handle the event - we handle all events."""
        return True

    async def handle(self, event: PipelineEvent) -> None:
        """Handle pipeline events."""
        if isinstance(event, NodeStartedEvent):
            self.metrics["nodes_started"] += 1
            self.node_start_times[event.node_name] = time.time()
            self.metrics["events_received"].append(f"STARTED: {event.node_name}")
            print(f"   📊 Metrics: {event.node_name} started")

        elif isinstance(event, NodeCompletedEvent):
            self.metrics["nodes_completed"] += 1

            # Calculate execution time
            if event.node_name in self.node_start_times:
                start_time = self.node_start_times[event.node_name]
                execution_time = time.time() - start_time
                self.metrics["node_timings"][event.node_name] = execution_time
                self.metrics["total_execution_time"] += execution_time

            self.metrics["events_received"].append(f"COMPLETED: {event.node_name}")
            print(f"   📊 Metrics: {event.node_name} completed in {event.execution_time:.3f}s")

        elif isinstance(event, NodeFailedEvent):
            self.metrics["nodes_failed"] += 1
            self.metrics["events_received"].append(f"FAILED: {event.node_name}")
            print(f"   📊 Metrics: {event.node_name} failed")

    def get_summary(self) -> dict:
        """Get metrics summary."""
        return {
            "execution_summary": {
                "total_nodes": self.metrics["nodes_started"],
                "successful_nodes": self.metrics["nodes_completed"],
                "failed_nodes": self.metrics["nodes_failed"],
                "success_rate": self.metrics["nodes_completed"]
                / max(1, self.metrics["nodes_started"]),
            },
            "performance": {
                "total_time": round(self.metrics["total_execution_time"], 3),
                "average_node_time": round(
                    self.metrics["total_execution_time"] / max(1, self.metrics["nodes_completed"]),
                    3,
                ),
                "fastest_node": (
                    min(self.metrics["node_timings"].items(), key=lambda x: x[1])
                    if self.metrics["node_timings"]
                    else None
                ),
                "slowest_node": (
                    max(self.metrics["node_timings"].items(), key=lambda x: x[1])
                    if self.metrics["node_timings"]
                    else None
                ),
            },
            "detailed_timings": self.metrics["node_timings"],
        }


class ProgressTracker(Observer):
    """Observer that tracks overall progress."""

    def __init__(self, total_nodes: int):
        """Initialize progress tracking."""
        self.total_nodes = total_nodes
        self.completed_nodes = 0
        self.start_time = None

    def can_handle(self, event: PipelineEvent) -> bool:
        """Check if this observer can handle the event - we handle all events."""
        return True

    async def handle(self, event: PipelineEvent) -> None:
        """Handle pipeline events."""
        if isinstance(event, PipelineStartedEvent):
            self.start_time = time.time()
            print(f"   🚀 Progress: Pipeline started with {self.total_nodes} nodes")

        elif isinstance(event, NodeCompletedEvent):
            self.completed_nodes += 1
            progress_percent = (self.completed_nodes / self.total_nodes) * 100

            elapsed = time.time() - (self.start_time or time.time())
            estimated_total = (
                elapsed * (self.total_nodes / self.completed_nodes)
                if self.completed_nodes > 0
                else 0
            )
            remaining = max(0, estimated_total - elapsed)

            print(
                f"   🔄 Progress: {progress_percent:.1f}% ({self.completed_nodes}/{self.total_nodes}) - ETA: {remaining:.1f}s"
            )


class ErrorLogger(Observer):
    """Observer that logs errors and warnings."""

    def __init__(self):
        """Initialize error logging."""
        self.errors = []
        self.warnings = []

    def can_handle(self, event: PipelineEvent) -> bool:
        """Check if this observer can handle the event - we handle all events."""
        return True

    async def handle(self, event: PipelineEvent) -> None:
        """Handle pipeline events."""
        if isinstance(event, NodeFailedEvent):
            error_info = {
                "node": event.node_name,
                "timestamp": time.time(),
                "error": str(event.error),
            }
            self.errors.append(error_info)
            print(f"   ❌ Error: {event.node_name} failed - {error_info['error']}")

    def get_error_summary(self) -> dict:
        """Get error summary."""
        return {
            "total_errors": len(self.errors),
            "total_warnings": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
        }


# Sample processing functions with different execution times
async def fast_task(input_data: Any, **kwargs) -> dict:
    """Quick processing task."""
    await asyncio.sleep(0.1)
    return {"data": input_data, "processing_time": "fast", "result": "processed_quickly"}


async def medium_task(input_data: dict, **kwargs) -> dict:
    """Medium duration task."""
    await asyncio.sleep(0.3)
    original = input_data.get("data", "")
    return {"data": f"medium_{original}", "processing_time": "medium", "details": input_data}


async def slow_task(input_data: dict, **kwargs) -> dict:
    """Slower processing task."""
    await asyncio.sleep(0.5)
    return {
        "data": "slow_processing_complete",
        "processing_time": "slow",
        "input_summary": input_data,
    }


async def aggregator_task(input_data: Any, **kwargs) -> dict:
    """Combine results from multiple nodes."""
    await asyncio.sleep(0.1)
    return {
        "aggregated_data": {
            "fast": input_data.get("fast_node", {}).get("data"),
            "medium": input_data.get("medium_node", {}).get("data"),
            "slow": input_data.get("slow_node", {}).get("data"),
        },
        "total_inputs": 3,
        "aggregation_complete": True,
    }


async def demonstrate_event_monitoring():
    """Demonstrate comprehensive event monitoring."""

    print("\n📊 Event System Demo: Multi-Observer Monitoring")
    print("=" * 55)

    # Create DAG
    graph = DirectedGraph()

    # Add nodes with different execution times
    graph.add(NodeSpec("fast_node", fast_task))
    graph.add(NodeSpec("medium_node", medium_task).after("fast_node"))
    graph.add(NodeSpec("slow_node", slow_task).after("fast_node"))
    graph.add(
        NodeSpec("aggregator", aggregator_task).after("medium_node", "slow_node", "fast_node")
    )

    total_nodes = len(graph.nodes)
    print(f"\n📋 Pipeline has {total_nodes} nodes with complex dependencies")

    # Create observers
    metrics = CustomMetricsObserver()
    progress = ProgressTracker(total_nodes)
    error_logger = ErrorLogger()
    logging_observer = LoggingObserver()

    # Create event manager
    event_manager = PipelineEventManager()
    event_manager.subscribe(metrics)
    event_manager.subscribe(progress)
    event_manager.subscribe(error_logger)
    event_manager.subscribe(logging_observer)

    # Create orchestrator
    orchestrator = Orchestrator(validator=coerce_validator())

    print("\n🔍 Observers attached:")
    print("   • CustomMetricsObserver - Performance tracking")
    print("   • ProgressTracker - Execution progress")
    print("   • ErrorLogger - Error and warning capture")
    print("   • LoggingObserver - Built-in logging")
    print("   • PipelineEventManager - Event management")

    # Execute pipeline
    print("\n🚀 Executing pipeline with real-time monitoring...")
    print("-" * 50)

    start_time = time.time()
    results = await orchestrator.run(
        graph, "sample_input_data", additional_ports={"event_manager": event_manager}
    )
    end_time = time.time()

    print("-" * 50)
    print(f"✅ Pipeline completed in {end_time - start_time:.3f} seconds")

    # Display comprehensive metrics
    print("\n📊 Detailed Metrics Report:")
    metrics_summary = metrics.get_summary()

    print("\n   Execution Summary:")
    exec_summary = metrics_summary["execution_summary"]
    print(f"   • Total nodes: {exec_summary['total_nodes']}")
    print(f"   • Successful: {exec_summary['successful_nodes']}")
    print(f"   • Failed: {exec_summary['failed_nodes']}")
    print(f"   • Success rate: {exec_summary['success_rate']:.2%}")

    print("\n   Performance Analysis:")
    perf = metrics_summary["performance"]
    print(f"   • Total execution time: {perf['total_time']}s")
    print(f"   • Average node time: {perf['average_node_time']}s")
    if perf["fastest_node"]:
        print(f"   • Fastest node: {perf['fastest_node'][0]} ({perf['fastest_node'][1]:.3f}s)")
    if perf["slowest_node"]:
        print(f"   • Slowest node: {perf['slowest_node'][0]} ({perf['slowest_node'][1]:.3f}s)")

    print("\n   Node Timings:")
    for node_name, timing in metrics_summary["detailed_timings"].items():
        print(f"   • {node_name}: {timing:.3f}s")

    # Error summary
    error_summary = error_logger.get_error_summary()
    print("\n   Error Summary:")
    print(f"   • Errors: {error_summary['total_errors']}")
    print(f"   • Warnings: {error_summary['total_warnings']}")

    # Final results
    print("\n📋 Final Results:")
    print(f"   • Aggregated data keys: {list(results['aggregator']['aggregated_data'].keys())}")
    print(f"   • Pipeline success: {results['aggregator']['aggregation_complete']}")

    return results, metrics_summary


async def main():
    """Demonstrate event system capabilities."""

    print("📊 Example 05: Event System and Monitoring")
    print("=" * 50)

    print("\n🎯 This example demonstrates:")
    print("   • Custom event observers")
    print("   • Real-time metrics collection")
    print("   • Progress tracking")
    print("   • Error logging and monitoring")
    print("   • Performance analysis")

    results, metrics = await demonstrate_event_monitoring()

    print("\n🎯 Key Concepts Learned:")
    print("   ✅ Event Observers - Monitor pipeline execution")
    print("   ✅ Metrics Collection - Track performance and timing")
    print("   ✅ Progress Tracking - Real-time execution progress")
    print("   ✅ Error Handling - Capture and log failures")
    print("   ✅ Performance Analysis - Identify bottlenecks")

    print("\n🔗 Next: Run example 06 to learn about ports and adapters!")


if __name__ == "__main__":
    asyncio.run(main())
