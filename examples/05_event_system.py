#!/usr/bin/env python3
"""
📊 Example 05: Clean Event System and Monitoring.

This example demonstrates the NEW clean event system:
- Simple event data classes (NodeStarted, NodeCompleted, etc.)
- ObserverManager for logging and metrics (fire-and-forget)
- ControlBus for execution control (can veto)
- Custom observers and control handlers
- Real-time execution tracking

Run: python examples/05_event_system.py
"""

import asyncio
import time
from typing import Any

from hexai.core.application.events import (
    EventBus,
    LoggingObserver,
    MetricsObserver,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    ObserverManager,
    PipelineCompleted,
    PipelineStarted,
)
from hexai.core.application.orchestrator import Orchestrator
from hexai.core.domain.dag import DirectedGraph, NodeSpec
from hexai.core.validation import coerce_validator


class CustomMetricsObserver:
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
        self.pipeline_start_time = None

    async def handle(self, event: Any) -> None:
        """Handle pipeline events."""
        # Track all events
        self.metrics["events_received"].append(
            {"type": event.__class__.__name__, "time": time.time()}
        )

        # Handle specific event types
        match event:
            case PipelineStarted():
                self.pipeline_start_time = time.time()
                print(f"📊 Pipeline started: {event.name}")

            case NodeStarted(name=name):
                self.metrics["nodes_started"] += 1
                self.node_start_times[name] = time.time()
                print(f"  ▶️  Node '{name}' started")

            case NodeCompleted(name=name, duration_ms=duration):
                self.metrics["nodes_completed"] += 1
                self.metrics["node_timings"][name] = duration / 1000
                print(f"  ✅ Node '{name}' completed in {duration / 1000:.2f}s")

            case NodeFailed(name=name, error=error):
                self.metrics["nodes_failed"] += 1
                print(f"  ❌ Node '{name}' failed: {error}")

            case PipelineCompleted():
                if self.pipeline_start_time:
                    self.metrics["total_execution_time"] = time.time() - self.pipeline_start_time
                print(f"📊 Pipeline completed in {self.metrics['total_execution_time']:.2f}s")

    def get_summary(self):
        """Get execution summary."""
        return {
            "total_events": len(self.metrics["events_received"]),
            "nodes_started": self.metrics["nodes_started"],
            "nodes_completed": self.metrics["nodes_completed"],
            "nodes_failed": self.metrics["nodes_failed"],
            "execution_time": self.metrics["total_execution_time"],
            "avg_node_time": (
                sum(self.metrics["node_timings"].values()) / len(self.metrics["node_timings"])
                if self.metrics["node_timings"]
                else 0
            ),
        }


class RateLimitHandler:
    """Control handler that limits execution rate."""

    def __init__(self, max_nodes_per_second: float = 5):
        """Initialize with rate limit."""
        self.max_nodes_per_second = max_nodes_per_second
        self.last_node_time = 0

    async def check(self, event: Any) -> bool:
        """Check if we should allow this event based on rate limiting."""
        if isinstance(event, NodeStarted):
            current_time = time.time()
            time_since_last = current_time - self.last_node_time
            min_interval = 1.0 / self.max_nodes_per_second

            if time_since_last < min_interval:
                # Too fast, would veto but for demo we just log
                print(f"  ⚠️  Rate limit: {event.name} (interval: {time_since_last:.3f}s)")

            self.last_node_time = current_time

        # Always allow for demo
        return True


# Example node functions
async def fetch_data(input_data: dict[str, Any], **_kwargs) -> dict[str, Any]:
    """Simulate fetching data."""
    await asyncio.sleep(0.5)  # Simulate API call
    return {"data": f"Fetched data for {input_data.get('query', 'default')}", "count": 100}


async def process_data(input_data: dict[str, Any], **_kwargs) -> dict[str, Any]:
    """Simulate processing data."""
    await asyncio.sleep(0.3)
    count = input_data.get("count", 0)
    return {"processed": f"Processed {count} items", "score": count * 1.5}


async def analyze_results(input_data: dict[str, Any], **_kwargs) -> dict[str, Any]:
    """Simulate analyzing results."""
    await asyncio.sleep(0.2)
    score = input_data.get("score", 0)
    return {"analysis": f"Analysis complete, score: {score}", "final_score": score * 1.1}


async def failing_node(input_data: dict[str, Any], **_kwargs) -> dict[str, Any]:
    """Simulate a failing node."""
    await asyncio.sleep(0.1)
    raise RuntimeError("Simulated failure in node")


async def generate_report(input_data: dict[str, Any], **_kwargs) -> dict[str, Any]:
    """Generate final report."""
    await asyncio.sleep(0.4)
    # Access outputs from multiple nodes
    final_score = input_data.get("final_score", 0)
    processed = input_data.get("processed", "No data")
    return {"report": f"Report: {processed}, Final score: {final_score:.1f}"}


async def main():
    """Demonstrate the event system."""
    print("=" * 60)
    print("CLEAN EVENT SYSTEM DEMO")
    print("=" * 60)

    # Create observers
    observers = ObserverManager()

    # Add built-in observers
    observers.attach(LoggingObserver())
    observers.attach(MetricsObserver())

    # Add custom observer
    custom_metrics = CustomMetricsObserver()
    observers.attach(custom_metrics)

    # Create control bus with rate limiter
    control = EventBus()
    control.register(RateLimitHandler(max_nodes_per_second=10))

    # Build a pipeline
    graph = DirectedGraph()

    # Add nodes
    graph.add(NodeSpec(name="fetch", fn=fetch_data))
    graph.add(NodeSpec(name="process", fn=process_data, deps={"fetch"}))
    graph.add(NodeSpec(name="analyze", fn=analyze_results, deps={"process"}))
    # Add a parallel branch
    graph.add(NodeSpec(name="validate", fn=process_data, deps={"fetch"}))
    # Node that depends on multiple inputs
    graph.add(NodeSpec(name="report", fn=generate_report, deps={"analyze", "validate"}))

    # Create orchestrator with validation
    orchestrator = Orchestrator(
        max_concurrent_nodes=3,
        validator=coerce_validator(),
        ports={
            "observers": observers,  # For observability
            "control_bus": control,  # For execution control
        },
    )

    print("\n🚀 Running pipeline with event monitoring...\n")

    # Run the pipeline
    results = await orchestrator.run(graph, {"query": "test data"}, validate=False)

    # Show results
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)
    for node_name, result in results.items():
        print(f"{node_name}: {result}")

    # Show metrics
    print("\n" + "=" * 60)
    print("EXECUTION METRICS")
    print("=" * 60)
    summary = custom_metrics.get_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    # Demonstrate pipeline with failure
    print("\n" + "=" * 60)
    print("PIPELINE WITH FAILURE DEMO")
    print("=" * 60)

    # Add a failing node
    graph2 = DirectedGraph()
    graph2.add(NodeSpec(name="fetch", fn=fetch_data))
    graph2.add(NodeSpec(name="fail", fn=failing_node, deps={"fetch"}))
    graph2.add(NodeSpec(name="process", fn=process_data, deps={"fail"}))

    print("\n🚀 Running pipeline with failing node...\n")

    try:
        await orchestrator.run(graph2, {"query": "test"}, validate=False)
    except Exception as e:
        print(f"\n⚠️  Pipeline failed as expected: {e}")

    print("\n✅ Event system demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
