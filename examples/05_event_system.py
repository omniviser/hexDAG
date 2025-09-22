"""Example demonstrating the event system with basic observers."""

import asyncio
import logging

from hexai.core.application.events import (
    CollectingObserver,
    CompositeObserver,
    Event,
    FilteringObserver,
    LoggingObserver,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    PipelineCompleted,
    PipelineStarted,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# Example custom observer
class MetricsObserver:
    """Observer that collects metrics."""

    def __init__(self):
        self.node_count = 0
        self.total_duration = 0.0
        self.failed_nodes = []

    async def handle(self, event: Event) -> None:
        """Handle events for metrics collection."""
        if isinstance(event, NodeStarted):
            self.node_count += 1
            print(f"  Metrics: Node '{event.name}' started (total: {self.node_count})")
        elif isinstance(event, NodeCompleted):
            self.total_duration += event.duration_ms
            print(f"  Metrics: Added {event.duration_ms}ms to total duration")
        elif isinstance(event, NodeFailed):
            self.failed_nodes.append(event.name)
            print(f"  Metrics: Node '{event.name}' failed")


# Example custom observer for debugging
class DebugObserver:
    """Observer that provides detailed debugging information."""

    async def handle(self, event: Event) -> None:
        """Print detailed event information."""
        event_type = event.__class__.__name__
        timestamp = event.timestamp.strftime("%H:%M:%S.%f")[:-3]

        if isinstance(event, NodeStarted):
            deps = f" (deps: {', '.join(event.dependencies)})" if event.dependencies else ""
            print(f"  [DEBUG {timestamp}] {event_type}: '{event.name}'{deps}")
        elif isinstance(event, NodeCompleted):
            print(f"  [DEBUG {timestamp}] {event_type}: '{event.name}' -> {event.result}")
        elif isinstance(event, NodeFailed):
            print(f"  [DEBUG {timestamp}] {event_type}: '{event.name}' ERROR: {event.error}")
        else:
            print(f"  [DEBUG {timestamp}] {event_type}")


async def simulate_pipeline_execution():
    """Simulate a pipeline execution with various observers."""

    print("Event System Demo with Observers\n" + "=" * 50)

    # Create various observers
    logger = logging.getLogger("pipeline")
    logging_observer = LoggingObserver(logger, logging.INFO)

    # Collecting observer for testing
    collector = CollectingObserver()

    # Custom metrics observer
    metrics = MetricsObserver()

    # Debug observer
    debug = DebugObserver()

    # Filter to only observe failures
    failure_logger = LoggingObserver(logging.getLogger("failures"), logging.ERROR)
    failure_filter = FilteringObserver(failure_logger, {NodeFailed})

    # Composite observer that notifies all observers
    composite = CompositeObserver(
        [
            logging_observer,
            collector,
            metrics,
            debug,
            failure_filter,
        ]
    )

    print("\n1. Simulating Pipeline Start:")
    print("-" * 30)
    pipeline_start = PipelineStarted(name="data_pipeline", total_waves=3, total_nodes=5)
    await composite.handle(pipeline_start)

    print("\n2. Simulating Normal Node Execution:")
    print("-" * 30)
    # Node 1 starts
    node1_start = NodeStarted(name="load_data", wave_index=1, dependencies=[])
    await composite.handle(node1_start)

    # Simulate some work
    await asyncio.sleep(0.1)

    # Node 1 completes
    node1_complete = NodeCompleted(
        name="load_data", wave_index=1, result={"records": 1000}, duration_ms=105.5
    )
    await composite.handle(node1_complete)

    print("\n3. Simulating Parallel Node Execution:")
    print("-" * 30)
    # Two nodes start in parallel
    node2_start = NodeStarted(name="validate_data", wave_index=2, dependencies=["load_data"])
    node3_start = NodeStarted(name="transform_data", wave_index=2, dependencies=["load_data"])

    await asyncio.gather(composite.handle(node2_start), composite.handle(node3_start))

    # Complete them
    node2_complete = NodeCompleted(
        name="validate_data", wave_index=2, result={"valid": 950, "invalid": 50}, duration_ms=50.0
    )
    node3_complete = NodeCompleted(
        name="transform_data", wave_index=2, result={"transformed": 1000}, duration_ms=200.0
    )

    await asyncio.gather(composite.handle(node2_complete), composite.handle(node3_complete))

    print("\n4. Simulating Node Failure:")
    print("-" * 30)
    node4_fail = NodeFailed(
        name="api_call", wave_index=3, error=Exception("Connection timeout after 30s")
    )
    await composite.handle(node4_fail)

    print("\n5. Simulating Recovery Node:")
    print("-" * 30)
    node5_start = NodeStarted(name="fallback_cache", wave_index=3, dependencies=["transform_data"])
    await composite.handle(node5_start)

    node5_complete = NodeCompleted(
        name="fallback_cache",
        wave_index=3,
        result={"source": "cache", "records": 950},
        duration_ms=25.0,
    )
    await composite.handle(node5_complete)

    print("\n6. Pipeline Complete:")
    print("-" * 30)
    pipeline_complete = PipelineCompleted(
        name="data_pipeline",
        duration_ms=380.5,
        node_results={
            "load_data": {"records": 1000},
            "validate_data": {"valid": 950, "invalid": 50},
            "transform_data": {"transformed": 1000},
            "api_call": None,
            "fallback_cache": {"source": "cache", "records": 950},
        },
    )
    await composite.handle(pipeline_complete)

    print("\n" + "=" * 50)
    print("Summary Statistics:")
    print("-" * 30)
    print(f"Metrics: {metrics.node_count} nodes started")
    print(f"Total duration: {metrics.total_duration}ms")
    print(f"Failed nodes: {metrics.failed_nodes}")

    print(f"\nCollected events: {len(collector.events)} total")
    print(f"  - NodeStarted: {len(collector.get_events_by_type(NodeStarted))}")
    print(f"  - NodeCompleted: {len(collector.get_events_by_type(NodeCompleted))}")
    print(f"  - NodeFailed: {len(collector.get_events_by_type(NodeFailed))}")
    print(f"  - PipelineStarted: {len(collector.get_events_by_type(PipelineStarted))}")
    print(f"  - PipelineCompleted: {len(collector.get_events_by_type(PipelineCompleted))}")

    print("\nDemo complete!")


async def main():
    """Run the event system demonstration."""
    await simulate_pipeline_execution()


if __name__ == "__main__":
    asyncio.run(main())
