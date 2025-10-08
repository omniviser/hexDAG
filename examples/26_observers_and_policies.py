"""Comprehensive Observers and Policies Demo.

This example demonstrates the powerful combination of observers (for observability)
and policies (for execution control) in hexDAG.

Key Concepts:
- Observers: READ-ONLY monitoring - track what happens
- Policies: CONTROL execution - decide what should happen

This example shows:
1. Multiple observer types working together
2. Policy-based execution control
3. Real-world patterns like retry, circuit breaker, rate limiting
4. Monitoring performance, quality, and resources
5. How observers and policies complement each other
"""

import asyncio

from hexai.adapters.local import LocalObserverManager, LocalPolicyManager
from hexai.core.application.events import (
    ALL_EXECUTION_EVENTS,
    NODE_LIFECYCLE_EVENTS,
    AlertingObserver,
    DataQualityObserver,
    ExecutionTracerObserver,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    PerformanceMetricsObserver,
    PipelineCompleted,
    PipelineStarted,
    ResourceMonitorObserver,
    SimpleLoggingObserver,
)
from hexai.core.application.policies import (
    CircuitBreakerPolicy,
    ConditionalSkipPolicy,
    ExponentialBackoffPolicy,
    FallbackPolicy,
    PolicyContext,
    PolicyResponse,
    PolicySignal,
    RateLimitPolicy,
    RetryPolicy,
    SubscriberType,
)

# ==============================================================================
# CUSTOM POLICY EXAMPLES
# ==============================================================================


class PriorityNodePolicy:
    """Custom policy that prioritizes certain nodes."""

    def __init__(self, priority_nodes: list[str]):
        self.priority_nodes = set(priority_nodes)

    @property
    def name(self) -> str:
        return "priority_node_policy"

    @property
    def priority(self) -> int:
        return 1  # Very high priority

    async def evaluate(self, context: PolicyContext) -> PolicyResponse:
        """Give priority to specific nodes."""
        from hexai.core.application.events import NodeStarted

        if isinstance(context.event, NodeStarted) and context.node_id in self.priority_nodes:
            # In real impl, this would signal to scheduler
            return PolicyResponse(signal=PolicySignal.PROCEED, data={"priority": "high"})

        return PolicyResponse(signal=PolicySignal.PROCEED)


# ==============================================================================
# SIMULATION FUNCTIONS
# ==============================================================================


async def simulate_pipeline(
    observer_manager: LocalObserverManager,
    policy_manager: LocalPolicyManager,
) -> None:
    """Simulate a complex pipeline execution with various scenarios."""

    # Pipeline start
    pipeline_start = PipelineStarted(name="demo_pipeline", total_waves=8, total_nodes=8)
    await observer_manager.notify(pipeline_start)

    # Simulate nodes with different scenarios
    scenarios = [
        # (node_name, duration_ms, error, result)
        ("data_ingestion", 150.0, None, {"status": "success", "records": 1000}),
        ("data_validation", 200.0, None, {"status": "success", "valid": 950}),
        ("rate_limited_api", 100.0, None, {"status": "success", "data": [1, 2, 3]}),
        ("slow_transform", 1200.0, None, {"status": "success", "processed": 950}),  # Slow!
        ("flaky_service", None, Exception("Connection timeout"), None),  # Will fail
        ("test_node", 50.0, None, {"status": "success"}),  # Will be skipped by policy
        ("quality_check", 80.0, None, None),  # Will trigger data quality alert
        ("final_export", 180.0, None, {"status": "success", "exported": 950}),
    ]

    for idx, (node_name, duration, error, result) in enumerate(scenarios):
        print(f"\n{'=' * 60}")
        print(f"Processing node {idx + 1}/{len(scenarios)}: {node_name}")
        print(f"{'=' * 60}")

        # Node started event
        start_event = NodeStarted(name=node_name, wave_index=idx, dependencies=[])
        await observer_manager.notify(start_event)

        # Evaluate policies for node start
        policy_ctx = PolicyContext(
            event=start_event,
            dag_id="demo_pipeline",
            node_id=node_name,
            wave_index=idx,
            attempt=1,
        )
        policy_response = await policy_manager.evaluate(policy_ctx)

        print(f"Policy decision: {policy_response.signal.value}")
        if policy_response.data:
            print(f"Policy data: {policy_response.data}")

        # Check if node should be skipped
        if policy_response.signal == PolicySignal.SKIP:
            print(f"→ Node '{node_name}' skipped by policy")
            continue

        # Simulate execution time
        if duration:
            await asyncio.sleep(duration / 1000.0)  # Convert ms to seconds

        # Node completion or failure
        if error:
            # Simulate retries for failed nodes
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                fail_event = NodeFailed(name=node_name, wave_index=idx, error=error)
                await observer_manager.notify(fail_event)

                # Evaluate retry policy
                retry_ctx = PolicyContext(
                    event=fail_event,
                    dag_id="demo_pipeline",
                    node_id=node_name,
                    wave_index=idx,
                    attempt=attempt,
                    error=error,
                )
                retry_response = await policy_manager.evaluate(retry_ctx)

                print(f"  Attempt {attempt} failed: {error}")
                print(f"  Policy decision: {retry_response.signal.value}")

                if retry_response.signal == PolicySignal.RETRY and attempt < max_retries:
                    print(f"  → Retrying... (attempt {attempt + 1})")
                    await asyncio.sleep(0.1)  # Small delay between retries
                elif retry_response.signal == PolicySignal.FALLBACK:
                    print("  → Using fallback value")
                    complete_event = NodeCompleted(
                        name=node_name,
                        wave_index=idx,
                        result=retry_response.data,
                        duration_ms=duration or 0.0,
                    )
                    await observer_manager.notify(complete_event)
                    break
                else:
                    print("  → Max retries reached or policy prevented retry")
                    break
        else:
            # Success
            complete_event = NodeCompleted(
                name=node_name, wave_index=idx, result=result, duration_ms=duration or 0.0
            )
            await observer_manager.notify(complete_event)

    # Pipeline completion
    pipeline_complete = PipelineCompleted(name="demo_pipeline", duration_ms=2500.0)
    await observer_manager.notify(pipeline_complete)


# ==============================================================================
# MAIN DEMO
# ==============================================================================


async def main():
    """Demonstrate comprehensive observer and policy patterns."""
    print("\n" + "=" * 70)
    print("HEXDAG OBSERVERS AND POLICIES - COMPREHENSIVE DEMO")
    print("=" * 70)

    # ==============================================================================
    # 1. SETUP MANAGERS
    # ==============================================================================

    print("\n[1] Setting up Observer and Policy Managers...")

    observer_manager = LocalObserverManager(
        max_concurrent_observers=20,
        observer_timeout=10.0,
        use_weak_refs=True,
    )

    policy_manager = LocalPolicyManager()

    # ==============================================================================
    # 2. REGISTER OBSERVERS (OBSERVABILITY)
    # ==============================================================================

    print("\n[2] Registering Observers for Observability...")

    # Core monitoring observers with EVENT FILTERING for ~90% performance boost
    metrics_observer = PerformanceMetricsObserver()
    observer_manager.register(
        metrics_observer.handle,
        event_types=ALL_EXECUTION_EVENTS,  # Only pipeline, wave, and node events
    )
    print("  ✓ Performance Metrics Observer (filtered: ALL_EXECUTION_EVENTS)")

    alerting_observer = AlertingObserver(
        slow_threshold_ms=1000.0,
        on_alert=lambda alert: print(f"    [ALERT] {alert.type.value}"),  # Alert is dataclass
    )
    observer_manager.register(
        alerting_observer.handle,
        event_types=[NodeCompleted, NodeFailed],  # Only need these two
    )
    print("  ✓ Alerting Observer (filtered: NodeCompleted, NodeFailed)")

    tracer = ExecutionTracerObserver()
    observer_manager.register(tracer.handle)  # No filter - captures everything
    print("  ✓ Execution Tracer (captures all events)")

    logger_obs = SimpleLoggingObserver(verbose=False)
    observer_manager.register(logger_obs.handle, event_types=ALL_EXECUTION_EVENTS)
    print("  ✓ Simple Logging Observer (filtered: ALL_EXECUTION_EVENTS)")

    resource_monitor = ResourceMonitorObserver()
    observer_manager.register(
        resource_monitor.handle,
        event_types=NODE_LIFECYCLE_EVENTS,  # Only node lifecycle events
    )
    print("  ✓ Resource Monitor (filtered: NODE_LIFECYCLE_EVENTS)")

    quality_observer = DataQualityObserver()
    observer_manager.register(
        quality_observer.handle,
        event_types=[NodeCompleted],  # Only need completed nodes
    )
    print("  ✓ Data Quality Observer (filtered: NodeCompleted only)")

    print(f"\nRegistered {len(observer_manager)} observers")

    # ==============================================================================
    # 3. REGISTER POLICIES (EXECUTION CONTROL)
    # ==============================================================================

    print("\n[3] Registering Policies for Execution Control...")

    # Retry policy for failures
    retry_policy = RetryPolicy(max_retries=3)
    policy_manager.subscribe(retry_policy, SubscriberType.CORE)
    print("  ✓ Retry Policy (max 3 retries)")

    # Circuit breaker to prevent cascading failures
    circuit_breaker = CircuitBreakerPolicy(failure_threshold=5)
    policy_manager.subscribe(circuit_breaker, SubscriberType.CORE)
    print("  ✓ Circuit Breaker Policy (threshold: 5 failures)")

    # Fallback for failed services
    fallback_policy = FallbackPolicy(
        fallback_value={"status": "fallback", "data": [], "message": "Using cached data"}
    )
    policy_manager.subscribe(fallback_policy, SubscriberType.USER)
    print("  ✓ Fallback Policy (with cached data)")

    # Skip test nodes in production
    def skip_test_nodes(ctx: PolicyContext) -> bool:
        return "test" in ctx.node_id.lower()

    skip_policy = ConditionalSkipPolicy(should_skip=skip_test_nodes, policy_name="skip_tests")
    policy_manager.subscribe(skip_policy, SubscriberType.USER)
    print("  ✓ Conditional Skip Policy (skip test nodes)")

    # Rate limit API calls
    rate_limiter = RateLimitPolicy(max_executions=3, window_seconds=5.0)
    policy_manager.subscribe(rate_limiter, SubscriberType.USER)
    print("  ✓ Rate Limit Policy (3 calls per 5 seconds)")

    # Exponential backoff for retries
    backoff_policy = ExponentialBackoffPolicy(
        max_retries=3, initial_delay_ms=50.0, backoff_factor=2.0
    )
    policy_manager.subscribe(backoff_policy, SubscriberType.USER)
    print("  ✓ Exponential Backoff Policy")

    # Custom priority policy
    priority_policy = PriorityNodePolicy(priority_nodes=["data_ingestion", "final_export"])
    policy_manager.subscribe(priority_policy, SubscriberType.USER)
    print("  ✓ Priority Node Policy (prioritizes critical nodes)")

    print(f"\nRegistered {len(policy_manager)} policies")

    # ==============================================================================
    # 4. RUN SIMULATION
    # ==============================================================================

    print("\n[4] Running Pipeline Simulation...\n")

    await simulate_pipeline(observer_manager, policy_manager)

    # ==============================================================================
    # 5. ANALYZE RESULTS
    # ==============================================================================

    print("\n" + "=" * 70)
    print("[5] ANALYSIS & RESULTS")
    print("=" * 70)

    # Performance Metrics
    print("\n📊 PERFORMANCE METRICS:")
    print("-" * 70)
    metrics = metrics_observer.get_summary()
    print(f"Total nodes executed:     {metrics['total_nodes_executed']}")
    print(f"Unique nodes:             {metrics['unique_nodes']}")
    print(f"Total duration:           {metrics['total_duration_ms']:.1f}ms")
    print(f"Success rate:             {metrics['overall_success_rate']:.1f}%")
    print(f"Total failures:           {metrics['total_failures']}")

    print("\nNode-by-node timing:")
    for node, avg_time in metrics["average_timings_ms"].items():
        min_time = metrics["min_timings_ms"].get(node, 0)
        max_time = metrics["max_timings_ms"].get(node, 0)
        print(
            f"  {node:25s} avg: {avg_time:6.1f}ms  min: {min_time:6.1f}ms  max: {max_time:6.1f}ms"
        )

    # Alerts (now using Alert dataclass)
    print("\n🚨 ALERTS TRIGGERED:")
    print("-" * 70)
    alerts = alerting_observer.get_alerts()
    if alerts:
        for alert in alerts:
            print(f"  [{alert.type.value:15s}] {alert.message}")
    else:
        print("  No alerts triggered")

    # Data Quality (now using Alert dataclass)
    print("\n🔍 DATA QUALITY ISSUES:")
    print("-" * 70)
    if quality_observer.has_issues():
        for issue in quality_observer.get_issues():
            severity = issue.severity.value.upper()
            print(f"  [{severity}] {issue.node}: {issue.message}")
    else:
        print("  No quality issues detected")

    # Resource Usage
    print("\n💻 RESOURCE USAGE:")
    print("-" * 70)
    resource_stats = resource_monitor.get_stats()
    print(f"  Max concurrent nodes:     {resource_stats['max_concurrent']}")
    print(f"  Total waves:              {resource_stats['total_waves']}")
    print(f"  Average wave size:        {resource_stats['avg_wave_size']:.1f} nodes")

    # Execution Trace (first 15 events)
    print("\n📝 EXECUTION TRACE (First 15 events):")
    print("-" * 70)
    tracer.print_trace(max_events=15)

    # ==============================================================================
    # 6. KEY TAKEAWAYS
    # ==============================================================================

    print("\n" + "=" * 70)
    print("KEY FEATURES DEMONSTRATED:")
    print("=" * 70)
    print("\n🔭 OBSERVERS (Observability - Read-Only):")
    print("  ✓ Performance metrics collection")
    print("  ✓ Real-time alerting on thresholds")
    print("  ✓ Execution tracing for debugging")
    print("  ✓ Data quality monitoring")
    print("  ✓ Resource usage tracking")
    print("  ✓ Fault isolation - observer failures don't crash pipeline")

    print("\n🎛️  POLICIES (Control - Execution Decisions):")
    print("  ✓ Retry logic with exponential backoff")
    print("  ✓ Circuit breaker pattern")
    print("  ✓ Fallback values for failures")
    print("  ✓ Conditional node skipping")
    print("  ✓ Rate limiting")
    print("  ✓ Priority-based execution")
    print("  ✓ Composable policy chains")

    print("\n💡 BEST PRACTICES:")
    print("  ✓ Observers = WHAT HAPPENED (monitoring)")
    print("  ✓ Policies = WHAT SHOULD HAPPEN (control)")
    print("  ✓ Observers never affect execution")
    print("  ✓ Policies make execution decisions")
    print("  ✓ Both are composable and reusable")
    print("  ✓ Clean separation of concerns")

    print("\n" + "=" * 70)

    # Cleanup
    await observer_manager.close()
    print("\n✓ Observer manager closed")
    print("✓ Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
