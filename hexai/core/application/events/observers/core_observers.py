"""Core observer implementations for common use cases.

This module provides ready-to-use observer implementations that demonstrate
best practices and common patterns for event observation in hexDAG.

All observers follow these principles:
- READ-ONLY: Observers never modify execution or state
- FAULT-ISOLATED: Observer failures don't affect pipeline execution
- ASYNC-FIRST: All observers support async operation
- TYPE-SAFE: Proper type hints and Pydantic validation where applicable
- EVENT FILTERING: Use event_types at registration for performance
- FRAMEWORK FEATURES: Leverage built-in event taxonomy and helpers
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hexai.core.application.events.events import (
    Event,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    PipelineCompleted,
    PipelineStarted,
)
from hexai.core.application.events.observers.models import (
    Alert,
    AlertSeverity,
    AlertType,
    NodeMetrics,
)
from hexai.core.logging import get_logger
from hexai.core.registry.decorators import observer

logger = get_logger(__name__)


# ==============================================================================
# METRICS AND MONITORING OBSERVERS
# ==============================================================================


@observer(name="performance_metrics", description="Collect detailed performance metrics")
class PerformanceMetricsObserver:
    """Observer that collects comprehensive performance metrics.

    This observer tracks:
    - Node execution counts and timings
    - Success/failure rates
    - Average, min, max execution times per node
    - Total pipeline duration

    Uses consolidated NodeMetrics dataclass following the HandlerEntry pattern
    for efficient storage and computation.

    Example
    -------
        >>> from hexai.adapters.local import LocalObserverManager  # doctest: +SKIP
        >>> from hexai.core.application.events import (  # doctest: +SKIP
        ...     PerformanceMetricsObserver,
        ...     ALL_EXECUTION_EVENTS,
        ... )
        >>> observer_manager = LocalObserverManager()  # doctest: +SKIP
        >>> metrics = PerformanceMetricsObserver()  # doctest: +SKIP
        >>> # Register with event filtering for ~90% performance improvement
        >>> observer_manager.register(  # doctest: +SKIP
        ...     metrics.handle,
        ...     event_types=ALL_EXECUTION_EVENTS
        ... )  # doctest: +SKIP
        >>> # ... run pipeline ...
        >>> print(metrics.get_summary())  # doctest: +SKIP
    """

    def __init__(self) -> None:
        """Initialize the performance metrics observer."""
        # Consolidated storage using NodeMetrics dataclass
        self.metrics: dict[str, NodeMetrics] = {}
        self.total_nodes = 0
        self.total_duration_ms = 0.0
        self.pipeline_start_times: dict[str, float] = {}
        self.pipeline_end_times: dict[str, float] = {}

    async def handle(self, event: Event) -> None:
        """Handle performance-related events.

        Note: Should be registered with event_types=ALL_EXECUTION_EVENTS
        for optimal performance. Event filtering at registration provides
        ~90% reduction in unnecessary handler invocations.

        Parameters
        ----------
        event : Event
            The event to process
        """
        if isinstance(event, PipelineStarted):
            self.pipeline_start_times[event.name] = event.timestamp.timestamp()

        elif isinstance(event, NodeStarted):
            # Initialize metrics for new nodes
            if event.name not in self.metrics:
                self.metrics[event.name] = NodeMetrics()
            self.metrics[event.name].executions += 1
            self.total_nodes += 1

        elif isinstance(event, NodeCompleted):
            if event.name not in self.metrics:
                self.metrics[event.name] = NodeMetrics()
            self.metrics[event.name].timings.append(event.duration_ms)
            self.total_duration_ms += event.duration_ms

        elif isinstance(event, NodeFailed):
            if event.name not in self.metrics:
                self.metrics[event.name] = NodeMetrics()
            self.metrics[event.name].failures += 1

        elif isinstance(event, PipelineCompleted):
            self.pipeline_end_times[event.name] = event.timestamp.timestamp()

    def get_summary(self) -> dict[str, Any]:
        """Generate comprehensive metrics summary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing performance metrics including:
            - total_nodes_executed: Total number of nodes executed
            - unique_nodes: Number of unique node types
            - total_duration_ms: Total execution time across all nodes
            - average_timings_ms: Average execution time per node
            - min_timings_ms: Minimum execution time per node
            - max_timings_ms: Maximum execution time per node
            - node_executions: Execution count per node
            - failures: Failure count per node
            - success_rates: Success rate per node (percentage)
            - total_failures: Total failures across all nodes
            - overall_success_rate: Overall success rate percentage
        """
        avg_timings, min_timings, max_timings = {}, {}, {}
        node_executions, failures, success_rates = {}, {}, {}

        for node, m in self.metrics.items():
            avg_timings[node] = m.average_ms
            min_timings[node] = m.min_ms
            max_timings[node] = m.max_ms
            node_executions[node] = m.executions
            failures[node] = m.failures
            success_rates[node] = m.success_rate

        total_failures = sum(failures.values())
        overall_success_rate = (
            (self.total_nodes - total_failures) / self.total_nodes * 100
            if self.total_nodes > 0
            else 0.0
        )

        return {
            "total_nodes_executed": self.total_nodes,
            "unique_nodes": len(self.metrics),
            "total_duration_ms": self.total_duration_ms,
            "average_timings_ms": avg_timings,
            "min_timings_ms": min_timings,
            "max_timings_ms": max_timings,
            "node_executions": node_executions,
            "failures": failures,
            "success_rates": success_rates,
            "total_failures": total_failures,
            "overall_success_rate": overall_success_rate,
        }

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        self.metrics.clear()
        self.total_nodes = 0
        self.total_duration_ms = 0.0
        self.pipeline_start_times.clear()
        self.pipeline_end_times.clear()


@observer(name="alerting", description="Trigger alerts based on performance thresholds")
class AlertingObserver:
    """Observer that triggers alerts based on configurable thresholds.

    This observer monitors execution and triggers alerts when:
    - Node execution exceeds time threshold (slow node alert)
    - Node fails (failure alert)
    - Custom conditions are met via callback

    Uses typed Alert dataclass for type safety and validation.

    Parameters
    ----------
    slow_threshold_ms : float
        Threshold in milliseconds for slow node alert (default: 1000.0)
    on_alert : callable, optional
        Callback function(Alert) called when alert is triggered

    Example
    -------
        >>> from hexai.core.application.events import NODE_LIFECYCLE_EVENTS
        >>> def handle_alert(alert: Alert):
        ...     print(f"ALERT: {alert.message}")
        ...     # Send to monitoring system, etc.
        >>> alerting = AlertingObserver(slow_threshold_ms=500.0, on_alert=handle_alert)
        >>> # Register with event filtering for performance
        >>> observer_manager.register(  # doctest: +SKIP
        ...     alerting.handle,
        ...     event_types=[NodeCompleted, NodeFailed]
        ... )
        >>> # Check alerts programmatically
        >>> alerts = alerting.get_alerts()
    """

    def __init__(
        self,
        slow_threshold_ms: float = 1000.0,
        on_alert: Callable[[Alert], None] | None = None,
    ):
        """Initialize alerting observer.

        Parameters
        ----------
        slow_threshold_ms : float
            Millisecond threshold for slow node warnings
        on_alert : callable, optional
            Function to call when alert is triggered with Alert object
        """
        self.slow_threshold = slow_threshold_ms
        self.on_alert = on_alert
        self.alerts: list[Alert] = []

    async def handle(self, event: Event) -> None:
        """Monitor events and trigger alerts.

        Note: Should be registered with event_types=[NodeCompleted, NodeFailed]
        for optimal performance.

        Parameters
        ----------
        event : Event
            Event to monitor
        """
        alert: Alert | None = None

        if isinstance(event, NodeCompleted):
            if event.duration_ms > self.slow_threshold:
                alert = Alert(
                    type=AlertType.SLOW_NODE,
                    node=event.name,
                    message=(
                        f"Node '{event.name}' took {event.duration_ms:.1f}ms "
                        f"(threshold: {self.slow_threshold}ms)"
                    ),
                    timestamp=event.timestamp.timestamp(),
                    severity=AlertSeverity.WARNING,
                    duration_ms=event.duration_ms,
                    threshold_ms=self.slow_threshold,
                )
                logger.warning(alert.message)

        elif isinstance(event, NodeFailed):
            alert = Alert(
                type=AlertType.NODE_FAILURE,
                node=event.name,
                message=f"Node '{event.name}' failed: {event.error}",
                timestamp=event.timestamp.timestamp(),
                severity=AlertSeverity.ERROR,
                error=str(event.error),
            )
            logger.error(alert.message)

        # Store and callback if alert was triggered
        if alert:
            self.alerts.append(alert)
            if self.on_alert:
                try:
                    self.on_alert(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")

    def get_alerts(
        self,
        alert_type: AlertType | None = None,
        severity: AlertSeverity | None = None,
    ) -> list[Alert]:
        """Get triggered alerts, optionally filtered by type or severity.

        Parameters
        ----------
        alert_type : AlertType, optional
            Filter alerts by type
        severity : AlertSeverity, optional
            Filter alerts by severity level

        Returns
        -------
        list[Alert]
            List of Alert objects matching the criteria
        """
        return [
            a
            for a in self.alerts
            if (alert_type is None or a.type == alert_type)
            and (severity is None or a.severity == severity)
        ]

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self.alerts.clear()


# ==============================================================================
# LOGGING AND DEBUGGING OBSERVERS
# ==============================================================================


@dataclass
class ExecutionTrace:
    """Execution trace with timing information.

    Uses event timestamps for precise, reproducible timing information
    instead of wall-clock time.
    """

    events: list[tuple[float, str, Event]] = field(default_factory=list)
    start_time: float | None = None

    def add(self, event: Event) -> None:
        """Add event to trace with elapsed time from first event."""
        event_timestamp = event.timestamp.timestamp()

        # Set start time from first event
        if self.start_time is None:
            self.start_time = event_timestamp

        elapsed_ms = (event_timestamp - self.start_time) * 1000
        event_type = type(event).__name__
        self.events.append((elapsed_ms, event_type, event))


@observer(name="execution_tracer", description="Build detailed execution traces for debugging")
class ExecutionTracerObserver:
    """Observer that builds detailed execution traces.

    Useful for debugging and understanding execution flow. Captures all events
    with precise timing information.

    Example
    -------
        >>> from hexai.adapters.local import LocalObserverManager
        >>> tracer = ExecutionTracerObserver()
        >>> observer_manager = LocalObserverManager()
        >>> observer_manager.register(tracer.handle)  # doctest: +SKIP
        >>> # ... run pipeline ...
        >>> tracer.print_trace()
    """

    def __init__(self) -> None:
        """Initialize execution tracer."""
        self.trace = ExecutionTrace()

    async def handle(self, event: Event) -> None:
        """Capture event in trace.

        Parameters
        ----------
        event : Event
            Event to capture
        """
        self.trace.add(event)

    def get_trace(self) -> ExecutionTrace:
        """Get the current execution trace.

        Returns
        -------
        ExecutionTrace
            The captured execution trace
        """
        return self.trace

    def print_trace(self, max_events: int | None = None) -> None:
        """Print the execution trace in a readable format.

        Parameters
        ----------
        max_events : int, optional
            Maximum number of events to print. If None, prints all events.
        """
        events_to_print = self.trace.events
        if max_events is not None:
            events_to_print = events_to_print[:max_events]

        for elapsed_ms, event_type, event in events_to_print:
            # Type narrowing: check if event has 'name' attribute
            event_name = getattr(event, "name", None)
            if event_name:
                print(f"[{elapsed_ms:7.1f}ms] {event_type:25s} | {event_name}")
            else:
                print(f"[{elapsed_ms:7.1f}ms] {event_type:25s}")

    def reset(self) -> None:
        """Reset the trace."""
        self.trace = ExecutionTrace()


@observer(name="simple_logger", description="Simple logging observer for basic monitoring")
class SimpleLoggingObserver:
    """Simple observer that logs events to console/logger.

    Provides basic logging of pipeline execution with optional verbose mode
    for detailed information. Leverages event.log_message() for consistent formatting.

    Parameters
    ----------
    verbose : bool
        If True, log detailed information including results and dependencies

    Example
    -------
        >>> from hexai.core.application.events import ALL_EXECUTION_EVENTS
        >>> logger_obs = SimpleLoggingObserver(verbose=True)
        >>> # Register with event filtering
        >>> observer_manager.register(  # doctest: +SKIP
        ...     logger_obs.handle,
        ...     event_types=ALL_EXECUTION_EVENTS
        ... )
    """

    def __init__(self, verbose: bool = False):
        """Initialize simple logging observer.

        Parameters
        ----------
        verbose : bool
            Enable verbose logging with additional details
        """
        self.verbose = verbose

    async def handle(self, event: Event) -> None:
        """Log events to console.

        Note: Should be registered with event_types=ALL_EXECUTION_EVENTS
        for optimal performance. Uses event.log_message() for consistent formatting.

        Parameters
        ----------
        event : Event
            Event to log
        """
        # Use built-in log_message() for consistent formatting
        if isinstance(event, NodeStarted):
            logger.info(event.log_message())
            if self.verbose:
                logger.debug(f"  Wave: {event.wave_index}, Dependencies: {event.dependencies}")

        elif isinstance(event, NodeCompleted):
            logger.info(event.log_message())
            if self.verbose and event.result is not None:
                result_preview = str(event.result)[:100]
                logger.debug(f"  Result: {result_preview}...")

        elif isinstance(event, NodeFailed):
            logger.error(event.log_message())
            if self.verbose:
                logger.error(f"  Error: {event.error}")

        elif isinstance(event, (PipelineStarted, PipelineCompleted)):
            logger.info(event.log_message())


# ==============================================================================
# RESOURCE AND QUALITY MONITORING OBSERVERS
# ==============================================================================


@observer(name="resource_monitor", description="Monitor resource usage and concurrency patterns")
class ResourceMonitorObserver:
    """Observer that monitors resource usage patterns.

    Tracks:
    - Concurrent node execution levels
    - Wave-based parallelism patterns
    - Maximum concurrency reached
    - Average wave sizes

    Example
    -------
        >>> from hexai.core.application.events import NODE_LIFECYCLE_EVENTS
        >>> resource_mon = ResourceMonitorObserver()
        >>> # Register with event filtering
        >>> observer_manager.register(  # doctest: +SKIP
        ...     resource_mon.handle,
        ...     event_types=NODE_LIFECYCLE_EVENTS
        ... )
        >>> # ... run pipeline ...
        >>> stats = resource_mon.get_stats()  # doctest: +SKIP
        >>> print(f"Max concurrency: {stats['max_concurrent']}")  # doctest: +SKIP
    """

    def __init__(self) -> None:
        """Initialize resource monitor."""
        self.concurrent_nodes = 0
        self.max_concurrent = 0
        self.wave_sizes: list[int] = []
        self.current_wave_nodes: set[str] = set()
        self.current_wave: int = -1

    async def handle(self, event: Event) -> None:
        """Track resource usage patterns.

        Note: Should be registered with event_types=NODE_LIFECYCLE_EVENTS
        for optimal performance.

        Parameters
        ----------
        event : Event
            Event to process
        """
        if isinstance(event, NodeStarted):
            self.concurrent_nodes += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent_nodes)
            self.current_wave_nodes.add(event.name)

            # Track wave transitions
            if event.wave_index != self.current_wave:
                if self.current_wave_nodes and self.current_wave >= 0:
                    self.wave_sizes.append(len(self.current_wave_nodes))
                    self.current_wave_nodes.clear()
                self.current_wave = event.wave_index

        elif isinstance(event, (NodeCompleted, NodeFailed)):
            self.concurrent_nodes = max(0, self.concurrent_nodes - 1)

    def get_stats(self) -> dict[str, Any]:
        """Get resource usage statistics.

        Returns
        -------
        dict[str, Any]
            Resource usage statistics including:
            - max_concurrent: Maximum concurrent nodes
            - wave_sizes: List of node counts per wave
            - total_waves: Total number of execution waves
            - avg_wave_size: Average nodes per wave
        """
        avg_wave_size = sum(self.wave_sizes) / len(self.wave_sizes) if self.wave_sizes else 0
        return {
            "max_concurrent": self.max_concurrent,
            "wave_sizes": self.wave_sizes.copy(),
            "total_waves": len(self.wave_sizes),
            "avg_wave_size": avg_wave_size,
        }

    def reset(self) -> None:
        """Reset resource monitoring state."""
        self.concurrent_nodes = 0
        self.max_concurrent = 0
        self.wave_sizes.clear()
        self.current_wave_nodes.clear()
        self.current_wave = -1


@observer(name="data_quality", description="Monitor data quality through the pipeline")
class DataQualityObserver:
    """Observer that monitors data quality in pipeline execution.

    Checks for common data quality issues:
    - None/null values
    - Empty collections (lists, dicts, strings)
    - Error indicators in result data

    Uses typed Alert dataclass for quality issues.

    Example
    -------
        >>> quality = DataQualityObserver()
        >>> # Register with event filtering - only need NodeCompleted
        >>> observer_manager.register(  # doctest: +SKIP
        ...     quality.handle,
        ...     event_types=[NodeCompleted]
        ... )
        >>> # ... run pipeline ...
        >>> if quality.has_issues():
        ...     for issue in quality.get_issues():
        ...         print(f"Quality issue in {issue.node}: {issue.message}")
    """

    # Error status constants
    ERROR_STATUSES = frozenset(["error", "failed", "failure"])

    def __init__(self) -> None:
        """Initialize data quality observer."""
        self.quality_issues: list[Alert] = []
        self.validated_nodes = 0

    async def handle(self, event: Event) -> None:
        """Check data quality in node outputs.

        Note: Should be registered with event_types=[NodeCompleted]
        for optimal performance.

        Parameters
        ----------
        event : Event
            Event to process
        """
        if isinstance(event, NodeCompleted):
            self.validated_nodes += 1
            result = event.result

            # Check for None results
            if result is None:
                self.quality_issues.append(
                    Alert(
                        type=AlertType.QUALITY_ISSUE,
                        node=event.name,
                        message="Node returned None",
                        timestamp=event.timestamp.timestamp(),
                        severity=AlertSeverity.WARNING,
                        metadata={"issue_type": "null_result"},
                    )
                )

            # Check for empty collections
            elif isinstance(result, (list, dict, str)) and not result:
                self.quality_issues.append(
                    Alert(
                        type=AlertType.QUALITY_ISSUE,
                        node=event.name,
                        message=f"Node returned empty {type(result).__name__}",
                        timestamp=event.timestamp.timestamp(),
                        severity=AlertSeverity.WARNING,
                        metadata={"issue_type": "empty_result"},
                    )
                )

            # Check for error indicators in dict results
            elif isinstance(result, dict):
                if result.get("error"):
                    self.quality_issues.append(
                        Alert(
                            type=AlertType.QUALITY_ISSUE,
                            node=event.name,
                            message="Result contains error flag",
                            timestamp=event.timestamp.timestamp(),
                            severity=AlertSeverity.ERROR,
                            error=str(result.get("error")),
                            metadata={"issue_type": "error_in_result"},
                        )
                    )
                # Check for common error status codes using constant
                if result.get("status") in self.ERROR_STATUSES:
                    self.quality_issues.append(
                        Alert(
                            type=AlertType.QUALITY_ISSUE,
                            node=event.name,
                            message=f"Result has error status: {result.get('status')}",
                            timestamp=event.timestamp.timestamp(),
                            severity=AlertSeverity.ERROR,
                            metadata={"issue_type": "error_status", "status": result.get("status")},
                        )
                    )

    def has_issues(self, severity: AlertSeverity | None = None) -> bool:
        """Check if any quality issues were detected.

        Parameters
        ----------
        severity : AlertSeverity, optional
            Filter by severity level

        Returns
        -------
        bool
            True if issues were found matching the criteria
        """
        if severity is None:
            return len(self.quality_issues) > 0
        return any(issue.severity == severity for issue in self.quality_issues)

    def get_issues(self, severity: AlertSeverity | None = None) -> list[Alert]:
        """Get all detected quality issues.

        Parameters
        ----------
        severity : AlertSeverity, optional
            Filter by severity level

        Returns
        -------
        list[Alert]
            List of quality issue alerts
        """
        if severity is None:
            return self.quality_issues
        return [i for i in self.quality_issues if i.severity == severity]

    def clear_issues(self) -> None:
        """Clear all quality issues and reset counters."""
        self.quality_issues.clear()
        self.validated_nodes = 0
