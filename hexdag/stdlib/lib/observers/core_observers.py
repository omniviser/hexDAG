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

from dataclasses import dataclass, field

from hexdag.kernel.logging import get_logger
from hexdag.kernel.orchestration.events.events import (
    Event,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    PipelineCompleted,
    PipelineStarted,
)

logger = get_logger(__name__)


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

        if self.start_time is None:
            self.start_time = event_timestamp

        elapsed_ms = (event_timestamp - self.start_time) * 1000
        event_type = type(event).__name__
        self.events.append((elapsed_ms, event_type, event))


class ExecutionTracerObserver:
    """Observer that builds detailed execution traces.

    Useful for debugging and understanding execution flow. Captures all events
    with precise timing information.

    Example
    -------
        >>> from hexdag.drivers.observer_manager import LocalObserverManager
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
            if event_name := getattr(event, "name", None):
                print(f"[{elapsed_ms:7.1f}ms] {event_type:25s} | {event_name}")
            else:
                print(f"[{elapsed_ms:7.1f}ms] {event_type:25s}")

    def reset(self) -> None:
        """Reset the trace."""
        self.trace = ExecutionTrace()


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
        >>> logger_obs = SimpleLoggingObserver(verbose=True)
        >>> # Register with observer manager
        >>> # observer_manager.register(logger_obs.handle)
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

        Uses event.log_message() for consistent formatting.

        Parameters
        ----------
        event : Event
            Event to log
        """
        # Use built-in log_message() for consistent formatting
        if isinstance(event, NodeStarted):
            logger.info(event.log_message())
            if self.verbose:
                logger.debug("  Wave: {}, Dependencies: {}", event.wave_index, event.dependencies)

        elif isinstance(event, NodeCompleted):
            logger.info(event.log_message())
            if self.verbose and event.result is not None:
                result_preview = str(event.result)[:100]
                logger.debug("  Result: {}...", result_preview)

        elif isinstance(event, NodeFailed):
            logger.error(event.log_message())
            if self.verbose:
                logger.error("  Error: {}", event.error)

        elif isinstance(event, (PipelineStarted, PipelineCompleted)):
            logger.info(event.log_message())
