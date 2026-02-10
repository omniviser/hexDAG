"""Cost profiler observer for tracking token usage, costs, and bottlenecks.

This module provides the CostProfilerObserver that aggregates token usage data
from LLM calls, estimates costs based on model pricing, and identifies
performance bottlenecks and parallelization opportunities.
"""

from dataclasses import dataclass
from typing import Any

from hexdag.core.orchestration.events.events import (
    Event,
    LLMResponseReceived,
    NodeCompleted,
    NodeStarted,
    PipelineCompleted,
    PipelineStarted,
    WaveCompleted,
    WaveStarted,
)


@dataclass(slots=True)
class NodeCostMetrics:
    """Cost and token metrics for a single node.

    Attributes
    ----------
    input_tokens : int
        Total input tokens across all LLM calls for this node
    output_tokens : int
        Total output tokens across all LLM calls for this node
    total_tokens : int
        Total tokens (input + output) across all LLM calls
    llm_calls : int
        Number of LLM API calls made by this node
    llm_duration_ms : float
        Total duration of LLM API calls in milliseconds
    node_duration_ms : float
        Total node execution duration in milliseconds
    estimated_cost : float
        Estimated cost in USD based on model pricing
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    llm_duration_ms: float = 0.0
    node_duration_ms: float = 0.0
    estimated_cost: float = 0.0


class CostProfilerObserver:
    """Observer that tracks token usage, costs, and identifies bottlenecks.

    This observer listens to pipeline, node, wave, and LLM events to build
    a comprehensive cost and performance profile of a pipeline run.

    Parameters
    ----------
    model : str | None
        Model name for cost estimation. If None, costs are not estimated.
    pricing : dict[str, tuple[float, float]] | None
        Custom pricing table mapping model names to (input_per_1m, output_per_1m).
        Falls back to DEFAULT_PRICING if not provided.

    Example
    -------
        >>> from hexdag.core.orchestration.events import (
        ...     CostProfilerObserver,
        ...     COST_PROFILING_EVENTS,
        ... )
        >>> profiler = CostProfilerObserver(model="gpt-4o-mini")
        >>> observer_manager.register(  # doctest: +SKIP
        ...     profiler.handle,
        ...     event_types=COST_PROFILING_EVENTS
        ... )
        >>> # ... run pipeline ...
        >>> print(profiler.format_report())  # doctest: +SKIP
    """

    # Default pricing per 1M tokens (input, output) — user-overridable
    # Sources: https://platform.claude.com/docs/en/about-claude/pricing
    #          https://openai.com/api/pricing/
    DEFAULT_PRICING: dict[str, tuple[float, float]] = {
        # OpenAI — current models (Feb 2026)
        "gpt-5.2": (1.75, 14.00),
        "gpt-5": (1.25, 10.00),
        "gpt-5-nano": (0.05, 0.40),
        "gpt-4.1": (2.00, 8.00),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4.1-nano": (0.10, 0.40),
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "o3": (0.40, 1.60),
        "o4-mini": (0.40, 1.60),
        # Anthropic — current models (Feb 2026)
        "claude-opus-4-6-20260204": (5.00, 25.00),
        "claude-opus-4-5-20250701": (5.00, 25.00),
        "claude-sonnet-4-5-20250929": (3.00, 15.00),
        "claude-sonnet-4-20250514": (3.00, 15.00),
        "claude-haiku-4-5-20251001": (1.00, 5.00),
        "claude-haiku-3-5-20241022": (0.80, 4.00),
    }

    def __init__(
        self,
        model: str | None = None,
        pricing: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.model = model
        self.pricing = pricing or self.DEFAULT_PRICING
        self.node_metrics: dict[str, NodeCostMetrics] = {}
        self.pipeline_name: str | None = None
        self.pipeline_duration_ms: float = 0.0
        self.waves: list[list[str]] = []
        self._current_wave_nodes: list[str] = []

    async def handle(self, event: Event) -> None:
        """Handle events to track cost metrics.

        Parameters
        ----------
        event : Event
            The event to process. Should be registered with
            event_types=COST_PROFILING_EVENTS for performance.
        """
        if isinstance(event, PipelineStarted):
            self.pipeline_name = event.name

        elif isinstance(event, PipelineCompleted):
            self.pipeline_duration_ms = event.duration_ms

        elif isinstance(event, NodeStarted):
            if event.name not in self.node_metrics:
                self.node_metrics[event.name] = NodeCostMetrics()

        elif isinstance(event, NodeCompleted):
            if event.name not in self.node_metrics:
                self.node_metrics[event.name] = NodeCostMetrics()
            self.node_metrics[event.name].node_duration_ms += event.duration_ms

        elif isinstance(event, WaveStarted):
            self._current_wave_nodes = list(event.nodes)

        elif isinstance(event, WaveCompleted):
            if self._current_wave_nodes:
                self.waves.append(list(self._current_wave_nodes))
                self._current_wave_nodes = []

        elif isinstance(event, LLMResponseReceived):
            node_name = event.node_name
            if node_name not in self.node_metrics:
                self.node_metrics[node_name] = NodeCostMetrics()

            metrics = self.node_metrics[node_name]
            metrics.llm_calls += 1
            metrics.llm_duration_ms += event.duration_ms

            if event.usage:
                metrics.input_tokens += event.usage.get("input_tokens", 0)
                metrics.output_tokens += event.usage.get("output_tokens", 0)
                metrics.total_tokens += event.usage.get("total_tokens", 0)

                # Compute cost if model pricing is available
                if self.model and self.model in self.pricing:
                    input_price, output_price = self.pricing[self.model]
                    metrics.estimated_cost += (
                        event.usage.get("input_tokens", 0) * input_price / 1_000_000
                        + event.usage.get("output_tokens", 0) * output_price / 1_000_000
                    )

    def get_report(self) -> dict[str, Any]:
        """Generate cost profiling report.

        Returns
        -------
        dict[str, Any]
            Structured report with keys:
            - pipeline_name: Name of the pipeline
            - pipeline_duration_ms: Total pipeline duration
            - total_tokens: Aggregate token count
            - total_input_tokens: Aggregate input tokens
            - total_output_tokens: Aggregate output tokens
            - total_cost: Estimated total cost in USD
            - total_llm_calls: Total LLM API calls
            - model: Model name used for cost estimation
            - nodes: Per-node metrics dict
            - bottleneck: Node with highest duration
            - highest_token_node: Node with most tokens
            - parallelization_suggestions: List of optimization hints
        """
        total_tokens = 0
        total_input = 0
        total_output = 0
        total_cost = 0.0
        total_llm_calls = 0

        nodes: dict[str, dict[str, Any]] = {}
        for name, m in self.node_metrics.items():
            total_tokens += m.total_tokens
            total_input += m.input_tokens
            total_output += m.output_tokens
            total_cost += m.estimated_cost
            total_llm_calls += m.llm_calls

            nodes[name] = {
                "input_tokens": m.input_tokens,
                "output_tokens": m.output_tokens,
                "total_tokens": m.total_tokens,
                "llm_calls": m.llm_calls,
                "llm_duration_ms": m.llm_duration_ms,
                "node_duration_ms": m.node_duration_ms,
                "estimated_cost": m.estimated_cost,
            }

        # Find bottleneck (slowest node)
        bottleneck: dict[str, Any] | None = None
        if self.node_metrics:
            slowest = max(self.node_metrics.items(), key=lambda x: x[1].node_duration_ms)
            bottleneck = {
                "node": slowest[0],
                "duration_ms": slowest[1].node_duration_ms,
                "total_tokens": slowest[1].total_tokens,
            }

        # Find highest token consumer
        highest_token_node: dict[str, Any] | None = None
        if self.node_metrics:
            most_tokens = max(self.node_metrics.items(), key=lambda x: x[1].total_tokens)
            if most_tokens[1].total_tokens > 0:
                highest_token_node = {
                    "node": most_tokens[0],
                    "total_tokens": most_tokens[1].total_tokens,
                    "estimated_cost": most_tokens[1].estimated_cost,
                }

        # Parallelization suggestions
        suggestions = self._find_parallelization_opportunities()

        return {
            "pipeline_name": self.pipeline_name,
            "pipeline_duration_ms": self.pipeline_duration_ms,
            "total_tokens": total_tokens,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost": total_cost,
            "total_llm_calls": total_llm_calls,
            "model": self.model,
            "nodes": nodes,
            "bottleneck": bottleneck,
            "highest_token_node": highest_token_node,
            "parallelization_suggestions": suggestions,
        }

    def format_report(self) -> str:
        """Generate human-readable profiling report.

        Returns
        -------
        str
            Formatted multi-line report string
        """
        report = self.get_report()
        lines: list[str] = []

        name = report["pipeline_name"] or "unnamed"
        lines.append(f"Pipeline: {name}")
        lines.append(f"Total tokens:  {report['total_tokens']:,}")

        if report["model"]:
            lines.append(f"Model:         {report['model']} (est. ${report['total_cost']:.4f})")

        duration_s = report["pipeline_duration_ms"] / 1000
        lines.append(f"Total latency: {duration_s:.2f}s")
        lines.append(f"LLM calls:     {report['total_llm_calls']}")
        lines.append("")

        # Per-node breakdown
        if report["nodes"]:
            lines.append("Node breakdown:")
            for node_name, node_data in report["nodes"].items():
                dur = node_data["node_duration_ms"] / 1000
                tokens = node_data["total_tokens"]
                cost = node_data["estimated_cost"]
                llm_calls = node_data["llm_calls"]
                line = f"  {node_name}: {dur:.2f}s, {tokens:,} tokens, {llm_calls} LLM call(s)"
                if cost > 0:
                    line += f", ${cost:.4f}"
                lines.append(line)
            lines.append("")

        # Bottleneck
        if report["bottleneck"]:
            bn = report["bottleneck"]
            lines.append(
                f"Bottleneck:    {bn['node']} "
                f"({bn['duration_ms'] / 1000:.2f}s, {bn['total_tokens']:,} tokens)"
            )

        # Suggestions
        if report["parallelization_suggestions"]:
            lines.append("")
            lines.append("Suggestions:")
            lines.extend(
                f"  - {suggestion}" for suggestion in report["parallelization_suggestions"]
            )

        return "\n".join(lines)

    def _find_parallelization_opportunities(self) -> list[str]:
        """Find consecutive single-node waves that could be parallelized.

        Returns
        -------
        list[str]
            List of human-readable suggestions
        """
        suggestions: list[str] = []
        for i in range(len(self.waves) - 1):
            current_wave = self.waves[i]
            next_wave = self.waves[i + 1]
            if len(current_wave) == 1 and len(next_wave) == 1:
                node_a = current_wave[0]
                node_b = next_wave[0]
                suggestions.append(
                    f"{node_a} -> {node_b} are sequential but may be parallelizable. "
                    f"Check if {node_b} truly depends on {node_a}."
                )
        return suggestions

    def reset(self) -> None:
        """Reset all profiling state."""
        self.node_metrics.clear()
        self.pipeline_name = None
        self.pipeline_duration_ms = 0.0
        self.waves.clear()
        self._current_wave_nodes.clear()
