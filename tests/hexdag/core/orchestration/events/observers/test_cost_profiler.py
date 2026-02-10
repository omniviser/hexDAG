"""Tests for CostProfilerObserver."""

import pytest

from hexdag.core.orchestration.events import (
    CostProfilerObserver,
    LLMResponseReceived,
    NodeCompleted,
    NodeStarted,
    PipelineCompleted,
    PipelineStarted,
    WaveCompleted,
    WaveStarted,
)
from hexdag.core.orchestration.events.observers.cost_profiler import NodeCostMetrics

# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def profiler():
    """Create a CostProfilerObserver with gpt-4o-mini pricing."""
    return CostProfilerObserver(model="gpt-4o-mini")


@pytest.fixture
def profiler_no_model():
    """Create a CostProfilerObserver without a model (no cost estimation)."""
    return CostProfilerObserver()


# ==============================================================================
# TOKEN AGGREGATION TESTS
# ==============================================================================


class TestTokenAggregation:
    """Test token usage aggregation from LLMResponseReceived events."""

    @pytest.mark.asyncio
    async def test_single_llm_call(self, profiler):
        """Verify tokens from a single LLM call are tracked."""
        await profiler.handle(
            LLMResponseReceived(
                node_name="summarizer",
                response="Hello world",
                duration_ms=150.0,
                usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            )
        )

        metrics = profiler.node_metrics["summarizer"]
        assert metrics.input_tokens == 100
        assert metrics.output_tokens == 50
        assert metrics.total_tokens == 150
        assert metrics.llm_calls == 1
        assert metrics.llm_duration_ms == 150.0

    @pytest.mark.asyncio
    async def test_multiple_llm_calls_same_node(self, profiler):
        """Verify tokens accumulate across multiple LLM calls for the same node."""
        for i in range(3):
            await profiler.handle(
                LLMResponseReceived(
                    node_name="agent",
                    response=f"step {i}",
                    duration_ms=100.0,
                    usage={"input_tokens": 200, "output_tokens": 100, "total_tokens": 300},
                )
            )

        metrics = profiler.node_metrics["agent"]
        assert metrics.input_tokens == 600
        assert metrics.output_tokens == 300
        assert metrics.total_tokens == 900
        assert metrics.llm_calls == 3
        assert metrics.llm_duration_ms == 300.0

    @pytest.mark.asyncio
    async def test_multiple_nodes(self, profiler):
        """Verify separate tracking for different nodes."""
        await profiler.handle(
            LLMResponseReceived(
                node_name="node_a",
                response="a",
                duration_ms=100.0,
                usage={"input_tokens": 500, "output_tokens": 200, "total_tokens": 700},
            )
        )
        await profiler.handle(
            LLMResponseReceived(
                node_name="node_b",
                response="b",
                duration_ms=200.0,
                usage={"input_tokens": 1000, "output_tokens": 400, "total_tokens": 1400},
            )
        )

        assert profiler.node_metrics["node_a"].total_tokens == 700
        assert profiler.node_metrics["node_b"].total_tokens == 1400

    @pytest.mark.asyncio
    async def test_report_total_tokens(self, profiler):
        """Verify report aggregates total tokens across all nodes."""
        await profiler.handle(
            LLMResponseReceived(
                node_name="a",
                response="",
                duration_ms=50.0,
                usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            )
        )
        await profiler.handle(
            LLMResponseReceived(
                node_name="b",
                response="",
                duration_ms=50.0,
                usage={"input_tokens": 200, "output_tokens": 80, "total_tokens": 280},
            )
        )

        report = profiler.get_report()
        assert report["total_tokens"] == 430
        assert report["total_input_tokens"] == 300
        assert report["total_output_tokens"] == 130
        assert report["total_llm_calls"] == 2


# ==============================================================================
# COST CALCULATION TESTS
# ==============================================================================


class TestCostCalculation:
    """Test cost estimation logic."""

    @pytest.mark.asyncio
    async def test_gpt4o_mini_cost(self, profiler):
        """Verify cost for gpt-4o-mini: $0.15/$0.60 per 1M tokens."""
        await profiler.handle(
            LLMResponseReceived(
                node_name="node",
                response="",
                duration_ms=100.0,
                usage={
                    "input_tokens": 1_000_000,
                    "output_tokens": 1_000_000,
                    "total_tokens": 2_000_000,
                },
            )
        )

        metrics = profiler.node_metrics["node"]
        # input: 1M * $0.15/1M = $0.15, output: 1M * $0.60/1M = $0.60
        assert metrics.estimated_cost == pytest.approx(0.75, abs=0.001)

    @pytest.mark.asyncio
    async def test_custom_pricing(self):
        """Verify custom pricing overrides defaults."""
        custom_pricing = {"my-model": (10.0, 30.0)}
        profiler = CostProfilerObserver(model="my-model", pricing=custom_pricing)

        await profiler.handle(
            LLMResponseReceived(
                node_name="node",
                response="",
                duration_ms=50.0,
                usage={"input_tokens": 500_000, "output_tokens": 100_000, "total_tokens": 600_000},
            )
        )

        # input: 500k * $10/1M = $5.00, output: 100k * $30/1M = $3.00
        report = profiler.get_report()
        assert report["total_cost"] == pytest.approx(8.0, abs=0.001)

    @pytest.mark.asyncio
    async def test_no_model_no_cost(self, profiler_no_model):
        """Verify no cost is estimated when model is None."""
        await profiler_no_model.handle(
            LLMResponseReceived(
                node_name="node",
                response="",
                duration_ms=100.0,
                usage={"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500},
            )
        )

        report = profiler_no_model.get_report()
        assert report["total_cost"] == 0.0
        assert profiler_no_model.node_metrics["node"].estimated_cost == 0.0

    @pytest.mark.asyncio
    async def test_unknown_model_no_cost(self):
        """Verify no cost is estimated for a model not in pricing table."""
        profiler = CostProfilerObserver(model="unknown-model-xyz")

        await profiler.handle(
            LLMResponseReceived(
                node_name="node",
                response="",
                duration_ms=100.0,
                usage={"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500},
            )
        )

        assert profiler.node_metrics["node"].estimated_cost == 0.0


# ==============================================================================
# NO-USAGE FALLBACK TESTS
# ==============================================================================


class TestNoUsageFallback:
    """Test graceful handling when events have no usage data."""

    @pytest.mark.asyncio
    async def test_none_usage(self, profiler):
        """Events with usage=None should be handled gracefully."""
        await profiler.handle(
            LLMResponseReceived(
                node_name="node",
                response="hello",
                duration_ms=100.0,
                usage=None,
            )
        )

        metrics = profiler.node_metrics["node"]
        assert metrics.total_tokens == 0
        assert metrics.llm_calls == 1
        assert metrics.llm_duration_ms == 100.0

    @pytest.mark.asyncio
    async def test_partial_usage(self, profiler):
        """Events with partial usage dict should use defaults."""
        await profiler.handle(
            LLMResponseReceived(
                node_name="node",
                response="",
                duration_ms=50.0,
                usage={"input_tokens": 100},
            )
        )

        metrics = profiler.node_metrics["node"]
        assert metrics.input_tokens == 100
        assert metrics.output_tokens == 0
        assert metrics.total_tokens == 0


# ==============================================================================
# BOTTLENECK DETECTION TESTS
# ==============================================================================


class TestBottleneckDetection:
    """Test bottleneck and highest-token-node detection."""

    @pytest.mark.asyncio
    async def test_bottleneck_is_slowest_node(self, profiler):
        """Verify bottleneck is the node with highest duration."""
        await profiler.handle(NodeStarted(name="fast", wave_index=0, dependencies=[]))
        await profiler.handle(
            NodeCompleted(name="fast", wave_index=0, result={}, duration_ms=100.0)
        )

        await profiler.handle(NodeStarted(name="slow", wave_index=1, dependencies=["fast"]))
        await profiler.handle(
            NodeCompleted(name="slow", wave_index=1, result={}, duration_ms=2000.0)
        )

        await profiler.handle(NodeStarted(name="medium", wave_index=1, dependencies=["fast"]))
        await profiler.handle(
            NodeCompleted(name="medium", wave_index=1, result={}, duration_ms=500.0)
        )

        report = profiler.get_report()
        assert report["bottleneck"]["node"] == "slow"
        assert report["bottleneck"]["duration_ms"] == 2000.0

    @pytest.mark.asyncio
    async def test_highest_token_node(self, profiler):
        """Verify highest_token_node is the node with the most tokens."""
        await profiler.handle(
            LLMResponseReceived(
                node_name="light",
                response="",
                duration_ms=50.0,
                usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            )
        )
        await profiler.handle(
            LLMResponseReceived(
                node_name="heavy",
                response="",
                duration_ms=200.0,
                usage={"input_tokens": 3000, "output_tokens": 1000, "total_tokens": 4000},
            )
        )

        report = profiler.get_report()
        assert report["highest_token_node"]["node"] == "heavy"
        assert report["highest_token_node"]["total_tokens"] == 4000

    @pytest.mark.asyncio
    async def test_no_nodes_no_bottleneck(self, profiler):
        """Verify no bottleneck when there are no nodes."""
        report = profiler.get_report()
        assert report["bottleneck"] is None
        assert report["highest_token_node"] is None


# ==============================================================================
# PIPELINE LIFECYCLE TESTS
# ==============================================================================


class TestPipelineLifecycle:
    """Test pipeline and wave event handling."""

    @pytest.mark.asyncio
    async def test_pipeline_name_and_duration(self, profiler):
        """Verify pipeline name and duration are tracked."""
        await profiler.handle(PipelineStarted(name="my-pipeline", total_waves=2, total_nodes=3))
        await profiler.handle(PipelineCompleted(name="my-pipeline", duration_ms=5000.0))

        report = profiler.get_report()
        assert report["pipeline_name"] == "my-pipeline"
        assert report["pipeline_duration_ms"] == 5000.0

    @pytest.mark.asyncio
    async def test_wave_tracking(self, profiler):
        """Verify waves are tracked for parallelization analysis."""
        await profiler.handle(WaveStarted(wave_index=0, nodes=["a", "b"]))
        await profiler.handle(WaveCompleted(wave_index=0, duration_ms=100.0))

        await profiler.handle(WaveStarted(wave_index=1, nodes=["c"]))
        await profiler.handle(WaveCompleted(wave_index=1, duration_ms=200.0))

        assert profiler.waves == [["a", "b"], ["c"]]


# ==============================================================================
# PARALLELIZATION SUGGESTIONS TESTS
# ==============================================================================


class TestParallelizationSuggestions:
    """Test parallelization opportunity detection."""

    @pytest.mark.asyncio
    async def test_consecutive_single_node_waves(self, profiler):
        """Two consecutive single-node waves trigger a suggestion."""
        await profiler.handle(WaveStarted(wave_index=0, nodes=["a"]))
        await profiler.handle(WaveCompleted(wave_index=0, duration_ms=100.0))

        await profiler.handle(WaveStarted(wave_index=1, nodes=["b"]))
        await profiler.handle(WaveCompleted(wave_index=1, duration_ms=100.0))

        report = profiler.get_report()
        suggestions = report["parallelization_suggestions"]
        assert len(suggestions) == 1
        assert "a" in suggestions[0]
        assert "b" in suggestions[0]

    @pytest.mark.asyncio
    async def test_no_suggestion_for_multi_node_waves(self, profiler):
        """Multi-node waves should not trigger suggestions."""
        await profiler.handle(WaveStarted(wave_index=0, nodes=["a", "b"]))
        await profiler.handle(WaveCompleted(wave_index=0, duration_ms=100.0))

        await profiler.handle(WaveStarted(wave_index=1, nodes=["c"]))
        await profiler.handle(WaveCompleted(wave_index=1, duration_ms=100.0))

        report = profiler.get_report()
        assert len(report["parallelization_suggestions"]) == 0


# ==============================================================================
# REPORT FORMAT TESTS
# ==============================================================================


class TestFormatReport:
    """Test human-readable report generation."""

    @pytest.mark.asyncio
    async def test_format_includes_key_sections(self, profiler):
        """Verify format_report produces readable output with all sections."""
        await profiler.handle(PipelineStarted(name="test-pipeline", total_waves=1, total_nodes=2))
        await profiler.handle(NodeStarted(name="node_a", wave_index=0, dependencies=[]))
        await profiler.handle(
            LLMResponseReceived(
                node_name="node_a",
                response="",
                duration_ms=100.0,
                usage={"input_tokens": 500, "output_tokens": 200, "total_tokens": 700},
            )
        )
        await profiler.handle(
            NodeCompleted(name="node_a", wave_index=0, result={}, duration_ms=150.0)
        )
        await profiler.handle(PipelineCompleted(name="test-pipeline", duration_ms=200.0))

        output = profiler.format_report()
        assert "Pipeline: test-pipeline" in output
        assert "Total tokens:" in output
        assert "700" in output
        assert "gpt-4o-mini" in output
        assert "node_a" in output
        assert "Bottleneck:" in output

    @pytest.mark.asyncio
    async def test_format_no_model(self, profiler_no_model):
        """Verify format_report works without model/cost."""
        await profiler_no_model.handle(PipelineStarted(name="p", total_waves=1, total_nodes=1))
        await profiler_no_model.handle(PipelineCompleted(name="p", duration_ms=100.0))

        output = profiler_no_model.format_report()
        assert "Pipeline: p" in output
        assert "Model:" not in output


# ==============================================================================
# RESET TESTS
# ==============================================================================


class TestReset:
    """Test profiler state reset."""

    @pytest.mark.asyncio
    async def test_reset_clears_all_state(self, profiler):
        """Verify reset clears all accumulated state."""
        await profiler.handle(PipelineStarted(name="pipeline", total_waves=1, total_nodes=1))
        await profiler.handle(
            LLMResponseReceived(
                node_name="node",
                response="",
                duration_ms=100.0,
                usage={"input_tokens": 500, "output_tokens": 200, "total_tokens": 700},
            )
        )
        await profiler.handle(WaveStarted(wave_index=0, nodes=["node"]))
        await profiler.handle(WaveCompleted(wave_index=0, duration_ms=100.0))
        await profiler.handle(PipelineCompleted(name="pipeline", duration_ms=200.0))

        # Verify state exists
        assert len(profiler.node_metrics) == 1
        assert profiler.pipeline_name == "pipeline"
        assert len(profiler.waves) == 1

        profiler.reset()

        # Verify all state cleared
        assert len(profiler.node_metrics) == 0
        assert profiler.pipeline_name is None
        assert profiler.pipeline_duration_ms == 0.0
        assert len(profiler.waves) == 0

        # Verify report is empty after reset
        report = profiler.get_report()
        assert report["total_tokens"] == 0
        assert report["bottleneck"] is None


# ==============================================================================
# NODE COST METRICS TESTS
# ==============================================================================


class TestNodeCostMetrics:
    """Test NodeCostMetrics dataclass."""

    def test_default_values(self):
        """Verify defaults are all zero."""
        m = NodeCostMetrics()
        assert m.input_tokens == 0
        assert m.output_tokens == 0
        assert m.total_tokens == 0
        assert m.llm_calls == 0
        assert m.llm_duration_ms == 0.0
        assert m.node_duration_ms == 0.0
        assert m.estimated_cost == 0.0
