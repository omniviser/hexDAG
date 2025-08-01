"""Parallel execution test pipeline for verifying DAG parallelization."""

from pipelines.base import PipelineDefinition

from .functions import result_aggregator, setup_timer, slow_task_a, slow_task_b, timing_analyzer


class ParallelTestPipeline(PipelineDefinition):
    """Pipeline to verify parallel execution capabilities with timing measurements.

    This pipeline demonstrates and verifies that the hexAI framework correctly
    executes independent nodes in parallel rather than sequentially.

    Execution Structure:
    - Wave 1: Setup timer (1 node)
    - Wave 2: Two 3-second tasks in parallel (2 nodes)
    - Wave 3: Analyze timing results (1 node)

    Expected Results:
    - Parallel execution: ~3-4 seconds total
    - Sequential execution: ~6-7 seconds total
    """

    @property
    def name(self) -> str:
        """Pipeline name."""
        return "parallel_test_pipeline"

    @property
    def description(self) -> str:
        """Pipeline description."""
        return "Parallel execution test"

    def _register_functions(self) -> None:
        """Register all pipeline functions with the builder."""
        # Register timing and parallel execution functions
        self.builder.register_function("setup_timer", setup_timer)
        self.builder.register_function("slow_task_a", slow_task_a)
        self.builder.register_function("slow_task_b", slow_task_b)
        self.builder.register_function("timing_analyzer", timing_analyzer)
        self.builder.register_function("result_aggregator", result_aggregator)
