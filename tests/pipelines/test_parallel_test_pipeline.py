"""Tests for parallel_test pipeline."""

import time

import pytest

from hexai.pipelines.parallel_test.functions import (
    result_aggregator,
    setup_timer,
    slow_task_a,
    slow_task_b,
    timing_analyzer,
)
from hexai.pipelines.parallel_test.pipeline import ParallelTestPipeline


class MockEventManager:
    """Mock event manager for testing."""

    def __init__(self):
        """Initialize the mock event manager."""
        self.traces = []
        self.memory = {}

    def add_trace(self, node_name: str, message: str):
        """Add a trace message."""
        self.traces.append(f"{node_name}: {message}")

    def set_memory(self, key: str, value):
        """Set a memory value."""
        self.memory[key] = value

    def get_memory(self, key: str, default=None):
        """Get a memory value."""
        return self.memory.get(key, default)


class TestParallelTestFunctions:
    """Tests for parallel_test pipeline functions."""

    @pytest.mark.asyncio
    async def test_setup_timer(self):
        """Test setup_timer function."""
        event_manager = MockEventManager()
        input_data = {"test_data": "test input"}

        result = await setup_timer(input_data, event_manager=event_manager)

        assert result["setup_complete"] is True
        assert "start_time" in result
        assert result["test_data"] == "test input"
        assert result["metadata"]["node_name"] == "setup_timer"

        # Check event manager was used
        assert len(event_manager.traces) > 0
        assert any("setup_timer" in trace for trace in event_manager.traces)
        assert "timer_start" in event_manager.memory

    @pytest.mark.asyncio
    async def test_slow_task_a(self):
        """Test slow_task_a function."""
        event_manager = MockEventManager()
        input_data = {"test_data": "test input"}

        result = await slow_task_a(input_data, event_manager=event_manager)

        assert result["task_name"] == "Task A"
        assert "execution_time" in result
        assert result["result"] == "Task A processing completed successfully"
        assert "start_time" in result
        assert "end_time" in result
        assert result["test_data"] == "test input"
        assert result["metadata"]["node_name"] == "slow_task_a"

        # Check event manager was used
        assert len(event_manager.traces) >= 2
        assert any("slow_task_a" in trace for trace in event_manager.traces)

    @pytest.mark.asyncio
    async def test_slow_task_b(self):
        """Test slow_task_b function."""
        event_manager = MockEventManager()
        input_data = {"test_data": "test input"}

        result = await slow_task_b(input_data, event_manager=event_manager)

        assert result["task_name"] == "Task B"
        assert "execution_time" in result
        assert result["result"] == "Task B processing completed successfully"
        assert "start_time" in result
        assert "end_time" in result
        assert result["test_data"] == "test input"
        assert result["metadata"]["node_name"] == "slow_task_b"

        # Check event manager was used
        assert len(event_manager.traces) >= 2
        assert any("slow_task_b" in trace for trace in event_manager.traces)

    @pytest.mark.asyncio
    async def test_timing_analyzer(self):
        """Test timing_analyzer function."""
        event_manager = MockEventManager()
        # Set up memory with timer start
        event_manager.set_memory("timer_start", time.time() - 2.0)

        input_data = {
            "task_a_result": {
                "task_name": "Task A",
                "execution_time": 1.0,
                "result": "Task A completed",
            },
            "task_b_result": {
                "task_name": "Task B",
                "execution_time": 1.5,
                "result": "Task B completed",
            },
        }

        result = await timing_analyzer(input_data, event_manager=event_manager)

        assert "timing_analysis" in result
        assert "total_pipeline_time" in result
        assert "task_timings" in result
        assert result["metadata"]["node_name"] == "timing_analyzer"

        # Check timing analysis structure
        timing_analysis = result["timing_analysis"]
        assert "fastest_task" in timing_analysis
        assert "slowest_task" in timing_analysis
        assert "average_task_time" in timing_analysis

        # Check task timings
        task_timings = result["task_timings"]
        assert "Task A" in task_timings
        assert "Task B" in task_timings
        assert task_timings["Task A"] == 1.0
        assert task_timings["Task B"] == 1.5

        # Check event manager was used
        assert len(event_manager.traces) > 0
        assert any("timing_analyzer" in trace for trace in event_manager.traces)

    @pytest.mark.asyncio
    async def test_timing_analyzer_no_memory(self):
        """Test timing_analyzer function without timer start in memory."""
        event_manager = MockEventManager()
        input_data = {
            "task_a_result": {"execution_time": 1.0},
            "task_b_result": {"execution_time": 1.5},
        }

        result = await timing_analyzer(input_data, event_manager=event_manager)

        # Should still work but with default pipeline time
        assert "timing_analysis" in result
        assert "total_pipeline_time" in result
        assert result["total_pipeline_time"] >= 0  # Should be a valid time

    @pytest.mark.asyncio
    async def test_result_aggregator(self):
        """Test result_aggregator function."""
        event_manager = MockEventManager()
        input_data = {
            "timer_result": {"setup_complete": True, "start_time": time.time() - 3.0},
            "task_a_result": {
                "task_name": "Task A",
                "execution_time": 1.0,
                "result": "Task A completed",
            },
            "task_b_result": {
                "task_name": "Task B",
                "execution_time": 1.5,
                "result": "Task B completed",
            },
            "analysis_result": {
                "timing_analysis": {
                    "fastest_task": "Task A",
                    "slowest_task": "Task B",
                    "average_task_time": 1.25,
                },
                "total_pipeline_time": 3.0,
            },
        }

        result = await result_aggregator(input_data, event_manager=event_manager)

        assert "aggregated_results" in result
        assert "summary" in result
        assert "detailed_results" in result
        assert result["metadata"]["node_name"] == "result_aggregator"

        # Check aggregated results
        aggregated = result["aggregated_results"]
        assert "total_execution_time" in aggregated
        assert "task_count" in aggregated
        assert "fastest_task" in aggregated
        assert "slowest_task" in aggregated

        # Check summary
        summary = result["summary"]
        assert "pipeline_success" in summary
        assert "tasks_completed" in summary
        assert "performance_summary" in summary

        # Check event manager was used
        assert len(event_manager.traces) > 0
        assert any("result_aggregator" in trace for trace in event_manager.traces)

    @pytest.mark.asyncio
    async def test_functions_without_event_manager(self):
        """Test that functions work without event manager."""
        input_data = {"test_data": "test input"}

        # All functions should work without event manager
        timer_result = await setup_timer(input_data)
        task_a_result = await slow_task_a(input_data)
        task_b_result = await slow_task_b(input_data)

        assert timer_result["setup_complete"] is True
        assert task_a_result["task_name"] == "Task A"
        assert task_b_result["task_name"] == "Task B"

    @pytest.mark.asyncio
    async def test_function_pipeline_integration(self):
        """Test that functions work together in pipeline order."""
        event_manager = MockEventManager()
        input_data = {"test_data": "pipeline integration test"}

        # Step 1: Setup timer
        timer_result = await setup_timer(input_data, event_manager=event_manager)

        # Step 2: Run tasks in parallel (simulated)
        task_a_result = await slow_task_a(timer_result, event_manager=event_manager)
        task_b_result = await slow_task_b(timer_result, event_manager=event_manager)

        # Step 3: Analyze timing
        analysis_input = {
            **timer_result,
            "task_a_result": task_a_result,
            "task_b_result": task_b_result,
        }
        analysis_result = await timing_analyzer(analysis_input, event_manager=event_manager)

        # Step 4: Aggregate results
        aggregation_input = {
            "timer_result": timer_result,
            "task_a_result": task_a_result,
            "task_b_result": task_b_result,
            "analysis_result": analysis_result,
        }
        final_result = await result_aggregator(aggregation_input, event_manager=event_manager)

        # Verify the pipeline flow
        assert timer_result["setup_complete"] is True
        assert task_a_result["task_name"] == "Task A"
        assert task_b_result["task_name"] == "Task B"
        assert "timing_analysis" in analysis_result
        assert "aggregated_results" in final_result

        # Verify event manager usage throughout the pipeline
        assert len(event_manager.traces) >= 5  # At least one trace per function

    @pytest.mark.asyncio
    async def test_event_manager_memory_usage(self):
        """Test that functions properly use event manager memory."""
        event_manager = MockEventManager()
        input_data = {"test_data": "memory test"}

        # Setup timer should set memory
        await setup_timer(input_data, event_manager=event_manager)
        assert "timer_start" in event_manager.memory

        # Timing analyzer should read from memory
        analysis_input = {
            "task_a_result": {"execution_time": 1.0},
            "task_b_result": {"execution_time": 1.5},
        }
        result = await timing_analyzer(analysis_input, event_manager=event_manager)

        # Should have used the timer start from memory
        assert result["total_pipeline_time"] > 0


class TestParallelTestPipeline:
    """Tests for ParallelTestPipeline class."""

    def test_pipeline_attributes(self):
        """Test pipeline basic attributes."""
        pipeline = ParallelTestPipeline()

        assert pipeline.name == "parallel_test_pipeline"
        assert "Parallel execution test" in pipeline.description
        assert hasattr(pipeline, "builder")

    def test_pipeline_config(self):
        """Test pipeline configuration."""
        pipeline = ParallelTestPipeline()
        config = pipeline.get_config()

        assert config is not None
        assert "nodes" in config
        assert len(config["nodes"]) >= 5  # Should have timer, tasks, analyzer, aggregator nodes

        # Check that parallel tasks have no dependencies on each other
        nodes = config["nodes"]
        node_ids = [node["id"] for node in nodes]

        expected_nodes = ["timer", "task_a", "task_b", "analyzer", "aggregator"]
        for expected_node in expected_nodes:
            assert any(expected_node in node_id for node_id in node_ids)

    def test_function_registration(self):
        """Test that all required functions are registered."""
        pipeline = ParallelTestPipeline()

        # Check that all functions are registered
        expected_functions = [
            "setup_timer",
            "slow_task_a",
            "slow_task_b",
            "timing_analyzer",
            "result_aggregator",
        ]
        for func_name in expected_functions:
            assert func_name in pipeline.builder.registered_functions
            func = pipeline.builder.registered_functions[func_name]
            assert callable(func)
