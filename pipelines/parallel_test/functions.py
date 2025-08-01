"""Functions for parallel execution testing and timing verification."""

import asyncio
import time
from typing import Any

from pipelines.shared_functions import get_input_data


async def setup_timer(input_data: dict[str, Any], **ports: Any) -> dict[str, Any]:
    """Set up the pipeline execution timer and prepare data."""
    event_manager = ports.get("event_manager")
    data = get_input_data(input_data)

    # Record pipeline start time
    start_time = time.time()

    if event_manager:
        event_manager.set_memory("timer_start", start_time)
        event_manager.add_trace("setup_timer", f"Pipeline started at {start_time}")

    return {
        "setup_complete": True,
        "start_time": start_time,
        "test_data": data.get("test_data", "parallel execution test"),
        **data,
        "metadata": {"node_name": "setup_timer"},
    }


async def slow_task_a(input_data: dict[str, Any], **ports: Any) -> dict[str, Any]:
    """Execute task A with 3-second processing time."""
    event_manager = ports.get("event_manager")
    data = get_input_data(input_data)

    task_start = time.time()
    if event_manager:
        event_manager.add_trace("slow_task_a", "Starting Task A (3 seconds)...")

    # Simulate 3 seconds of processing
    await asyncio.sleep(3)

    task_end = time.time()
    execution_time = task_end - task_start

    if event_manager:
        event_manager.add_trace("slow_task_a", f"Task A completed in {execution_time:.2f} seconds")

    return {
        "task_name": "Task A",
        "execution_time": execution_time,
        "result": "Task A processing completed successfully",
        "start_time": task_start,
        "end_time": task_end,
        **data,
        "metadata": {"node_name": "slow_task_a"},
    }


async def slow_task_b(input_data: dict[str, Any], **ports: Any) -> dict[str, Any]:
    """Execute task B with 3-second processing time."""
    event_manager = ports.get("event_manager")
    data = get_input_data(input_data)

    task_start = time.time()
    if event_manager:
        event_manager.add_trace("slow_task_b", "Starting Task B (3 seconds)...")

    # Simulate 3 seconds of processing
    await asyncio.sleep(3)

    task_end = time.time()
    execution_time = task_end - task_start

    if event_manager:
        event_manager.add_trace("slow_task_b", f"Task B completed in {execution_time:.2f} seconds")

    return {
        "task_name": "Task B",
        "execution_time": execution_time,
        "result": "Task B processing completed successfully",
        "start_time": task_start,
        "end_time": task_end,
        **data,
        "metadata": {"node_name": "slow_task_b"},
    }


async def timing_analyzer(input_data: dict[str, Any], **ports: Any) -> dict[str, Any]:
    """Analyze execution timing to verify parallel execution."""
    event_manager = ports.get("event_manager")
    data = get_input_data(input_data)

    # Get timing data from memory
    pipeline_start = None
    if event_manager:
        pipeline_start = event_manager.get_memory("timer_start", time.time())

    # Get task results from input data
    task_a_result = input_data.get("task_a_result", {})
    task_b_result = input_data.get("task_b_result", {})

    analysis_time = time.time()
    total_pipeline_time = analysis_time - pipeline_start if pipeline_start else 0

    # Calculate parallel execution verification
    task_a_time = task_a_result.get("execution_time", 0)
    task_b_time = task_b_result.get("execution_time", 0)

    # If tasks ran in parallel, total time should be ~3 seconds (not 6)
    # If tasks ran sequentially, total time would be ~6 seconds
    expected_parallel_time = max(task_a_time, task_b_time)  # ~3 seconds
    expected_sequential_time = task_a_time + task_b_time  # ~6 seconds

    # Determine if execution was parallel (with some tolerance for overhead)
    tolerance = 0.5  # 0.5 second tolerance
    was_parallel = total_pipeline_time <= (expected_parallel_time + tolerance)

    # Calculate efficiency
    efficiency = (
        (expected_parallel_time / total_pipeline_time) * 100 if total_pipeline_time > 0 else 0
    )

    analysis_result = {
        "total_pipeline_time": round(total_pipeline_time, 2),
        "task_a_time": round(task_a_time, 2),
        "task_b_time": round(task_b_time, 2),
        "expected_parallel_time": round(expected_parallel_time, 2),
        "expected_sequential_time": round(expected_sequential_time, 2),
        "was_parallel": was_parallel,
        "efficiency_percentage": round(efficiency, 1),
        "tolerance_used": tolerance,
    }

    if event_manager:
        event_manager.add_trace(
            "timing_analyzer",
            f"Analysis complete: parallel={was_parallel}, efficiency={efficiency:.1f}%",
        )

    # Create timing analysis with expected structure from tests
    timing_analysis = {
        "fastest_task": "Task A" if task_a_time <= task_b_time else "Task B",
        "slowest_task": "Task B" if task_b_time >= task_a_time else "Task A",
        "average_task_time": (task_a_time + task_b_time) / 2,
        **analysis_result,
    }

    # Create task timings structure expected by tests
    task_timings = {"Task A": task_a_time, "Task B": task_b_time}

    return {
        "timing_analysis": timing_analysis,
        "total_pipeline_time": total_pipeline_time,
        "task_timings": task_timings,
        "task_a_result": task_a_result,
        "task_b_result": task_b_result,
        "pipeline_start_time": pipeline_start,
        "analysis_time": analysis_time,
        **data,
        "metadata": {"node_name": "timing_analyzer"},
    }


async def result_aggregator(input_data: dict[str, Any], **ports: Any) -> dict[str, Any]:
    """Aggregate all results into a final summary."""
    event_manager = ports.get("event_manager")
    data = get_input_data(input_data)

    # Extract results from previous nodes
    timing_analysis = input_data.get("timing_analysis", {})
    task_a_result = input_data.get("task_a_result", {})
    task_b_result = input_data.get("task_b_result", {})
    setup_data = input_data.get("setup_timer", {})

    # Create comprehensive summary
    summary = {
        "test_name": "Parallel Execution Test",
        "test_data": setup_data.get("test_data", "parallel execution test"),
        "execution_summary": {
            "total_pipeline_time": timing_analysis.get("total_pipeline_time", 0),
            "was_parallel": timing_analysis.get("was_parallel", False),
            "efficiency_percentage": timing_analysis.get("efficiency_percentage", 0),
        },
        "task_results": {
            "task_a": {
                "name": task_a_result.get("task_name", "Task A"),
                "execution_time": task_a_result.get("execution_time", 0),
                "result": task_a_result.get("result", ""),
            },
            "task_b": {
                "name": task_b_result.get("task_name", "Task B"),
                "execution_time": task_b_result.get("execution_time", 0),
                "result": task_b_result.get("result", ""),
            },
        },
        "performance_metrics": {
            "expected_parallel_time": timing_analysis.get("expected_parallel_time", 0),
            "expected_sequential_time": timing_analysis.get("expected_sequential_time", 0),
            "actual_time": timing_analysis.get("total_pipeline_time", 0),
            "time_saved": timing_analysis.get("expected_sequential_time", 0)
            - timing_analysis.get("total_pipeline_time", 0),
        },
    }

    if event_manager:
        event_manager.add_trace("result_aggregator", "Aggregated all test results successfully")

    # Create aggregated results structure expected by tests
    aggregated_results = {
        "total_execution_time": timing_analysis.get("total_pipeline_time", 0),
        "task_count": 2,
        "fastest_task": timing_analysis.get("fastest_task", "Task A"),
        "slowest_task": timing_analysis.get("slowest_task", "Task B"),
    }

    # Create summary structure expected by tests
    test_summary = {
        "pipeline_success": True,
        "tasks_completed": 2,
        "performance_summary": (
            f"Completed {2} tasks in {timing_analysis.get('total_pipeline_time', 0):.2f}s"
        ),
    }

    # Create detailed results structure
    detailed_results = {
        "full_summary": summary,
        "timing_data": timing_analysis,
        "individual_tasks": {"task_a": task_a_result, "task_b": task_b_result},
    }

    return {
        "aggregated_results": aggregated_results,
        "summary": test_summary,
        "detailed_results": detailed_results,
        "final_summary": summary,
        "timing_analysis": timing_analysis,
        "task_results": {"task_a": task_a_result, "task_b": task_b_result},
        "setup_data": setup_data,
        **data,
        "metadata": {"node_name": "result_aggregator"},
    }
