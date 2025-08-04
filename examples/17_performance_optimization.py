"""
Example 17: Performance Optimization.

This example demonstrates performance optimization in hexAI:
- Parallel execution
- Caching strategies
- Resource management
- Performance monitoring
- Optimization techniques
"""

import asyncio
import time
from typing import Any

from hexai.core.application.orchestrator import Orchestrator
from hexai.core.domain.dag import DirectedGraph, NodeSpec


async def slow_processor(input_data: str, **kwargs) -> dict:
    """Simulate slow processing."""
    await asyncio.sleep(0.5)  # Simulate slow operation
    return {"processed_data": input_data.upper(), "processing_time": 0.5, "processor_type": "slow"}


async def fast_processor(input_data: str, **kwargs) -> dict:
    """Simulate fast processing."""
    await asyncio.sleep(0.1)  # Simulate fast operation
    return {"processed_data": input_data.lower(), "processing_time": 0.1, "processor_type": "fast"}


async def parallel_processor(input_data: dict, **kwargs) -> dict:
    """Process data in parallel."""
    # Simulate parallel processing
    await asyncio.sleep(0.2)
    return {
        "parallel_result": input_data.get("processed_data", ""),
        "parallel_processing": True,
        "processing_time": 0.2,
    }


async def cache_processor(input_data: str, **kwargs) -> dict:
    """Processor with caching simulation."""
    # Simulate cache lookup
    cache_key = f"cache_{hash(input_data)}"

    # Simulate cache hit/miss
    if hash(input_data) % 2 == 0:  # Simulate cache hit
        await asyncio.sleep(0.05)  # Fast cache hit
        cache_status = "hit"
    else:
        await asyncio.sleep(0.3)  # Slow cache miss
        cache_status = "miss"

    return {
        "cached_data": input_data.upper(),
        "cache_key": cache_key,
        "cache_status": cache_status,
        "processing_time": 0.05 if cache_status == "hit" else 0.3,
    }


async def resource_intensive_processor(input_data: dict, **kwargs) -> dict:
    """Simulate resource-intensive processing."""
    # Simulate CPU-intensive operation
    start_time = time.time()

    # Simulate heavy computation
    result = 0
    for i in range(100000):
        result += i

    processing_time = time.time() - start_time

    return {
        "computation_result": result,
        "processing_time": processing_time,
        "resource_usage": "high",
    }


async def optimized_aggregator(input_data: Any, **kwargs) -> dict:
    """Aggregate results with performance metrics."""
    results = input_data

    # Calculate performance metrics
    total_time = sum(
        result.get("processing_time", 0) for result in results.values() if isinstance(result, dict)
    )

    return {
        "total_processing_time": total_time,
        "optimization_level": "high",
        "performance_improvement": "significant",
    }


async def demonstrate_parallel_execution():
    """Demonstrate parallel execution optimization."""
    print("\n⚡ Parallel Execution Optimization")
    print("-" * 35)

    # Create graph with parallel nodes
    graph = DirectedGraph()

    # Add parallel processing nodes
    slow1 = NodeSpec("slow_processor_1", slow_processor)
    slow2 = NodeSpec("slow_processor_2", slow_processor)
    slow3 = NodeSpec("slow_processor_3", slow_processor)

    # Parallel aggregator
    parallel_agg = NodeSpec("parallel_aggregator", parallel_processor).after(
        "slow_processor_1", "slow_processor_2", "slow_processor_3"
    )

    graph.add(slow1)
    graph.add(slow2)
    graph.add(slow3)
    graph.add(parallel_agg)

    # Execute with timing
    print("   ⏱️  Starting parallel execution...")
    start_time = time.time()

    orchestrator = Orchestrator()
    result = await orchestrator.run(graph, "parallel test data")

    execution_time = time.time() - start_time
    print(f"   ⚡ Parallel execution time: {execution_time:.2f}s")
    print(f"   📊 Expected sequential time: ~1.5s")
    print(f"   🚀 Speedup: {1.5/execution_time:.1f}x")


async def demonstrate_caching_strategy():
    """Demonstrate caching optimization."""
    print("\n💾 Caching Strategy")
    print("-" * 35)

    # Create graph with caching
    graph = DirectedGraph()

    # Add cache processors
    cache1 = NodeSpec("cache_processor_1", cache_processor)
    cache2 = NodeSpec("cache_processor_2", cache_processor)
    cache3 = NodeSpec("cache_processor_3", cache_processor)

    # Aggregator
    cache_agg = NodeSpec("cache_aggregator", optimized_aggregator).after(
        "cache_processor_1", "cache_processor_2", "cache_processor_3"
    )

    graph.add(cache1)
    graph.add(cache2)
    graph.add(cache3)
    graph.add(cache_agg)

    # Execute with timing
    print("   ⏱️  Starting cached execution...")
    start_time = time.time()

    orchestrator = Orchestrator()
    result = await orchestrator.run(graph, "cache test data")

    execution_time = time.time() - start_time
    print(f"   💾 Cached execution time: {execution_time:.2f}s")
    print(f"   📊 Cache performance: {result['cache_aggregator']['performance_improvement']}")


async def demonstrate_resource_management():
    """Demonstrate resource management optimization."""
    print("\n🔧 Resource Management")
    print("-" * 35)

    # Create graph with resource-intensive nodes
    graph = DirectedGraph()

    # Add resource-intensive processors
    resource1 = NodeSpec("resource_processor_1", resource_intensive_processor)
    resource2 = NodeSpec("resource_processor_2", resource_intensive_processor)

    # Fast processor
    fast = NodeSpec("fast_processor", fast_processor)

    # Aggregator
    resource_agg = NodeSpec("resource_aggregator", optimized_aggregator).after(
        "resource_processor_1", "resource_processor_2", "fast_processor"
    )

    graph.add(resource1)
    graph.add(resource2)
    graph.add(fast)
    graph.add(resource_agg)

    # Execute with resource monitoring
    print("   ⏱️  Starting resource-intensive execution...")
    start_time = time.time()

    orchestrator = Orchestrator()
    result = await orchestrator.run(graph, "resource test")

    execution_time = time.time() - start_time
    print(f"   🔧 Resource execution time: {execution_time:.2f}s")
    print(
        f"   📊 Total processing time: {result['resource_aggregator']['total_processing_time']:.2f}s"
    )


async def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring."""
    print("\n📊 Performance Monitoring")
    print("-" * 35)

    # Create graph for monitoring
    graph = DirectedGraph()

    # Add mixed processors
    slow = NodeSpec("slow_processor", slow_processor)
    fast = NodeSpec("fast_processor", fast_processor)
    cache = NodeSpec("cache_processor", cache_processor)

    # Monitoring aggregator
    monitor_agg = NodeSpec("monitor_aggregator", optimized_aggregator).after(
        "slow_processor", "fast_processor", "cache_processor"
    )

    graph.add(slow)
    graph.add(fast)
    graph.add(cache)
    graph.add(monitor_agg)

    # Execute with monitoring
    print("   📊 Starting performance monitoring...")
    start_time = time.time()

    orchestrator = Orchestrator()
    result = await orchestrator.run(graph, "monitoring test")

    execution_time = time.time() - start_time
    print(f"   ⏱️  Total execution time: {execution_time:.2f}s")
    print(f"   📈 Processing time breakdown:")

    for node_name, node_result in result.items():
        if isinstance(node_result, dict) and "processing_time" in node_result:
            print(f"      • {node_name}: {node_result['processing_time']:.2f}s")


async def main():
    """Demonstrate performance optimization capabilities."""

    print("⚡ Example 17: Performance Optimization")
    print("=" * 40)

    print("\n🎯 This example demonstrates:")
    print("   • Parallel execution")
    print("   • Caching strategies")
    print("   • Resource management")
    print("   • Performance monitoring")
    print("   • Optimization techniques")

    # Test 1: Parallel Execution
    await demonstrate_parallel_execution()

    # Test 2: Caching Strategy
    await demonstrate_caching_strategy()

    # Test 3: Resource Management
    await demonstrate_resource_management()

    # Test 4: Performance Monitoring
    await demonstrate_performance_monitoring()

    print(f"\n🎯 Key Concepts Learned:")
    print("   ✅ Parallel Execution - Execute nodes concurrently")
    print("   ✅ Caching Strategies - Cache expensive operations")
    print("   ✅ Resource Management - Manage CPU and memory usage")
    print("   ✅ Performance Monitoring - Track execution metrics")
    print("   ✅ Optimization Techniques - Improve pipeline performance")

    print(f"\n🔗 Next: Run example 18 to learn about advanced patterns!")


if __name__ == "__main__":
    asyncio.run(main())
