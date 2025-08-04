"""
Example 21: Data Aggregation.

This example demonstrates data aggregation in hexAI:
- Fan-out and fan-in patterns
- Data collection from multiple nodes
- Aggregation strategies
- Result combination
- Multi-source data processing
"""

import asyncio
from typing import Any

from hexai.core.application.orchestrator import Orchestrator
from hexai.core.domain.dag import DirectedGraph, NodeSpec


async def data_source_1(input_data: str, **kwargs) -> dict:
    """First data source."""
    return {
        "source": "source_1",
        "data": f"data_1_{input_data}",
        "timestamp": "2024-01-01T00:00:00Z",
        "priority": "high",
    }


async def data_source_2(input_data: str, **kwargs) -> dict:
    """Second data source."""
    return {
        "source": "source_2",
        "data": f"data_2_{input_data}",
        "timestamp": "2024-01-01T00:01:00Z",
        "priority": "medium",
    }


async def data_source_3(input_data: str, **kwargs) -> dict:
    """Third data source."""
    return {
        "source": "source_3",
        "data": f"data_3_{input_data}",
        "timestamp": "2024-01-01T00:02:00Z",
        "priority": "low",
    }


async def data_source_4(input_data: str, **kwargs) -> dict:
    """Fourth data source."""
    return {
        "source": "source_4",
        "data": f"data_4_{input_data}",
        "timestamp": "2024-01-01T00:03:00Z",
        "priority": "medium",
    }


async def data_source_5(input_data: str, **kwargs) -> dict:
    """Fifth data source."""
    return {
        "source": "source_5",
        "data": f"data_5_{input_data}",
        "timestamp": "2024-01-01T00:04:00Z",
        "priority": "high",
    }


async def simple_aggregator(input_data: Any, **kwargs) -> dict:
    """Simple aggregation of all data sources."""
    results = input_data

    # Extract data from all sources
    aggregated_data = {}
    total_sources = 0

    for node_name, result in results.items():
        if isinstance(result, dict) and "source" in result:
            source_name = result["source"]
            aggregated_data[source_name] = {
                "data": result["data"],
                "timestamp": result["timestamp"],
                "priority": result["priority"],
            }
            total_sources += 1

    return {
        "aggregated_data": aggregated_data,
        "total_sources": total_sources,
        "aggregation_type": "simple",
        "aggregation_complete": True,
    }


async def priority_aggregator(input_data: Any, **kwargs) -> dict:
    """Aggregate data with priority-based ordering."""
    results = input_data

    # Collect all data with priorities
    priority_data = {"high": [], "medium": [], "low": []}

    for node_name, result in results.items():
        if isinstance(result, dict) and "priority" in result:
            priority = result["priority"]
            priority_data[priority].append(
                {
                    "source": result["source"],
                    "data": result["data"],
                    "timestamp": result["timestamp"],
                }
            )

    # Order by priority (high -> medium -> low)
    ordered_data = []
    ordered_data.extend(priority_data["high"])
    ordered_data.extend(priority_data["medium"])
    ordered_data.extend(priority_data["low"])

    return {
        "ordered_data": ordered_data,
        "high_priority_count": len(priority_data["high"]),
        "medium_priority_count": len(priority_data["medium"]),
        "low_priority_count": len(priority_data["low"]),
        "aggregation_type": "priority_based",
        "aggregation_complete": True,
    }


async def timestamp_aggregator(input_data: Any, **kwargs) -> dict:
    """Aggregate data ordered by timestamp."""
    results = input_data

    # Collect all data with timestamps
    timestamp_data = []

    for node_name, result in results.items():
        if isinstance(result, dict) and "timestamp" in result:
            timestamp_data.append(
                {
                    "source": result["source"],
                    "data": result["data"],
                    "timestamp": result["timestamp"],
                    "priority": result["priority"],
                }
            )

    # Sort by timestamp
    timestamp_data.sort(key=lambda x: x["timestamp"])

    return {
        "chronological_data": timestamp_data,
        "earliest_timestamp": timestamp_data[0]["timestamp"] if timestamp_data else None,
        "latest_timestamp": timestamp_data[-1]["timestamp"] if timestamp_data else None,
        "total_records": len(timestamp_data),
        "aggregation_type": "timestamp_based",
        "aggregation_complete": True,
    }


async def statistical_aggregator(input_data: Any, **kwargs) -> dict:
    """Aggregate data with statistical analysis."""
    results = input_data

    # Collect all data for analysis
    all_data = []
    priority_counts = {"high": 0, "medium": 0, "low": 0}

    for node_name, result in results.items():
        if isinstance(result, dict) and "source" in result:
            all_data.append(result["data"])
            priority_counts[result["priority"]] += 1

    # Calculate statistics
    total_sources = len(all_data)
    avg_data_length = sum(len(data) for data in all_data) / max(1, total_sources)

    return {
        "statistical_summary": {
            "total_sources": total_sources,
            "average_data_length": round(avg_data_length, 2),
            "priority_distribution": priority_counts,
            "data_samples": all_data[:3],  # First 3 samples
        },
        "aggregation_type": "statistical",
        "aggregation_complete": True,
    }


async def demonstrate_fan_out_fan_in():
    """Demonstrate fan-out and fan-in pattern."""
    print("\n🔄 Fan-Out and Fan-In Pattern")
    print("-" * 35)

    # Create graph with fan-out and fan-in
    graph = DirectedGraph()

    # Add data sources (fan-out)
    source1 = NodeSpec("data_source_1", data_source_1)
    source2 = NodeSpec("data_source_2", data_source_2)
    source3 = NodeSpec("data_source_3", data_source_3)
    source4 = NodeSpec("data_source_4", data_source_4)
    source5 = NodeSpec("data_source_5", data_source_5)

    # Add aggregator (fan-in)
    aggregator = NodeSpec("simple_aggregator", simple_aggregator).after(
        "data_source_1", "data_source_2", "data_source_3", "data_source_4", "data_source_5"
    )

    graph.add(source1)
    graph.add(source2)
    graph.add(source3)
    graph.add(source4)
    graph.add(source5)
    graph.add(aggregator)

    # Execute
    print("   🚀 Starting fan-out and fan-in execution...")

    orchestrator = Orchestrator()
    result = await orchestrator.run(graph, "test_data")

    print("   ✅ Fan-out and fan-in completed")
    print(f"   📊 Total sources aggregated: {result['simple_aggregator']['total_sources']}")
    print(
        f"   📋 Aggregated data keys: {list(result['simple_aggregator']['aggregated_data'].keys())}"
    )


async def demonstrate_priority_aggregation():
    """Demonstrate priority-based aggregation."""
    print("\n🎯 Priority-Based Aggregation")
    print("-" * 35)

    # Create graph with priority aggregation
    graph = DirectedGraph()

    # Add data sources
    source1 = NodeSpec("data_source_1", data_source_1)
    source2 = NodeSpec("data_source_2", data_source_2)
    source3 = NodeSpec("data_source_3", data_source_3)
    source4 = NodeSpec("data_source_4", data_source_4)
    source5 = NodeSpec("data_source_5", data_source_5)

    # Add priority aggregator
    priority_agg = NodeSpec("priority_aggregator", priority_aggregator).after(
        "data_source_1", "data_source_2", "data_source_3", "data_source_4", "data_source_5"
    )

    graph.add(source1)
    graph.add(source2)
    graph.add(source3)
    graph.add(source4)
    graph.add(source5)
    graph.add(priority_agg)

    # Execute
    print("   🚀 Starting priority-based aggregation...")

    orchestrator = Orchestrator()
    result = await orchestrator.run(graph, "priority_test")

    print("   ✅ Priority aggregation completed")
    print(f"   📊 High priority: {result['priority_aggregator']['high_priority_count']}")
    print(f"   📊 Medium priority: {result['priority_aggregator']['medium_priority_count']}")
    print(f"   📊 Low priority: {result['priority_aggregator']['low_priority_count']}")


async def demonstrate_timestamp_aggregation():
    """Demonstrate timestamp-based aggregation."""
    print("\n⏰ Timestamp-Based Aggregation")
    print("-" * 35)

    # Create graph with timestamp aggregation
    graph = DirectedGraph()

    # Add data sources
    source1 = NodeSpec("data_source_1", data_source_1)
    source2 = NodeSpec("data_source_2", data_source_2)
    source3 = NodeSpec("data_source_3", data_source_3)
    source4 = NodeSpec("data_source_4", data_source_4)
    source5 = NodeSpec("data_source_5", data_source_5)

    # Add timestamp aggregator
    timestamp_agg = NodeSpec("timestamp_aggregator", timestamp_aggregator).after(
        "data_source_1", "data_source_2", "data_source_3", "data_source_4", "data_source_5"
    )

    graph.add(source1)
    graph.add(source2)
    graph.add(source3)
    graph.add(source4)
    graph.add(source5)
    graph.add(timestamp_agg)

    # Execute
    print("   🚀 Starting timestamp-based aggregation...")

    orchestrator = Orchestrator()
    result = await orchestrator.run(graph, "timestamp_test")

    print("   ✅ Timestamp aggregation completed")
    print(f"   📊 Total records: {result['timestamp_aggregator']['total_records']}")
    print(f"   ⏰ Earliest: {result['timestamp_aggregator']['earliest_timestamp']}")
    print(f"   ⏰ Latest: {result['timestamp_aggregator']['latest_timestamp']}")


async def demonstrate_statistical_aggregation():
    """Demonstrate statistical aggregation."""
    print("\n📈 Statistical Aggregation")
    print("-" * 35)

    # Create graph with statistical aggregation
    graph = DirectedGraph()

    # Add data sources
    source1 = NodeSpec("data_source_1", data_source_1)
    source2 = NodeSpec("data_source_2", data_source_2)
    source3 = NodeSpec("data_source_3", data_source_3)
    source4 = NodeSpec("data_source_4", data_source_4)
    source5 = NodeSpec("data_source_5", data_source_5)

    # Add statistical aggregator
    statistical_agg = NodeSpec("statistical_aggregator", statistical_aggregator).after(
        "data_source_1", "data_source_2", "data_source_3", "data_source_4", "data_source_5"
    )

    graph.add(source1)
    graph.add(source2)
    graph.add(source3)
    graph.add(source4)
    graph.add(source5)
    graph.add(statistical_agg)

    # Execute
    print("   🚀 Starting statistical aggregation...")

    orchestrator = Orchestrator()
    result = await orchestrator.run(graph, "statistical_test")

    print("   ✅ Statistical aggregation completed")
    summary = result["statistical_aggregator"]["statistical_summary"]
    print(f"   📊 Total sources: {summary['total_sources']}")
    print(f"   📊 Average data length: {summary['average_data_length']}")
    print(f"   📊 Priority distribution: {summary['priority_distribution']}")


async def demonstrate_complex_aggregation():
    """Demonstrate complex multi-level aggregation."""
    print("\n🔗 Complex Multi-Level Aggregation")
    print("-" * 35)

    # Create graph with multiple aggregation levels
    graph = DirectedGraph()

    # Level 1: Data sources
    source1 = NodeSpec("data_source_1", data_source_1)
    source2 = NodeSpec("data_source_2", data_source_2)
    source3 = NodeSpec("data_source_3", data_source_3)
    source4 = NodeSpec("data_source_4", data_source_4)
    source5 = NodeSpec("data_source_5", data_source_5)

    # Level 2: First-level aggregators
    simple_agg = NodeSpec("simple_aggregator", simple_aggregator).after(
        "data_source_1", "data_source_2", "data_source_3"
    )
    priority_agg = NodeSpec("priority_aggregator", priority_aggregator).after(
        "data_source_4", "data_source_5"
    )

    # Level 3: Final aggregator
    final_agg = NodeSpec("statistical_aggregator", statistical_aggregator).after(
        "simple_aggregator", "priority_aggregator"
    )

    graph.add(source1)
    graph.add(source2)
    graph.add(source3)
    graph.add(source4)
    graph.add(source5)
    graph.add(simple_agg)
    graph.add(priority_agg)
    graph.add(final_agg)

    # Execute
    print("   🚀 Starting complex multi-level aggregation...")

    orchestrator = Orchestrator()
    result = await orchestrator.run(graph, "complex_test")

    print("   ✅ Complex aggregation completed")
    print(f"   📊 Final aggregation type: {result['statistical_aggregator']['aggregation_type']}")
    print(
        f"   📊 Total sources in final: {result['statistical_aggregator']['statistical_summary']['total_sources']}"
    )


async def main():
    """Demonstrate data aggregation capabilities."""

    print("📊 Example 12: Data Aggregation")
    print("=" * 35)

    print("\n🎯 This example demonstrates:")
    print("   • Fan-out and fan-in patterns")
    print("   • Data collection from multiple nodes")
    print("   • Aggregation strategies")
    print("   • Result combination")
    print("   • Multi-source data processing")

    # Test 1: Fan-Out and Fan-In
    await demonstrate_fan_out_fan_in()

    # Test 2: Priority Aggregation
    await demonstrate_priority_aggregation()

    # Test 3: Timestamp Aggregation
    await demonstrate_timestamp_aggregation()

    # Test 4: Statistical Aggregation
    await demonstrate_statistical_aggregation()

    # Test 5: Complex Multi-Level Aggregation
    await demonstrate_complex_aggregation()

    print(f"\n🎯 Key Concepts Learned:")
    print("   ✅ Fan-Out Fan-In - Distribute and collect data")
    print("   ✅ Data Aggregation - Combine results from multiple nodes")
    print("   ✅ Priority Ordering - Aggregate by priority levels")
    print("   ✅ Timestamp Ordering - Aggregate chronologically")
    print("   ✅ Statistical Analysis - Analyze aggregated data")
    print("   ✅ Multi-Level Aggregation - Complex aggregation patterns")

    print(f"\n🔗 This completes the data aggregation demonstration!")


if __name__ == "__main__":
    asyncio.run(main())
