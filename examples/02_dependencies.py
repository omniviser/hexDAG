#!/usr/bin/env python3
"""
🔧 Example 02: Understanding Dependencies and Parallel Execution.

This example teaches:
- Complex dependency patterns
- Parallel execution within waves
- Fan-out and fan-in patterns
- Execution wave analysis

Run: python examples/02_dependencies.py
"""

import asyncio
import time
from typing import Any

from hexai.core.application.orchestrator import Orchestrator
from hexai.core.domain.dag import DirectedGraph, NodeSpec


async def data_loader(input_data: str) -> dict:
    """Load initial data."""
    await asyncio.sleep(0.1)  # Simulate I/O
    return {"query": input_data, "timestamp": time.time(), "status": "loaded"}


async def fetch_user_data(input_data: dict) -> dict:
    """Fetch user information (parallel with product data)."""
    await asyncio.sleep(0.2)  # Simulate API call
    query = input_data.get("query", "")
    return {"user_data": f"User info for: {query}", "user_count": 150, "source": "user_service"}


async def fetch_product_data(input_data: dict) -> dict:
    """Fetch product information (parallel with user data)."""
    await asyncio.sleep(0.15)  # Simulate database query
    query = input_data.get("query", "")
    return {
        "product_data": f"Products for: {query}",
        "product_count": 45,
        "source": "product_service",
    }


async def fetch_analytics_data(input_data: dict) -> dict:
    """Fetch analytics (parallel with user and product data)."""
    await asyncio.sleep(0.3)  # Simulate complex computation
    return {
        "analytics": "Monthly trends",
        "metrics": {"conversion": 0.12, "retention": 0.85},
        "source": "analytics_service",
    }


async def combine_user_product(input_data: Any) -> dict:
    """Combine user and product data."""
    # Extract dependency results from input_data (which contains the aggregated results)
    fetch_users = input_data.get("fetch_users", {})
    fetch_products = input_data.get("fetch_products", {})

    return {
        "combined_data": {
            "users": fetch_users.get("user_count", 0),
            "products": fetch_products.get("product_count", 0),
        },
        "sources": [fetch_users.get("source"), fetch_products.get("source")],
        "status": "combined",
    }


async def final_report(input_data: Any) -> dict:
    """Generate final report from all data sources."""
    # Extract dependency results from input_data
    combine_data = input_data.get("combine_data", {})
    fetch_analytics = input_data.get("fetch_analytics", {})
    data_loader = input_data.get("data_loader", {})

    return {
        "report": {
            "original_query": data_loader.get("query"),
            "data_summary": combine_data.get("combined_data"),
            "analytics": fetch_analytics.get("metrics"),
            "timestamp": data_loader.get("timestamp"),
            "total_sources": len(combine_data.get("sources", [])) + 1,
        },
        "execution_complete": True,
    }


async def main():
    """Demonstrate complex dependencies and parallel execution."""

    print("🔧 Example 02: Dependencies and Parallel Execution")
    print("=" * 55)

    # Create graph
    graph = DirectedGraph()

    print("\n📊 Building complex dependency graph...")

    # Wave 1: Single data loader
    loader = NodeSpec("data_loader", data_loader)
    graph.add(loader)
    print("   ✅ Wave 1: data_loader")

    # Wave 2: Three parallel data fetchers (fan-out pattern)
    user_fetcher = NodeSpec("fetch_users", fetch_user_data).after("data_loader")
    product_fetcher = NodeSpec("fetch_products", fetch_product_data).after("data_loader")
    analytics_fetcher = NodeSpec("fetch_analytics", fetch_analytics_data).after("data_loader")

    graph.add(user_fetcher)
    graph.add(product_fetcher)
    graph.add(analytics_fetcher)
    print("   ✅ Wave 2: fetch_users, fetch_products, fetch_analytics (PARALLEL)")

    # Wave 3: Combine some results
    combiner = NodeSpec("combine_data", combine_user_product).after("fetch_users", "fetch_products")
    graph.add(combiner)
    print("   ✅ Wave 3: combine_data (waits for users + products)")

    # Wave 4: Final report (fan-in pattern)
    reporter = NodeSpec("final_report", final_report).after(
        "combine_data", "fetch_analytics", "data_loader"
    )
    graph.add(reporter)
    print("   ✅ Wave 4: final_report (waits for combined + analytics + original)")

    # Validate and show execution plan
    print("\n🔍 Validating dependency graph...")
    graph.validate()

    print("\n🌊 Execution Wave Analysis:")
    waves = graph.waves()
    for i, wave in enumerate(waves, 1):
        if len(wave) == 1:
            print(f"   Wave {i}: {wave[0]} (sequential)")
        else:
            print(f"   Wave {i}: {', '.join(wave)} (parallel)")

    print(f"\n⚡ Total waves: {len(waves)} (fewer waves = more parallelism)")

    # Execute and time it
    print("\n🚀 Executing with timing...")
    orchestrator = Orchestrator()

    start_time = time.time()
    results = await orchestrator.run(graph, "customer analytics request")
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"   ⏱️  Total execution time: {execution_time:.3f} seconds")

    # Show key results
    print("\n📋 Key Results:")
    print(f"   📊 Data Loader: {results['data_loader']['status']}")
    print(f"   👥 Users Found: {results['fetch_users']['user_count']}")
    print(f"   📦 Products Found: {results['fetch_products']['product_count']}")
    print(f"   📈 Analytics: {results['fetch_analytics']['metrics']}")
    print(f"   📝 Final Report Sources: {results['final_report']['report']['total_sources']}")

    print("\n🎯 Key Concepts Learned:")
    print("   ✅ Fan-out: One node triggers multiple parallel nodes")
    print("   ✅ Fan-in: Multiple nodes feed into one final node")
    print("   ✅ Waves: Parallel execution within each wave")
    print("   ✅ Dependencies: .after() with multiple nodes")
    print("   ✅ Performance: Parallel execution reduces total time")

    print("\n🔗 Next: Run example 03 to learn about validation!")


if __name__ == "__main__":
    asyncio.run(main())
