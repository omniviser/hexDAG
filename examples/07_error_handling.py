"""
Example 07: Error Handling Patterns

This example demonstrates various error handling patterns in hexAI:
- Graceful degradation
- Retry mechanisms
- Error recovery
- Circuit breaker patterns
- Error aggregation
"""

import asyncio
import random
from typing import Any

from hexai.core.application.orchestrator import Orchestrator
from hexai.core.domain.dag import DirectedGraph, NodeSpec


# Simulated external services that can fail
class UnreliableService:
    """Simulates an unreliable external service."""

    def __init__(self, failure_rate: float = 0.3):
        self.failure_rate = failure_rate
        self.call_count = 0

    async def call(self, data: str) -> dict:
        """Simulate an unreliable service call."""
        self.call_count += 1

        if random.random() < self.failure_rate:  # nosec B311 - example code only
            raise Exception(f"Service failed on call {self.call_count}")

        return {"result": f"Processed: {data}", "call_count": self.call_count, "status": "success"}


# Error handling functions
async def reliable_processor(input_data: str, **kwargs) -> dict:
    """Process data with retry mechanism."""
    service = kwargs.get("unreliable_service")
    max_retries = 3

    for attempt in range(max_retries):
        try:
            result = await service.call(input_data)
            return {"data": result, "attempts": attempt + 1, "status": "success"}
        except Exception as e:
            if attempt == max_retries - 1:
                return {"error": str(e), "attempts": attempt + 1, "status": "failed"}
            await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff

    return {"status": "failed", "error": "Max retries exceeded"}


async def graceful_degradation(input_data: Any, **kwargs) -> dict:
    """Process data with graceful degradation."""
    primary_service = kwargs.get("primary_service")
    fallback_service = kwargs.get("fallback_service")

    try:
        # Try primary service
        result = await primary_service.call(input_data)
        return {"data": result, "source": "primary", "status": "success"}
    except Exception as e:
        try:
            # Fallback to secondary service
            result = await fallback_service.call(input_data)
            return {"data": result, "source": "fallback", "status": "degraded"}
        except Exception as fallback_error:
            return {"error": f"Primary: {e}, Fallback: {fallback_error}", "status": "failed"}


async def circuit_breaker_processor(input_data: str, **kwargs) -> dict:
    """Process data with circuit breaker pattern."""
    service = kwargs.get("circuit_service")
    circuit = kwargs.get("circuit_breaker")

    if circuit.is_open():
        return {"status": "circuit_open", "message": "Circuit breaker is open, skipping call"}

    try:
        result = await service.call(input_data)
        circuit.record_success()
        return {"data": result, "status": "success"}
    except Exception as e:
        circuit.record_failure()
        return {"error": str(e), "status": "failed"}


async def error_aggregator(input_data: Any, **kwargs) -> dict:
    """Aggregate results and handle partial failures."""
    results = input_data

    successful_results = []
    failed_results = []

    for node_name, result in results.items():
        if result.get("status") == "success":
            successful_results.append({"node": node_name, "data": result.get("data")})
        else:
            failed_results.append({"node": node_name, "error": result.get("error")})

    return {
        "summary": {
            "total_nodes": len(results),
            "successful": len(successful_results),
            "failed": len(failed_results),
        },
        "successful_results": successful_results,
        "failed_results": failed_results,
        "overall_status": "partial" if failed_results else "success",
    }


# Circuit Breaker Implementation
class CircuitBreaker:
    """Simple circuit breaker implementation."""

    def __init__(self, failure_threshold: int = 3, timeout: float = 5.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.state == "OPEN":
            if (
                self.last_failure_time
                and (asyncio.get_event_loop().time() - self.last_failure_time) > self.timeout
            ):
                self.state = "HALF_OPEN"
                return False
            return True
        return False

    def record_success(self) -> None:
        """Record a successful call."""
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = asyncio.get_event_loop().time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


async def main():
    """Demonstrate error handling patterns."""

    print("🛡️ Example 07: Error Handling Patterns")
    print("=" * 45)

    print("\n🎯 This example demonstrates:")
    print("   • Retry mechanisms with exponential backoff")
    print("   • Graceful degradation with fallback services")
    print("   • Circuit breaker pattern")
    print("   • Error aggregation and partial failure handling")

    # Create unreliable services
    unreliable_service = UnreliableService(failure_rate=0.4)
    primary_service = UnreliableService(failure_rate=0.2)
    fallback_service = UnreliableService(failure_rate=0.1)
    circuit_service = UnreliableService(failure_rate=0.5)

    # Create circuit breaker
    circuit_breaker = CircuitBreaker(failure_threshold=2, timeout=1.0)

    # Create orchestrator with services
    orchestrator = Orchestrator(
        ports={
            "unreliable_service": unreliable_service,
            "primary_service": primary_service,
            "fallback_service": fallback_service,
            "circuit_service": circuit_service,
            "circuit_breaker": circuit_breaker,
        }
    )

    # Test 1: Retry Mechanism
    print("\n🔄 Test 1: Retry Mechanism")
    print("-" * 30)

    retry_graph = DirectedGraph()
    retry_node = NodeSpec("retry_processor", reliable_processor)
    retry_graph.add(retry_node)

    retry_result = await orchestrator.run(retry_graph, "test data")
    print(f"   Result: {retry_result['retry_processor']['status']}")
    print(f"   Attempts: {retry_result['retry_processor'].get('attempts', 0)}")

    # Test 2: Graceful Degradation
    print("\n🔄 Test 2: Graceful Degradation")
    print("-" * 30)

    degradation_graph = DirectedGraph()
    degradation_node = NodeSpec("degradation_processor", graceful_degradation)
    degradation_graph.add(degradation_node)

    degradation_result = await orchestrator.run(degradation_graph, "test data")
    print(f"   Result: {degradation_result['degradation_processor']['status']}")
    print(f"   Source: {degradation_result['degradation_processor'].get('source', 'unknown')}")

    # Test 3: Circuit Breaker
    print("\n🔄 Test 3: Circuit Breaker")
    print("-" * 30)

    circuit_graph = DirectedGraph()
    circuit_node = NodeSpec("circuit_processor", circuit_breaker_processor)
    circuit_graph.add(circuit_node)

    # Run multiple times to trigger circuit breaker
    for i in range(5):
        circuit_result = await orchestrator.run(circuit_graph, f"test data {i}")
        print(f"   Call {i + 1}: {circuit_result['circuit_processor']['status']}")

    # Test 4: Error Aggregation
    print("\n🔄 Test 4: Error Aggregation")
    print("-" * 30)

    # Create a complex graph with multiple nodes that can fail
    aggregation_graph = DirectedGraph()

    # Add multiple processing nodes
    for i in range(3):
        node = NodeSpec(f"processor_{i}", reliable_processor)
        aggregation_graph.add(node)

    # Add aggregation node
    aggregator = NodeSpec("aggregator", error_aggregator).after(
        "processor_0", "processor_1", "processor_2"
    )
    aggregation_graph.add(aggregator)

    aggregation_result = await orchestrator.run(aggregation_graph, "test data")

    summary = aggregation_result["aggregator"]["summary"]
    print(f"   Total nodes: {summary['total_nodes']}")
    print(f"   Successful: {summary['successful']}")
    print(f"   Failed: {summary['failed']}")
    print(f"   Overall status: {aggregation_result['aggregator']['overall_status']}")

    # Show detailed error analysis
    print("\n📊 Error Analysis:")
    print("   • Retry Mechanism: Handles transient failures")
    print("   • Graceful Degradation: Provides fallback options")
    print("   • Circuit Breaker: Prevents cascade failures")
    print("   • Error Aggregation: Manages partial failures")

    print("\n🎯 Key Concepts Learned:")
    print("   ✅ Retry Patterns - Handle transient failures")
    print("   ✅ Graceful Degradation - Provide fallback options")
    print("   ✅ Circuit Breaker - Prevent cascade failures")
    print("   ✅ Error Aggregation - Handle partial failures")
    print("   ✅ Resilience Patterns - Build robust systems")

    print("\n🔗 Next: Run example 08 to learn about function nodes!")


if __name__ == "__main__":
    asyncio.run(main())
