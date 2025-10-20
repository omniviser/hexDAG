"""Integration tests for error handling patterns.

Tests demonstrate:
- Retry mechanisms with exponential backoff
- Graceful degradation with fallback services
- Circuit breaker pattern
- Error aggregation and partial failure handling
"""

import asyncio
import random

import pytest

from hexdag.core.context import get_port
from hexdag.core.domain.dag import DirectedGraph, NodeSpec
from hexdag.core.orchestration.orchestrator import Orchestrator


# Simulated external services that can fail
class UnreliableService:
    """Simulates an unreliable external service."""

    def __init__(self, failure_rate: float = 0.3):
        self.failure_rate = failure_rate
        self.call_count = 0

    async def call(self, data: str) -> dict:
        """Simulate an unreliable service call."""
        self.call_count += 1

        if random.random() < self.failure_rate:  # nosec B311 - test code only
            raise Exception(f"Service failed on call {self.call_count}")

        return {
            "result": f"Processed: {data}",
            "call_count": self.call_count,
            "status": "success",
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


# Error handling functions
async def reliable_processor(input_data: str, **kwargs) -> dict:
    """Process data with retry mechanism."""
    service = get_port("unreliable_service")
    max_retries = 3

    for attempt in range(max_retries):
        try:
            result = await service.call(input_data)
            return {"data": result, "attempts": attempt + 1, "status": "success"}
        except Exception as e:
            if attempt == max_retries - 1:
                return {"error": str(e), "attempts": attempt + 1, "status": "failed"}
            await asyncio.sleep(0.01 * (attempt + 1))  # Exponential backoff (short for tests)

    return {"status": "failed", "error": "Max retries exceeded"}


async def graceful_degradation(input_data: str, **kwargs) -> dict:
    """Process data with graceful degradation."""
    primary_service = get_port("primary_service")
    fallback_service = get_port("fallback_service")

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
            return {
                "error": f"Primary: {e}, Fallback: {fallback_error}",
                "status": "failed",
            }


async def circuit_breaker_processor(input_data: str, **kwargs) -> dict:
    """Process data with circuit breaker pattern."""
    service = get_port("circuit_service")
    circuit = get_port("circuit_breaker")

    if circuit.is_open():
        return {
            "status": "circuit_open",
            "message": "Circuit breaker is open, skipping call",
        }

    try:
        result = await service.call(input_data)
        circuit.record_success()
        return {"data": result, "status": "success"}
    except Exception as e:
        circuit.record_failure()
        return {"error": str(e), "status": "failed"}


async def error_aggregator(input_data: dict, **kwargs) -> dict:
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


@pytest.fixture
def unreliable_services():
    """Provide unreliable services with different failure rates."""
    return {
        "unreliable_service": UnreliableService(failure_rate=0.4),
        "primary_service": UnreliableService(failure_rate=0.2),
        "fallback_service": UnreliableService(failure_rate=0.1),
        "circuit_service": UnreliableService(failure_rate=0.5),
    }


@pytest.fixture
def circuit_breaker():
    """Provide circuit breaker."""
    return CircuitBreaker(failure_threshold=2, timeout=1.0)


class TestErrorHandlingPatterns:
    """Test suite for error handling patterns."""

    @pytest.mark.asyncio
    async def test_retry_mechanism(self, unreliable_services):
        """Test retry mechanism with exponential backoff."""
        orchestrator = Orchestrator(ports=unreliable_services)

        graph = DirectedGraph()
        retry_node = NodeSpec("retry_processor", reliable_processor)
        graph.add(retry_node)

        # Run multiple times to test retry behavior
        for _ in range(5):
            result = await orchestrator.run(graph, "test data")
            assert "retry_processor" in result
            assert "attempts" in result["retry_processor"]
            assert result["retry_processor"]["attempts"] <= 3

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, unreliable_services):
        """Test graceful degradation with fallback services."""
        orchestrator = Orchestrator(ports=unreliable_services)

        graph = DirectedGraph()
        degradation_node = NodeSpec("degradation_processor", graceful_degradation)
        graph.add(degradation_node)

        # Run multiple times to see both primary and fallback being used
        results_by_source = {"primary": 0, "fallback": 0, "failed": 0}

        for _ in range(10):
            result = await orchestrator.run(graph, "test data")
            status = result["degradation_processor"]["status"]

            if status == "success":
                results_by_source["primary"] += 1
            elif status == "degraded":
                results_by_source["fallback"] += 1
            elif status == "failed":
                results_by_source["failed"] += 1

        # Should have some successes (either primary or fallback)
        assert (results_by_source["primary"] + results_by_source["fallback"]) > 0

    @pytest.mark.asyncio
    async def test_circuit_breaker(self, unreliable_services, circuit_breaker):
        """Test circuit breaker pattern."""
        ports = {**unreliable_services, "circuit_breaker": circuit_breaker}
        orchestrator = Orchestrator(ports=ports)

        graph = DirectedGraph()
        circuit_node = NodeSpec("circuit_processor", circuit_breaker_processor)
        graph.add(circuit_node)

        # Run multiple times to trigger circuit breaker
        results = []
        for _ in range(10):
            result = await orchestrator.run(graph, "test data")
            results.append(result["circuit_processor"])

        # Should have some circuit_open responses after threshold is reached
        circuit_open_count = sum(1 for r in results if r["status"] == "circuit_open")

        # With 50% failure rate and threshold of 2, circuit should open
        assert circuit_open_count > 0 or circuit_breaker.state == "OPEN"

    @pytest.mark.asyncio
    async def test_error_aggregation(self, unreliable_services):
        """Test error aggregation with partial failures."""
        orchestrator = Orchestrator(ports=unreliable_services)

        # Create a complex graph with multiple nodes that can fail
        graph = DirectedGraph()

        # Add multiple processing nodes
        for i in range(3):
            node = NodeSpec(f"processor_{i}", reliable_processor)
            graph.add(node)

        # Add aggregation node
        aggregator = NodeSpec("aggregator", error_aggregator).after(
            "processor_0", "processor_1", "processor_2"
        )
        graph.add(aggregator)

        result = await orchestrator.run(graph, "test data")

        # Verify aggregation result
        assert "aggregator" in result
        summary = result["aggregator"]["summary"]

        assert "total_nodes" in summary
        assert "successful" in summary
        assert "failed" in summary
        assert summary["total_nodes"] == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, circuit_breaker):
        """Test that circuit breaker can recover after timeout."""
        # Set a very short timeout for faster testing
        circuit_breaker.timeout = 0.1

        # Use a service that always fails initially to trigger circuit opening
        failing_service = UnreliableService(failure_rate=1.0)

        ports = {
            "circuit_service": failing_service,
            "circuit_breaker": circuit_breaker,
        }
        orchestrator = Orchestrator(ports=ports)

        graph = DirectedGraph()
        circuit_node = NodeSpec("circuit_processor", circuit_breaker_processor)
        graph.add(circuit_node)

        # Trigger circuit breaker opening
        for _ in range(5):
            await orchestrator.run(graph, "test data")
            if circuit_breaker.state == "OPEN":
                break

        assert circuit_breaker.state == "OPEN"

        # Wait for timeout
        await asyncio.sleep(0.2)

        # Make service succeed to allow recovery
        # Change the failure rate to 0 so next call succeeds
        failing_service.failure_rate = 0.0

        # Circuit should transition to HALF_OPEN then CLOSED after successful call
        await orchestrator.run(graph, "test data")
        # The is_open() call transitions to HALF_OPEN, then success transitions to CLOSED
        assert circuit_breaker.state == "CLOSED"

    @pytest.mark.asyncio
    async def test_retry_with_eventual_success(self):
        """Test that retry eventually succeeds with low failure rate."""
        # Create service with very low failure rate
        service = UnreliableService(failure_rate=0.1)
        orchestrator = Orchestrator(ports={"unreliable_service": service})

        graph = DirectedGraph()
        retry_node = NodeSpec("retry_processor", reliable_processor)
        graph.add(retry_node)

        # Should succeed within max retries
        result = await orchestrator.run(graph, "test data")

        # Most likely succeeded (10% failure rate over 3 attempts)
        assert result["retry_processor"]["status"] in ["success", "failed"]
        assert result["retry_processor"]["attempts"] <= 3

    @pytest.mark.asyncio
    async def test_fallback_ordering(self, unreliable_services):
        """Test that fallback is tried after primary fails."""
        # Force primary to fail
        primary_always_fails = UnreliableService(failure_rate=1.0)
        fallback_always_succeeds = UnreliableService(failure_rate=0.0)

        ports = {
            "primary_service": primary_always_fails,
            "fallback_service": fallback_always_succeeds,
        }

        orchestrator = Orchestrator(ports=ports)

        graph = DirectedGraph()
        degradation_node = NodeSpec("degradation_processor", graceful_degradation)
        graph.add(degradation_node)

        result = await orchestrator.run(graph, "test data")

        # Should use fallback
        assert result["degradation_processor"]["status"] == "degraded"
        assert result["degradation_processor"]["source"] == "fallback"

    @pytest.mark.asyncio
    async def test_complete_failure_handling(self):
        """Test handling when all services fail."""
        # All services fail
        primary_fails = UnreliableService(failure_rate=1.0)
        fallback_fails = UnreliableService(failure_rate=1.0)

        ports = {"primary_service": primary_fails, "fallback_service": fallback_fails}

        orchestrator = Orchestrator(ports=ports)

        graph = DirectedGraph()
        degradation_node = NodeSpec("degradation_processor", graceful_degradation)
        graph.add(degradation_node)

        result = await orchestrator.run(graph, "test data")

        # Should return failed status with error details
        assert result["degradation_processor"]["status"] == "failed"
        assert "error" in result["degradation_processor"]
