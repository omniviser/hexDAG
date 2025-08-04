#!/usr/bin/env python3
"""
🚀 Example 19: Complex Real-World Workflow.

This example teaches:
- Complex multi-stage workflows
- Error handling and recovery
- Performance optimization
- Real-world patterns and best practices

Run: python examples/19_complex_workflow.py
"""

import asyncio
import time
from typing import Any

from pydantic import BaseModel, Field

from hexai.core.application.orchestrator import Orchestrator
from hexai.core.domain.dag import DirectedGraph, NodeSpec
from hexai.validation import coerce_validator, strict_validator


class CustomerData(BaseModel):
    """Customer data model."""

    customer_id: str = Field(..., description="Unique customer identifier")
    name: str = Field(..., min_length=1, description="Customer name")
    email: str = Field(..., description="Customer email")
    age: int = Field(..., ge=0, le=150, description="Customer age")
    segment: str = Field(default="standard", description="Customer segment")


class ProductData(BaseModel):
    """Product data model."""

    product_id: str = Field(..., description="Unique product identifier")
    name: str = Field(..., description="Product name")
    category: str = Field(..., description="Product category")
    price: float = Field(..., ge=0, description="Product price")
    stock_level: int = Field(..., ge=0, description="Available stock")


class OrderData(BaseModel):
    """Order data model."""

    order_id: str = Field(..., description="Unique order identifier")
    customer_id: str = Field(..., description="Customer who placed order")
    product_id: str = Field(..., description="Product ordered")
    quantity: int = Field(..., ge=1, description="Order quantity")
    order_date: str = Field(..., description="Order date")


class AnalysisResult(BaseModel):
    """Analysis result model."""

    customer_insights: dict[str, Any] = Field(..., description="Customer analysis")
    product_recommendations: list[dict[str, Any]] = Field(
        ..., description="Product recommendations"
    )
    risk_assessment: dict[str, Any] = Field(..., description="Risk analysis")
    optimization_suggestions: list[str] = Field(..., description="Optimization suggestions")


# Data loading functions
async def load_customer_data(input_data: dict) -> dict:
    """Load customer data from external source."""
    await asyncio.sleep(0.1)  # Simulate API call

    customer_id = input_data.get("customer_id", "CUST001")

    # Simulate customer data
    customers = {
        "CUST001": {
            "name": "Alice Johnson",
            "email": "alice@example.com",
            "age": 28,
            "segment": "premium",
        },
        "CUST002": {
            "name": "Bob Smith",
            "email": "bob@example.com",
            "age": 35,
            "segment": "standard",
        },
        "CUST003": {
            "name": "Carol Davis",
            "email": "carol@example.com",
            "age": 42,
            "segment": "premium",
        },
    }

    customer = customers.get(
        customer_id,
        {"name": "Unknown", "email": "unknown@example.com", "age": 0, "segment": "standard"},
    )

    return {
        "customer_data": CustomerData(customer_id=customer_id, **customer),
        "load_timestamp": time.time(),
        "source": "customer_database",
    }


async def load_product_data(input_data: dict) -> dict:
    """Load product data from external source."""
    await asyncio.sleep(0.15)  # Simulate database query

    # Simulate product catalog
    products = {
        "PROD001": {
            "name": "Laptop",
            "category": "Electronics",
            "price": 999.99,
            "stock_level": 50,
        },
        "PROD002": {"name": "Coffee Maker", "category": "Home", "price": 89.99, "stock_level": 200},
        "PROD003": {
            "name": "Running Shoes",
            "category": "Sports",
            "price": 129.99,
            "stock_level": 75,
        },
    }

    product_id = input_data.get("product_id", "PROD001")
    product = products.get(
        product_id,
        {"name": "Unknown Product", "category": "Unknown", "price": 0.0, "stock_level": 0},
    )

    return {
        "product_data": ProductData(product_id=product_id, **product),
        "load_timestamp": time.time(),
        "source": "product_catalog",
    }


async def load_order_history(input_data: dict) -> dict:
    """Load order history for analysis."""
    await asyncio.sleep(0.2)  # Simulate complex query

    customer_id = input_data.get("customer_id", "CUST001")

    # Simulate order history
    orders = {
        "CUST001": [
            {
                "order_id": "ORD001",
                "product_id": "PROD001",
                "quantity": 1,
                "order_date": "2024-01-15",
            },
            {
                "order_id": "ORD002",
                "product_id": "PROD002",
                "quantity": 2,
                "order_date": "2024-02-01",
            },
        ],
        "CUST002": [
            {
                "order_id": "ORD003",
                "product_id": "PROD003",
                "quantity": 1,
                "order_date": "2024-01-20",
            }
        ],
        "CUST003": [
            {
                "order_id": "ORD004",
                "product_id": "PROD001",
                "quantity": 1,
                "order_date": "2024-01-10",
            },
            {
                "order_id": "ORD005",
                "product_id": "PROD002",
                "quantity": 1,
                "order_date": "2024-01-25",
            },
        ],
    }

    customer_orders = orders.get(customer_id, [])
    order_data = [OrderData(customer_id=customer_id, **order) for order in customer_orders]

    return {
        "order_history": order_data,
        "total_orders": len(order_data),
        "load_timestamp": time.time(),
        "source": "order_database",
    }


# Analysis functions
async def analyze_customer_behavior(input_data: Any, **kwargs) -> dict:
    """Analyze customer behavior and generate insights."""
    await asyncio.sleep(0.3)  # Simulate complex analysis

    # Extract data from input_data dictionary
    customer_data = input_data.get("load_customer_data", {})
    order_history = input_data.get("load_order_history", {})

    customer = customer_data.get("customer_data")
    orders = order_history.get("order_history", [])

    # Calculate insights
    total_spent = sum(order.quantity * 100 for order in orders)  # Simplified calculation
    order_frequency = len(orders) / max(1, (time.time() - time.time() + 30))  # Simplified

    # Determine customer value
    if total_spent > 1000:
        value_tier = "high"
    elif total_spent > 500:
        value_tier = "medium"
    else:
        value_tier = "low"

    return {
        "customer_insights": {
            "value_tier": value_tier,
            "total_spent": total_spent,
            "order_frequency": order_frequency,
            "loyalty_score": min(1.0, len(orders) / 10),
            "preferred_categories": list({order.product_id[:4] for order in orders}),
        },
        "analysis_timestamp": time.time(),
        "confidence_score": 0.85,
    }


async def generate_product_recommendations(input_data: Any, **kwargs) -> dict:
    """Generate personalized product recommendations."""
    await asyncio.sleep(0.25)  # Simulate ML inference

    # Extract data from input_data dictionary
    customer_analysis = input_data.get("analyze_customer", {})
    product_data = input_data.get("load_product_data", {})

    insights = customer_analysis.get("customer_insights", {})
    product = product_data.get("product_data")

    # Simple recommendation logic
    value_tier = insights.get("value_tier", "low")
    preferred_categories = insights.get("preferred_categories", [])

    recommendations = []

    # Generate recommendations based on customer value and preferences
    if value_tier == "high":
        recommendations.append(
            {
                "product_id": "PROD001",
                "reason": "Premium customer - high-value electronics",
                "confidence": 0.9,
                "estimated_value": 999.99,
            }
        )

    if "PROD" in preferred_categories:
        recommendations.append(
            {
                "product_id": "PROD002",
                "reason": "Matches customer preferences",
                "confidence": 0.7,
                "estimated_value": 89.99,
            }
        )

    # Add complementary products
    if product and product.category == "Electronics":
        recommendations.append(
            {
                "product_id": "PROD003",
                "reason": "Complementary to electronics purchase",
                "confidence": 0.6,
                "estimated_value": 129.99,
            }
        )

    return {
        "recommendations": recommendations,
        "total_recommendations": len(recommendations),
        "average_confidence": sum(r["confidence"] for r in recommendations)
        / max(1, len(recommendations)),
        "generation_timestamp": time.time(),
    }


async def assess_business_risk(input_data: Any, **kwargs) -> dict:
    """Assess business and operational risks."""
    await asyncio.sleep(0.2)  # Simulate risk analysis

    # Extract data from input_data dictionary
    customer_analysis = input_data.get("analyze_customer", {})
    product_data = input_data.get("load_product_data", {})
    recommendations = input_data.get("generate_recommendations", {})

    insights = customer_analysis.get("customer_insights", {})
    product = product_data.get("product_data")
    recs = recommendations.get("recommendations", [])

    risks = []
    risk_score = 0.0

    # Check stock levels
    if product and product.stock_level < 10:
        risks.append("Low stock level")
        risk_score += 0.3

    # Check customer value vs product value
    value_tier = insights.get("value_tier", "low")
    if value_tier == "high" and product and product.price < 100:
        risks.append("High-value customer, low-value product")
        risk_score += 0.2

    # Check recommendation confidence
    avg_confidence = recommendations.get("average_confidence", 0.0)
    if avg_confidence < 0.5:
        risks.append("Low confidence recommendations")
        risk_score += 0.2

    return {
        "risks": risks,
        "risk_score": min(1.0, risk_score),
        "risk_level": "high" if risk_score > 0.5 else "medium" if risk_score > 0.2 else "low",
        "assessment_timestamp": time.time(),
    }


async def generate_optimization_suggestions(input_data: Any, **kwargs) -> dict:
    """Generate optimization suggestions based on analysis."""
    await asyncio.sleep(0.15)  # Simulate optimization analysis

    # Extract data from input_data dictionary
    customer_analysis = input_data.get("analyze_customer", {})
    product_data = input_data.get("load_product_data", {})
    risk_assessment = input_data.get("assess_risk", {})

    insights = customer_analysis.get("customer_insights", {})
    product = product_data.get("product_data")
    risks = risk_assessment.get("risks", [])

    suggestions = []

    # Customer-specific optimizations
    value_tier = insights.get("value_tier", "low")
    if value_tier == "high":
        suggestions.append("Implement VIP customer program")
        suggestions.append("Offer premium support channels")

    # Product-specific optimizations
    if product and product.stock_level < 20:
        suggestions.append("Increase stock levels for popular products")
        suggestions.append("Implement just-in-time inventory system")

    # Risk-based optimizations
    if "Low stock level" in risks:
        suggestions.append("Set up automated stock alerts")
        suggestions.append("Establish backup suppliers")

    if "Low confidence recommendations" in risks:
        suggestions.append("Improve recommendation algorithm")
        suggestions.append("Add more customer data points")

    return {
        "suggestions": suggestions,
        "total_suggestions": len(suggestions),
        "priority": "high" if len(suggestions) > 3 else "medium",
        "optimization_timestamp": time.time(),
    }


async def compile_final_report(input_data: Any, **kwargs) -> dict:
    """Compile comprehensive final report."""
    await asyncio.sleep(0.1)  # Simulate report generation

    # Extract data from input_data dictionary
    customer_analysis = input_data.get("analyze_customer", {})
    recommendations = input_data.get("generate_recommendations", {})
    risk_assessment = input_data.get("assess_risk", {})
    optimizations = input_data.get("generate_optimizations", {})

    insights = customer_analysis.get("customer_insights", {})
    recs = recommendations.get("recommendations", [])
    risks = risk_assessment.get("risks", [])
    suggestions = optimizations.get("suggestions", [])

    return {
        "report_summary": {
            "customer_value": insights.get("value_tier", "unknown"),
            "total_recommendations": len(recs),
            "risk_level": risk_assessment.get("risk_level", "unknown"),
            "optimization_priority": optimizations.get("priority", "low"),
        },
        "detailed_analysis": {
            "customer_insights": insights,
            "recommendations": recs,
            "risk_assessment": risk_assessment,
            "optimization_suggestions": suggestions,
        },
        "executive_summary": {
            "key_findings": [
                f"Customer value tier: {insights.get('value_tier', 'unknown')}",
                f"Generated {len(recs)} product recommendations",
                f"Risk level: {risk_assessment.get('risk_level', 'unknown')}",
                f"Optimization priority: {optimizations.get('priority', 'low')}",
            ],
            "next_actions": suggestions[:3],  # Top 3 suggestions
        },
        "report_timestamp": time.time(),
        "report_version": "1.0",
    }


def create_complex_workflow() -> DirectedGraph:
    """Create a complex business analysis workflow."""

    graph = DirectedGraph()

    # Data loading layer (parallel)
    graph.add(NodeSpec("load_customer", load_customer_data))
    graph.add(NodeSpec("load_product", load_product_data))
    graph.add(NodeSpec("load_orders", load_order_history))

    # Analysis layer (depends on data loading)
    graph.add(
        NodeSpec("analyze_customer", analyze_customer_behavior).after(
            "load_customer", "load_orders"
        )
    )
    graph.add(
        NodeSpec("generate_recommendations", generate_product_recommendations).after(
            "analyze_customer", "load_product"
        )
    )
    graph.add(
        NodeSpec("assess_risk", assess_business_risk).after(
            "analyze_customer", "load_product", "generate_recommendations"
        )
    )
    graph.add(
        NodeSpec("generate_optimizations", generate_optimization_suggestions).after(
            "analyze_customer", "load_product", "assess_risk"
        )
    )

    # Final compilation
    graph.add(
        NodeSpec("compile_report", compile_final_report).after(
            "analyze_customer", "generate_recommendations", "assess_risk", "generate_optimizations"
        )
    )

    return graph


async def demonstrate_complex_workflow():
    """Demonstrate the complex workflow execution."""

    print("\n🚀 Complex Workflow Execution")
    print("=" * 40)

    # Create workflow
    graph = create_complex_workflow()

    print("\n📊 Workflow Analysis:")
    waves = graph.waves()
    print(f"   • Total waves: {len(waves)}")
    for i, wave in enumerate(waves, 1):
        if len(wave) == 1:
            print(f"   • Wave {i}: {wave[0]} (sequential)")
        else:
            print(f"   • Wave {i}: {', '.join(wave)} (parallel)")

    # Validate
    graph.validate()
    print("   ✅ Workflow validation passed")

    # Execute with different scenarios
    test_scenarios = [
        {"customer_id": "CUST001", "product_id": "PROD001"},  # Premium customer, high-value product
        {"customer_id": "CUST002", "product_id": "PROD002"},  # Standard customer, moderate product
        {"customer_id": "CUST003", "product_id": "PROD003"},  # Premium customer, sports product
    ]

    orchestrator = Orchestrator(validator=coerce_validator())

    for i, scenario in enumerate(test_scenarios, 1):
        print(
            f"\n🧪 Scenario {i}: Customer {scenario['customer_id']} → Product {scenario['product_id']}"
        )

        start_time = time.time()
        results = await orchestrator.run(graph, scenario)
        end_time = time.time()

        execution_time = end_time - start_time

        # Extract key results
        final_report = results.get("compile_report", {}).get("final_report", {})
        executive_summary = final_report.get("executive_summary", {})

        print(f"   ⏱️  Execution time: {execution_time:.3f}s")
        print(f"   💰 Customer value: {executive_summary.get('customer_value', 'unknown')}")
        print(f"   📊 Recommendations: {executive_summary.get('recommendation_count', 0)}")
        print(f"   ⚠️  Risk level: {executive_summary.get('risk_level', 'unknown')}")
        print(
            f"   🎯 Optimization priority: {executive_summary.get('optimization_priority', 'medium')}"
        )

        # Show business impact
        business_impact = final_report.get("business_impact", {})
        print(
            f"   💵 Revenue potential: ${business_impact.get('estimated_revenue_potential', 0):.0f}"
        )
        print(f"   🛡️  Risk factors: {business_impact.get('risk_mitigation_value', 0)}")
        print(
            f"   🔧 Optimization opportunities: {business_impact.get('optimization_opportunities', 0)}"
        )

    return results


async def demonstrate_error_handling():
    """Demonstrate error handling in complex workflows."""

    print("\n🛡️ Error Handling Demo")
    print("=" * 40)

    # Create a workflow that might fail
    async def failing_node(input_data: dict) -> dict:
        """A node that might fail."""
        if input_data.get("should_fail"):
            raise ValueError("Simulated failure for testing")
        return {"status": "success"}

    async def recovery_node(input_data: dict) -> dict:
        """A node that handles failures."""
        if "error" in str(input_data):
            return {"status": "recovered", "fallback_data": "default_value"}
        return {"status": "normal", "data": input_data}

    graph = DirectedGraph()
    graph.add(NodeSpec("failing", failing_node))
    graph.add(NodeSpec("recovery", recovery_node).after("failing"))

    orchestrator = Orchestrator(validator=coerce_validator())

    # Test normal execution
    print("\n   🟢 Normal execution:")
    try:
        results = await orchestrator.run(graph, {"should_fail": False})
        print("   ✅ Normal execution succeeded")
    except Exception as e:
        print(f"   ❌ Unexpected failure: {e}")

    # Test failure scenario
    print("\n   🔴 Failure scenario:")
    try:
        results = await orchestrator.run(graph, {"should_fail": True})
        print("   ⚠️  Unexpected success")
    except Exception as e:
        print(f"   ✅ Correctly caught error: {type(e).__name__}")
        print(f"   📝 Error message: {str(e)[:50]}...")


async def demonstrate_performance_optimization():
    """Demonstrate performance optimization techniques."""

    print("\n⚡ Performance Optimization Demo")
    print("=" * 40)

    # Create a performance-intensive workflow
    async def slow_operation(input_data: dict) -> dict:
        """Simulate a slow operation."""
        await asyncio.sleep(0.5)  # Simulate heavy computation
        return {"result": "computed", "input": input_data}

    async def fast_operation(input_data: dict) -> dict:
        """Simulate a fast operation."""
        await asyncio.sleep(0.05)  # Simulate quick operation
        return {"result": "quick", "input": input_data}

    async def parallel_processor(input_data: Any, **kwargs) -> dict:
        """Process data in parallel."""
        await asyncio.sleep(0.1)

        # Extract data from input_data dictionary
        fast_op1_result = input_data.get("fast_op1", {})
        fast_op2_result = input_data.get("fast_op2", {})

        return {
            "combined_result": f"{fast_op1_result.get('result')} + {fast_op2_result.get('result')}",
            "processing_time": time.time(),
        }

    # Create optimized workflow
    graph = DirectedGraph()
    graph.add(NodeSpec("slow_op", slow_operation))
    graph.add(NodeSpec("fast_op1", fast_operation).after("slow_op"))
    graph.add(NodeSpec("fast_op2", fast_operation).after("slow_op"))
    graph.add(NodeSpec("parallel_processor", parallel_processor).after("fast_op1", "fast_op2"))

    print("\n📊 Performance Analysis:")
    waves = graph.waves()
    print(f"   • Total waves: {len(waves)}")
    print(f"   • Parallel opportunities: {sum(1 for wave in waves if len(wave) > 1)}")

    # Execute and measure
    orchestrator = Orchestrator(validator=coerce_validator())

    start_time = time.time()
    results = await orchestrator.run(graph, {"test": "data"})
    end_time = time.time()

    total_time = end_time - start_time
    print(f"\n⏱️  Performance Results:")
    print(f"   • Total execution time: {total_time:.3f}s")
    print(f"   • Sequential time (estimated): 0.7s")
    print(f"   • Parallel time (actual): {total_time:.3f}s")
    print(f"   • Speedup: {0.7/total_time:.2f}x")

    return results


async def main():
    """Demonstrate complex workflow patterns."""

    print("🚀 Example 19: Complex Real-World Workflow")
    print("=" * 55)

    print("\n🎯 This example demonstrates:")
    print("   • Complex multi-stage business workflows")
    print("   • Error handling and recovery patterns")
    print("   • Performance optimization techniques")
    print("   • Real-world data processing patterns")
    print("   • Enterprise-level pipeline design")

    results = await demonstrate_complex_workflow()
    await demonstrate_error_handling()
    await demonstrate_performance_optimization()

    print("\n🎯 Key Concepts Learned:")
    print("   ✅ Complex Workflows - Multi-stage business processes")
    print("   ✅ Error Handling - Graceful failure and recovery")
    print("   ✅ Performance Optimization - Parallel execution strategies")
    print("   ✅ Data Processing - Real-world data transformation patterns")
    print("   ✅ Enterprise Patterns - Scalable and maintainable designs")

    print("\n💡 Best Practices:")
    print("   • Design workflows with clear separation of concerns")
    print("   • Implement comprehensive error handling")
    print("   • Optimize for parallel execution where possible")
    print("   • Use type-safe data models")
    print("   • Monitor performance and optimize bottlenecks")

    print("\n🔗 Next: Run example 20 to learn about performance optimization!")


if __name__ == "__main__":
    asyncio.run(main())
