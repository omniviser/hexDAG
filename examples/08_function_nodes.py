"""
Example 08: Function Nodes.

This example demonstrates function nodes in hexAI:
- Simple function nodes
- Function nodes with validation
- Function nodes with dependencies
- Function nodes with ports
- Function composition patterns
"""

import asyncio
import json

from hexai.app.application.nodes.function_node import FunctionNode
from hexai.app.application.orchestrator import Orchestrator
from hexai.app.domain.dag import DirectedGraph


# Define the functions that will be wrapped by FunctionNode
async def text_processor(input_data: str, **kwargs) -> dict:
    """Process text data."""
    return {
        "original": input_data,
        "length": len(input_data),
        "word_count": len(input_data.split()),
        "uppercase": input_data.upper(),
        "lowercase": input_data.lower(),
    }


async def number_calculator(input_data: dict, **kwargs) -> dict:
    """Calculate statistics from text processing."""
    return {
        "text_stats": {
            "characters": input_data["length"],
            "words": input_data["word_count"],
            "average_word_length": input_data["length"] / max(input_data["word_count"], 1),
        },
        "processing_complete": True,
    }


async def data_formatter(input_data: dict, **kwargs) -> dict:
    """Format data for output."""
    formatter = kwargs.get("formatter", "json")

    if formatter == "json":
        return {
            "formatted_data": json.dumps(input_data, indent=2),
            "format": "json",
            "size": len(json.dumps(input_data)),
        }
    elif formatter == "csv":
        # Simple CSV formatting
        lines = []
        for key, value in input_data.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    lines.append(f"{key}.{sub_key},{sub_value}")
            else:
                lines.append(f"{key},{value}")

        csv_data = "\n".join(lines)
        return {"formatted_data": csv_data, "format": "csv", "size": len(csv_data)}
    else:
        return {"formatted_data": str(input_data), "format": "text", "size": len(str(input_data))}


async def data_validator(input_data: dict, **kwargs) -> dict:
    """Validate processed data."""
    errors = []

    # Check required fields
    if "text_stats" not in input_data:
        errors.append("Missing text_stats")

    if "processing_complete" not in input_data:
        errors.append("Missing processing_complete")

    # Validate text stats
    if "text_stats" in input_data:
        stats = input_data["text_stats"]
        if not isinstance(stats.get("characters"), int):
            errors.append("Invalid characters count")
        if not isinstance(stats.get("words"), int):
            errors.append("Invalid word count")

    if errors:
        return {"valid": False, "errors": errors, "status": "validation_failed"}
    else:
        return {"valid": True, "status": "validation_passed", "data": input_data}


async def report_generator(input_data: dict, **kwargs) -> dict:
    """Generate a comprehensive report."""
    logger = kwargs.get("logger")

    if logger:
        await logger.log("INFO", "Generating report...")

    return {
        "report": {
            "summary": f"Processed {input_data.get('word_count', 0)} words",
            "validation_status": input_data.get("status", "unknown"),
            "formatted_output": input_data.get("formatted_data", ""),
            "generated_at": "2024-01-01T00:00:00Z",
        },
        "report_complete": True,
    }


class MockLogger:
    """Mock logger for demonstration."""

    def __init__(self):
        self.logs = []

    async def log(self, level: str, message: str) -> None:
        """Log a message."""
        self.logs.append({"level": level, "message": message})


async def main():
    """Demonstrate function nodes."""

    print("🔧 Example 08: Function Nodes")
    print("=" * 30)

    print("\n🎯 This example demonstrates:")
    print("   • Simple function nodes")
    print("   • Function nodes with validation")
    print("   • Function nodes with dependencies")
    print("   • Function nodes with ports")
    print("   • Function composition patterns")

    # Create FunctionNode factory
    function_node = FunctionNode()

    # Test 1: Simple Function Node
    print("\n🔧 Test 1: Simple Function Node")
    print("-" * 35)

    # Create simple function node
    text_processor_node = function_node(name="text_processor", fn=text_processor)

    # Create graph
    graph = DirectedGraph()
    graph.add(text_processor_node)

    # Execute
    orchestrator = Orchestrator()
    result = await orchestrator.run(graph, "Hello World Function Node!")

    print(f"   ✅ Function node executed successfully")
    print(f"   📊 Original text: {result['text_processor']['original']}")
    print(f"   📊 Word count: {result['text_processor']['word_count']}")
    print(f"   📊 Uppercase: {result['text_processor']['uppercase']}")

    # Test 2: Function Node with Validation
    print("\n🔧 Test 2: Function Node with Validation")
    print("-" * 35)

    # Create function node with validation
    validator_node = function_node(name="data_validator", fn=data_validator)

    # Create graph
    graph = DirectedGraph()
    graph.add(validator_node)

    # Execute with valid data
    valid_data = {"text_stats": {"characters": 10, "words": 2}, "processing_complete": True}

    result = await orchestrator.run(graph, valid_data)
    print(f"   ✅ Validation node executed successfully")
    print(f"   📊 Validation status: {result['data_validator']['status']}")

    # Test 3: Function Node with Dependencies
    print("\n🔧 Test 3: Function Node with Dependencies")
    print("-" * 35)

    # Create function nodes with dependencies
    text_node = function_node(name="text_processor", fn=text_processor)

    calculator_node = function_node(name="number_calculator", fn=number_calculator).after(
        "text_processor"
    )

    # Create graph
    graph = DirectedGraph()
    graph.add(text_node)
    graph.add(calculator_node)

    # Execute
    result = await orchestrator.run(graph, "Dependency test data")
    print(f"   ✅ Dependency chain executed successfully")
    print(f"   📊 Characters: {result['number_calculator']['text_stats']['characters']}")
    print(f"   📊 Words: {result['number_calculator']['text_stats']['words']}")

    # Test 4: Function Node with Ports
    print("\n🔧 Test 4: Function Node with Ports")
    print("-" * 35)

    # Create function node that uses ports
    formatter_node = function_node(name="data_formatter", fn=data_formatter)

    # Create graph
    graph = DirectedGraph()
    graph.add(formatter_node)

    # Execute with ports
    test_data = {"text_stats": {"characters": 15, "words": 3}}

    orchestrator_with_ports = Orchestrator(ports={"formatter": "json"})
    result = await orchestrator_with_ports.run(graph, test_data)

    print(f"   ✅ Port-based function executed successfully")
    print(f"   📊 Format: {result['data_formatter']['format']}")
    print(f"   📊 Size: {result['data_formatter']['size']} bytes")

    # Test 5: Function Composition
    print("\n🔧 Test 5: Function Composition")
    print("-" * 35)

    # Create multiple function nodes
    text_comp_node = function_node(name="text_processor", fn=text_processor)

    calculator_comp_node = function_node(name="number_calculator", fn=number_calculator).after(
        "text_processor"
    )

    validator_comp_node = function_node(name="data_validator", fn=data_validator).after(
        "number_calculator"
    )

    formatter_comp_node = function_node(name="data_formatter", fn=data_formatter).after(
        "data_validator"
    )

    # Create graph
    graph = DirectedGraph()
    graph.add(text_comp_node)
    graph.add(calculator_comp_node)
    graph.add(validator_comp_node)
    graph.add(formatter_comp_node)

    # Execute composition
    result = await orchestrator.run(graph, "Composition test data")
    print(f"   ✅ Function composition executed successfully")
    print(f"   📊 Validation status: {result['data_validator']['status']}")
    print(f"   📊 Format: {result['data_formatter']['format']}")

    print(f"\n🎯 Key Concepts Learned:")
    print("   ✅ FunctionNode - Create nodes from Python functions")
    print("   ✅ Input/Output Schemas - Validate data with Pydantic")
    print("   ✅ Dependencies - Chain function nodes together")
    print("   ✅ Ports - Access external services in functions")
    print("   ✅ Composition - Build complex pipelines from simple functions")

    print(f"\n🔗 Next: Run example 09 to learn about LLM nodes!")


if __name__ == "__main__":
    asyncio.run(main())
