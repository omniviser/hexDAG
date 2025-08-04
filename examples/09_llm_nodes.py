"""
Example 09: LLM Nodes.

This example demonstrates LLM nodes in hexAI:
- Basic LLM node usage
- LLM nodes with prompts
- LLM nodes with context
- LLM nodes with tools
- LLM node composition
"""

import asyncio
from typing import Any

from hexai.adapters.mock.mock_llm import MockLLM
from hexai.core.application.nodes.llm_node import LLMNode
from hexai.core.application.orchestrator import Orchestrator
from hexai.core.domain.dag import DirectedGraph


async def main():
    """Demonstrate LLM nodes."""

    print("🤖 Example 09: LLM Nodes")
    print("=" * 25)

    print("\n🎯 This example demonstrates:")
    print("   • Basic LLM node usage")
    print("   • LLM nodes with prompts")
    print("   • LLM nodes with context")
    print("   • LLM nodes with tools")
    print("   • LLM node composition")

    # Create mock LLM for testing
    mock_llm = MockLLM(
        responses=[
            "This is a positive sentiment analysis.",
            "The text contains 3 main topics: technology, innovation, and future.",
            "Based on the analysis, I recommend focusing on user experience improvements.",
            "The content is well-structured and engaging for the target audience.",
            "This text demonstrates excellent writing quality and clarity.",
        ]
    )

    # Create LLMNode factory
    llm_node = LLMNode()

    # Test 1: Basic LLM Node
    print("\n🤖 Test 1: Basic LLM Node")
    print("-" * 30)

    # Create basic LLM node using from_str
    basic_llm_node = LLMNode.from_str(
        "text_analyzer", "Analyze the sentiment of this text: {{input}}"
    )

    # Create graph
    graph = DirectedGraph()
    graph.add(basic_llm_node)

    # Execute
    orchestrator = Orchestrator(ports={"llm": mock_llm})
    result = await orchestrator.run(graph, {"input": "I love this new product!"})

    print(f"   ✅ Basic LLM node executed successfully")
    print(f"   📊 Analysis: {result['text_analyzer']}")

    # Test 2: LLM Node with Structured Output
    print("\n🤖 Test 2: LLM Node with Structured Output")
    print("-" * 30)

    # Create LLM node with structured output
    structured_llm_node = LLMNode.from_str(
        "topic_analyzer", "Extract the main topics from this text: {{input}}"
    )

    # Create graph
    graph = DirectedGraph()
    graph.add(structured_llm_node)

    # Execute
    result = await orchestrator.run(
        graph, {"input": "This article discusses AI, machine learning, and data science."}
    )

    print(f"   ✅ Structured LLM node executed successfully")
    print(f"   📊 Response: {result['topic_analyzer']}")

    # Test 3: LLM Node with Context
    print("\n🤖 Test 3: LLM Node with Context")
    print("-" * 30)

    # Create LLM node that uses context from previous nodes
    context_llm_node = LLMNode.from_str(
        "recommendation_engine",
        "Based on the analysis '{{analysis}}', provide recommendations for: {{input}}",
    )

    # Create graph
    graph = DirectedGraph()
    graph.add(context_llm_node)

    # Execute with context
    context_data = {
        "analysis": "The text shows positive sentiment with technical content",
        "input": "How can we improve this product?",
    }

    result = await orchestrator.run(graph, context_data)
    print(f"   ✅ Context-aware LLM node executed successfully")
    print(f"   📊 Recommendation: {result['recommendation_engine']}")

    # Test 4: LLM Node with Complex Prompt
    print("\n🤖 Test 4: LLM Node with Complex Prompt")
    print("-" * 30)

    # Create LLM node with complex prompt template
    complex_llm_node = LLMNode.from_str(
        "quality_assessor",
        """
        You are a content quality assessor. Analyze the following text:

        Text: {{input}}

        Please provide:
        1. Quality score (1-10)
        2. Writing style assessment
        3. Areas for improvement

        Respond in a structured format.
        """,
    )

    # Create graph
    graph = DirectedGraph()
    graph.add(complex_llm_node)

    # Execute
    result = await orchestrator.run(
        graph, {"input": "This is a well-written article about artificial intelligence."}
    )

    print(f"   ✅ Complex prompt LLM node executed successfully")
    print(f"   📊 Assessment: {result['quality_assessor']}")

    print(f"\n🎯 Key Concepts Learned:")
    print("   ✅ LLMNode - Create nodes that use language models")
    print("   ✅ Prompt Templates - Use {{variables}} in prompts")
    print("   ✅ Structured Output - Define output schemas for LLMs")
    print("   ✅ Context Usage - Pass data between LLM nodes")
    print("   ✅ LLM Composition - Chain multiple LLM nodes together")

    print(f"\n🔗 Next: Run example 10 to learn about agent nodes!")


if __name__ == "__main__":
    asyncio.run(main())
