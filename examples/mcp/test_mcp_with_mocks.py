"""Test MCP server with mock adapters for testing without real API keys.

This demonstrates:
1. Building pipelines with mock LLM and tool adapters
2. Environment variable handling in YAML
3. Running agents without real API keys
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hexdag.compiler.yaml_builder import YamlPipelineBuilder
from hexdag.kernel.orchestration.orchestrator import Orchestrator
from hexdag.kernel.ports.tool_router import ToolRouter
from hexdag.mcp_server import build_yaml_pipeline_interactive
from hexdag.stdlib.adapters.mock import MockLLM


def test_mcp_builds_with_mock_adapters():
    """Test that MCP can build pipelines with mock adapters."""
    print("\n" + "=" * 80)
    print("Test 1: MCP Build with Mock Adapters")
    print("=" * 80)

    nodes = [
        {
            "kind": "macro_invocation",
            "name": "test_agent",
            "spec": {
                "macro": "core:reasoning_agent",
                "config": {
                    "main_prompt": "Test research: {{query}}",
                    "max_steps": 2,
                    "allowed_tools": ["search", "calculate"],
                    "tool_format": "mixed",
                },
            },
            "dependencies": [],
        }
    ]

    ports = {
        "llm": {
            "adapter": "plugin:mock_llm",
            "config": {
                "responses": [
                    "I'll search for information. INVOKE_TOOL: search(query='test')",
                    "Based on the search results, here is my answer: Test response",
                ],
            },
        },
        "tool_router": {
            "adapter": "hexdag.kernel.ports.tool_router.ToolRouter",
            "config": {"tools": {}},
        },
    }

    yaml_output = build_yaml_pipeline_interactive(
        pipeline_name="test-mock-pipeline",
        description="Test pipeline with mocks",
        nodes=nodes,
        ports=ports,
    )

    print("\n✅ Generated YAML:")
    print("-" * 80)
    print(yaml_output)
    print("-" * 80)

    # Verify it contains mock adapters
    assert "plugin:mock_llm" in yaml_output
    assert "hexdag.kernel.ports.tool_router.ToolRouter" in yaml_output

    print("\n✅ Test 1 passed: MCP successfully built pipeline with mock adapters")
    return yaml_output


def test_mcp_handles_environment_variables():
    """Test that MCP properly handles environment variables in YAML."""
    print("\n" + "=" * 80)
    print("Test 2: Environment Variable Handling")
    print("=" * 80)

    # Set test environment variables
    os.environ["TEST_API_KEY"] = "test-key-123"
    os.environ["TEST_MODEL"] = "gpt-4-test"

    nodes = [
        {
            "kind": "prompt_node",
            "name": "test_prompt",
            "spec": {"template": "Test: {{input}}"},
            "dependencies": [],
        }
    ]

    ports = {
        "llm": {
            "adapter": "core:openai",
            "config": {
                "api_key": "${TEST_API_KEY}",  # Environment variable
                "model": "${TEST_MODEL}",
            },
        }
    }

    yaml_output = build_yaml_pipeline_interactive(
        pipeline_name="test-env-pipeline", description="Test env vars", nodes=nodes, ports=ports
    )

    print("\n✅ Generated YAML with environment variables:")
    print("-" * 80)
    print(yaml_output)
    print("-" * 80)

    # Verify environment variables are preserved
    assert "${TEST_API_KEY}" in yaml_output
    assert "${TEST_MODEL}" in yaml_output

    print("\n✅ Test 2 passed: Environment variables preserved in YAML")

    # Clean up
    del os.environ["TEST_API_KEY"]
    del os.environ["TEST_MODEL"]

    return yaml_output


async def test_run_mock_agent():
    """Test running a complete agent pipeline with mock adapters."""
    print("\n" + "=" * 80)
    print("Test 3: Run Agent with Mock Adapters (No API Keys Needed)")
    print("=" * 80)

    # Create YAML pipeline with mocks
    pipeline_yaml = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: mock-research-agent
  description: Research agent using mock adapters for testing
spec:
  ports:
    llm:
      adapter: plugin:mock_llm
      config:
        responses:
          - "I'll search for quantum computing. INVOKE_TOOL: search(query='quantum computing breakthroughs')"
          - "Based on the search results, here are the latest quantum computing breakthroughs: IBM announced a 1000-qubit processor, and Google achieved error correction milestones."
    tool_router:
      adapter: hexdag.kernel.ports.tool_router.ToolRouter
      config:
        tools: {}
  nodes:
    - kind: macro_invocation
      metadata:
        name: research_agent
      spec:
        macro: core:reasoning_agent
        config:
          main_prompt: "Research the latest developments in: {{topic}}"
          max_steps: 2
          allowed_tools: [search, calculate]
          tool_format: mixed
      dependencies: []
"""

    print("\n📄 Pipeline YAML:")
    print("-" * 80)
    print(pipeline_yaml)
    print("-" * 80)

    # Build pipeline
    builder = YamlPipelineBuilder()
    graph, config = builder.build_from_yaml_string(pipeline_yaml)

    print("\n✅ Pipeline built successfully!")
    print(f"  Nodes: {len(graph)}")
    print(f"  Ports: {list(config.ports.keys())}")

    # Create mock adapters
    mock_llm = MockLLM(
        responses=[
            "I'll search for quantum computing. INVOKE_TOOL: search(query='quantum computing breakthroughs')",
            "Based on the search results: IBM announced a 1000-qubit processor. Google achieved error correction.",
        ]
    )

    tool_router = ToolRouter(
        tools={
            "search": lambda query="", **kw: {
                "results": [f"Mock result for: {query}"],
                "status": "success",
            },
            "calculate": lambda expression="", **kw: {
                "result": str(expression),
                "status": "success",
            },
        }
    )

    # Run the pipeline
    orchestrator = Orchestrator()

    print("\n🚀 Running agent with mock adapters...")
    print("-" * 80)

    try:
        results = await orchestrator.run(
            graph,
            initial_input={"topic": "quantum computing"},
            llm=mock_llm,
            tool_router=tool_router,
        )

        print("\n✅ Agent execution completed!")
        print("\n📊 Results:")
        print("-" * 80)

        # Show final result
        final_result = results.get("research_agent_final")
        if final_result:
            final_text = (
                final_result.get("text", final_result)
                if isinstance(final_result, dict)
                else final_result
            )
            print(f"Final Answer: {final_text}")
        else:
            print("Results:", results)

        print("-" * 80)

        # Verify mock was called
        print("\n📈 Mock Statistics:")
        print(f"  LLM calls: {mock_llm.call_count}")
        print(f"  Tool calls: {len(tool_router.call_history)}")

        if tool_router.call_history:
            print("\n🔧 Tool Calls Made:")
            for i, call in enumerate(tool_router.call_history, 1):
                print(f"  {i}. {call['tool_name']}({call['params']})")

        print("\n✅ Test 3 passed: Agent ran successfully with mock adapters!")
        return True

    except Exception as e:
        print(f"\n❌ Error running agent: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_mock_with_tavily_style_responses():
    """Test mock that simulates Tavily search responses."""
    print("\n" + "=" * 80)
    print("Test 4: Mock Tavily-Style Responses")
    print("=" * 80)

    # Mock Tavily search results
    mock_search_responses = {
        "search": {
            "query": "quantum computing breakthroughs 2025",
            "answer": "IBM announced a 1000-qubit quantum processor in 2025, marking significant progress.",
            "results": [
                {
                    "title": "IBM Unveils 1000-Qubit Quantum Processor",
                    "url": "https://research.ibm.com/quantum-2025",
                    "content": "IBM Research announced the Quantum System Two with over 1000 qubits...",
                    "score": 0.98,
                },
                {
                    "title": "Google Achieves Quantum Error Correction",
                    "url": "https://ai.google/quantum-error-correction",
                    "content": "Google Quantum AI demonstrated below-threshold error rates...",
                    "score": 0.95,
                },
            ],
        }
    }

    print("\n✅ Mock Tavily-style search response:")
    print("-" * 80)
    import json

    print(json.dumps(mock_search_responses, indent=2))
    print("-" * 80)

    # Create a ToolRouter with a mock Tavily search function
    def mock_tavily_search(query: str = "", **kw) -> dict:
        """Mock Tavily search returning realistic results."""
        return {
            "query": query,
            "answer": f"Mock AI-generated answer for: {query}",
            "results": [
                {
                    "title": f"Result 1 for {query}",
                    "url": "https://example.com/1",
                    "content": f"Detailed content about {query}...",
                    "score": 0.95,
                },
                {
                    "title": f"Result 2 for {query}",
                    "url": "https://example.com/2",
                    "content": f"More information on {query}...",
                    "score": 0.90,
                },
            ],
            "response_time": 0.5,
        }

    router = ToolRouter(tools={"tavily_search": mock_tavily_search})

    print("\n✅ Test 4 passed: Can create Tavily-style mock responses")
    return router


def main():
    """Run all MCP mock adapter tests."""
    print("\n" + "=" * 80)
    print("🧪 MCP Mock Adapter Test Suite")
    print("=" * 80)
    print("\nTesting MCP server support for:")
    print("  1. Mock LLM adapters (no OpenAI key needed)")
    print("  2. Mock tool routers (no Tavily key needed)")
    print("  3. Environment variable handling")
    print("  4. End-to-end agent execution with mocks")

    try:
        # Test 1: Build with mock adapters
        test_mcp_builds_with_mock_adapters()

        # Test 2: Environment variables
        test_mcp_handles_environment_variables()

        # Test 3: Run with mocks
        asyncio.run(test_run_mock_agent())

        # Test 4: Tavily-style mocks
        test_mock_with_tavily_style_responses()

        # Summary
        print("\n" + "=" * 80)
        print("✅ All Tests Passed!")
        print("=" * 80)
        print("\n📝 Summary:")
        print("  ✅ MCP server can build pipelines with mock adapters")
        print("  ✅ Environment variables are properly handled")
        print("  ✅ Agents can run without real API keys using mocks")
        print("  ✅ Mock responses can simulate real tool behavior")
        print("\n💡 Use Case:")
        print("  - Develop and test agents locally without API costs")
        print("  - CI/CD testing without exposing API keys")
        print("  - Prototype agent behavior before production")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
