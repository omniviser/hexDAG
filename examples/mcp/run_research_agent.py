"""Unified runner for deep research agent across all environments.

Supports dev, staging, and production environments with a single script.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hexdag.builtin.adapters.mock import MockLLM, MockToolRouter
from hexdag.core.logging import get_logger
from hexdag.core.orchestration.orchestrator import Orchestrator
from hexdag.core.pipeline_builder.yaml_builder import YamlPipelineBuilder

logger = get_logger(__name__)


async def run_research_agent(
    question: str,
    environment: str = "dev",
    pipeline_file: str | None = None,
) -> dict:
    """Run deep research agent in specified environment.

    Args
    ----
        question: Research question to investigate
        environment: Environment to run in (dev/staging/prod)
        pipeline_file: Optional custom pipeline file path

    Returns
    -------
        Research results dict
    """
    print("\n" + "=" * 80)
    print(f"🔬 Deep Research Agent - {environment.upper()} Environment")
    print("=" * 80)

    # Determine pipeline file
    if pipeline_file is None:
        pipeline_file = f"deep_research_agent_{environment}.yaml"

    yaml_path = str(Path(__file__).parent / pipeline_file)

    # Validate file exists
    if not Path(yaml_path).exists():
        raise FileNotFoundError(
            f"Pipeline file not found: {yaml_path}\n"
            f"Available environments: dev, staging, prod\n"
            f"Run: python examples/mcp/generate_research_agent_environments.py"
        )

    print(f"\n📄 Loading pipeline: {pipeline_file}")
    print(f"   Environment: {environment}")

    # Import Tavily tools if needed (for prod/staging)
    if environment in ["prod", "staging"]:
        try:
            # Import to register tools
            import examples.mcp.tavily_adapter  # noqa: F401

            print("   ✅ Tavily tools registered")
        except ImportError:
            logger.warning("Tavily adapter not found, tools may not be available")

    # Load pipeline
    builder = YamlPipelineBuilder()
    graph, config = builder.build_from_yaml_file(yaml_path)

    print(f"   ✅ Pipeline loaded: {len(graph.nodes)} nodes")

    # Configure orchestrator based on environment
    if environment == "dev":
        print("\n🔧 Dev Configuration:")
        print("   LLM: Mock adapter (no API key)")
        print("   Tools: Mock router (no API key)")
        print("   Cost: $0.00")

        # Create mock adapters
        mock_llm = MockLLM(
            responses=[
                "I'll search for information about this topic. INVOKE_TOOL: search(query='research query')",
                "Let me analyze this further. INVOKE_TOOL: search(query='detailed analysis')",
                "Let me search for more context. INVOKE_TOOL: search(query='additional context')",
                "Based on my comprehensive research across multiple sources, here are the key findings: "
                "[Mock comprehensive research results with multiple insights and data points]",
            ],
            delay_seconds=0.05,
        )
        mock_tool_router = MockToolRouter(available_tools=["search", "calculate"])

        orchestrator = Orchestrator(ports={"llm": mock_llm, "tool_router": mock_tool_router})

    elif environment == "staging":
        print("\n🧪 Staging Configuration:")
        print("   LLM: OpenAI GPT-4o-mini (requires STAGING_OPENAI_API_KEY)")
        print("   Tools: Mock router (no real API calls)")
        print("   Cost: ~$0.01-0.05 per run")

        # Validate API key
        if not os.getenv("STAGING_OPENAI_API_KEY"):
            raise EnvironmentError(
                "STAGING_OPENAI_API_KEY not set\n"
                "Set with: export STAGING_OPENAI_API_KEY=sk-..."
            )

        # Use mock tools for staging
        mock_tool_router = MockToolRouter(
            available_tools=["research:tavily_search", "research:tavily_qna_search"]
        )

        orchestrator = Orchestrator(ports={"tool_router": mock_tool_router})

    elif environment == "prod":
        print("\n🚀 Production Configuration:")
        print("   LLM: OpenAI GPT-4o (requires OPENAI_API_KEY)")
        print("   Tools: Real Tavily search (requires TAVILY_API_KEY)")
        print("   Cost: ~$0.10-0.30 per run")

        # Validate API keys
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY not set\nSet with: export OPENAI_API_KEY=sk-...")

        if not os.getenv("TAVILY_API_KEY"):
            raise EnvironmentError(
                "TAVILY_API_KEY not set\nSet with: export TAVILY_API_KEY=tvly-..."
            )

        # Use real adapters (configured in YAML)
        orchestrator = Orchestrator()

    else:
        raise ValueError(f"Unknown environment: {environment}. Use: dev, staging, or prod")

    # Run the agent
    print(f"\n{'='*80}")
    print(f"Research Question: {question}")
    print(f"{'='*80}\n")

    print("🚀 Running agent...")
    print("-" * 80)

    results = await orchestrator.run(
        graph,
        initial_input={"research_question": question},
    )

    print("\n" + "=" * 80)
    print("📊 Research Results")
    print("=" * 80)

    # Show final answer
    final_key = "research_agent_final"
    if final_key in results:
        final_result = results[final_key]
        final_text = (
            final_result.get("text", final_result) if isinstance(final_result, dict) else final_result
        )

        print("\n✅ Final Answer:")
        print("-" * 80)
        print(final_text)
        print("-" * 80)
    else:
        print("\n⚠️  Final result not found in expected key")
        print("Available keys:", list(results.keys())[:5])

    # Show stats
    if environment == "dev":
        print(f"\n📈 Mock Statistics:")
        if hasattr(orchestrator.ports.get("llm"), "call_count"):
            print(f"   LLM calls: {orchestrator.ports['llm'].call_count}")
        if hasattr(orchestrator.ports.get("tool_router"), "call_history"):
            print(f"   Tool calls: {len(orchestrator.ports['tool_router'].call_history)}")

    print("\n" + "=" * 80)
    print(f"✅ {environment.upper()} agent completed successfully!")
    print("=" * 80)

    return results


async def interactive_mode(environment: str = "dev"):
    """Interactive mode for continuous research."""
    print("\n" + "=" * 80)
    print(f"🔬 Deep Research Agent - Interactive {environment.upper()} Mode")
    print("=" * 80)

    while True:
        try:
            question = input("\nResearch Question (or 'quit'): ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break

            if not question:
                continue

            await run_research_agent(question, environment)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print(f"\n❌ Error: {e}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Deep Research Agent Runner")
    parser.add_argument(
        "--env",
        "--environment",
        choices=["dev", "staging", "prod"],
        default="dev",
        help="Environment to run in (default: dev)",
    )
    parser.add_argument(
        "--pipeline", help="Custom pipeline file path (optional)", default=None
    )
    parser.add_argument(
        "--question",
        "-q",
        help="Research question (if not provided, enters interactive mode)",
        default=None,
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Run in interactive mode"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🔬 Deep Research Agent")
    print("=" * 80)
    print(f"\nEnvironment: {args.env.upper()}")
    print(f"Pipeline: {args.pipeline or f'deep_research_agent_{args.env}.yaml'}")

    try:
        if args.interactive or args.question is None:
            # Interactive mode
            asyncio.run(interactive_mode(args.env))
        else:
            # Single question mode
            asyncio.run(run_research_agent(args.question, args.env, args.pipeline))

    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return 1
    except EnvironmentError as e:
        print(f"\n❌ {e}")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
