#!/usr/bin/env python
"""
hexDAG Research and Report Generator Demo
Ported from LangGraph project-11-research-and-report-generation

Pattern: Multi-Agent Workflow
- Researcher agent gathers information on a topic
- Writer agent creates a blog post from the research

Run with: ..\..\.venv\Scripts\python.exe run_research.py
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables
env_path = project_root / "reference_examples" / "langgraph-tutorials" / ".env"
load_dotenv(env_path)

import google.generativeai as genai
from hexdag.core.domain.dag import DirectedGraph, NodeSpec
from hexdag.core.orchestration.orchestrator import Orchestrator

# Configure Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env file")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)


async def researcher(inputs: dict) -> dict:
    """
    Research agent - gathers information on a topic.

    LangGraph version:
        def researcher(state):
            response = llm.invoke("Research the latest trends in AI.")
            return {"messages": [response]}
    """
    topic = inputs.get("topic", "the latest trends in AI")

    print(f"  [RESEARCHER] Researching: {topic}")
    print(f"  [RESEARCHER] Gathering facts, trends, and insights...")

    prompt = f"""You are a research specialist. Research the following topic thoroughly:

Topic: {topic}

Provide:
1. Key facts and statistics
2. Current trends
3. Important developments
4. Expert opinions (if available)

Be comprehensive but concise. Focus on factual information."""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    research_content = response.text.strip()
    print(f"  [RESEARCHER] Research complete ({len(research_content)} chars)")

    return {
        "research_content": research_content,
        "topic": topic
    }


async def writer(inputs: dict) -> dict:
    """
    Writer agent - creates blog post from research.

    LangGraph version:
        def writer(state):
            response = llm.invoke(f"Write a blog post about: {state['messages'][-1].content}")
            return {"messages": [response]}
    """
    research_content = inputs.get("research_content", "")
    topic = inputs.get("topic", "")

    print(f"  [WRITER] Creating blog post from research...")

    prompt = f"""You are a professional blog writer. Based on the research provided,
write an engaging blog post.

Research:
{research_content}

Requirements:
- Catchy title
- Engaging introduction
- Well-structured body with subheadings
- Clear conclusion
- Professional but accessible tone
- Around 500-800 words"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    blog_post = response.text.strip()
    print(f"  [WRITER] Blog post complete ({len(blog_post)} chars)")

    return {
        "blog_post": blog_post,
        "topic": topic
    }


async def run_research_demo():
    """
    Demonstrate multi-agent research and writing workflow.
    """
    print("=" * 60)
    print("hexDAG Research and Report Generator Demo")
    print("=" * 60)
    print()

    # Create graph
    graph = DirectedGraph()
    graph.add(NodeSpec("researcher", researcher))
    graph.add(NodeSpec("writer", writer, depends_on=["researcher"]))

    orchestrator = Orchestrator()

    # Test topics
    topics = [
        "the latest trends in AI",
        "sustainable energy solutions in 2024",
    ]

    for i, topic in enumerate(topics, 1):
        print(f"[Topic {i}] {topic}")
        print("-" * 50)

        # Run the pipeline
        research_result = await researcher({"topic": topic})

        writer_input = {
            "topic": topic,
            "research_content": research_result["research_content"]
        }
        writer_result = await writer(writer_input)

        # Display results
        print()
        print("=" * 60)
        print("RESEARCH SUMMARY:")
        print("=" * 60)
        # Show first 500 chars of research
        research_preview = research_result["research_content"][:500]
        print(research_preview + "..." if len(research_result["research_content"]) > 500 else research_preview)

        print()
        print("=" * 60)
        print("BLOG POST:")
        print("=" * 60)
        print(writer_result["blog_post"])

        print()
        print("=" * 60)
        print()


async def run_custom_topic(topic: str):
    """
    Run with a custom topic.
    """
    print(f"Researching and writing about: {topic}")
    print("-" * 50)

    research_result = await researcher({"topic": topic})
    writer_result = await writer({
        "topic": topic,
        "research_content": research_result["research_content"]
    })

    print()
    print("BLOG POST:")
    print("=" * 60)
    print(writer_result["blog_post"])

    return writer_result


async def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1:
        # Custom topic from command line
        topic = " ".join(sys.argv[1:])
        await run_custom_topic(topic)
    else:
        await run_research_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
