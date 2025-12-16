#!/usr/bin/env python
"""
hexDAG Web Research and Report Generation Demo
Ported from LangGraph project-18-web-research-and-report-generation

Pattern: Web Research + Report Generation
- Search the web for information on a topic
- Compile research data from multiple sources
- Generate a comprehensive, structured report

Run with: ..\..\.venv\Scripts\python.exe run_web_research.py
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
from duckduckgo_search import DDGS
from hexdag.core.domain.dag import DirectedGraph, NodeSpec
from hexdag.core.orchestration.orchestrator import Orchestrator

# Configure Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env file")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)


async def conduct_research(inputs: dict) -> dict:
    """
    Search the web for information on a topic.

    LangGraph version uses Tavily:
        tool = TavilySearchResults(max_results=3)
        search_results = tool.invoke(query)

    hexDAG version uses DuckDuckGo (free, no API key).
    """
    topic = inputs.get("topic", "")

    print(f"  [RESEARCH] Searching web for: {topic}")

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(topic, max_results=5))

        # Format research data
        research_parts = []
        sources = []

        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            body = result.get('body', 'No content')
            url = result.get('href', '')

            research_parts.append(f"SOURCE {i}: {title}\n{body}")
            if url:
                sources.append(f"[{i}] {title}: {url}")

        research_data = "\n\n".join(research_parts)
        sources_list = "\n".join(sources)

        print(f"  [RESEARCH] Found {len(results)} sources")

        return {
            "research_data": research_data,
            "sources": sources_list,
            "source_count": len(results)
        }

    except Exception as e:
        print(f"  [RESEARCH] Error: {e}")
        return {
            "research_data": f"Search error: {str(e)}",
            "sources": "",
            "source_count": 0
        }


async def generate_report(inputs: dict) -> dict:
    """
    Generate a comprehensive report from research data.

    LangGraph version:
        response = llm.invoke(f"Generate a comprehensive report based on: {research_data}")

    hexDAG version: Structured report with sections.
    """
    topic = inputs.get("topic", "")
    research_data = inputs.get("research_data", "")
    sources = inputs.get("sources", "")

    print(f"  [REPORT] Generating comprehensive report...")

    prompt = f"""You are a professional research analyst. Generate a comprehensive report based on the following research data.

TOPIC: {topic}

RESEARCH DATA:
{research_data}

Create a well-structured report with the following sections:

# REPORT: {topic}

## 1. EXECUTIVE SUMMARY
(2-3 sentence overview of key findings)

## 2. KEY FINDINGS
(Bullet points of the most important discoveries)

## 3. DETAILED ANALYSIS
(In-depth discussion of the topic based on research)

## 4. CURRENT TRENDS
(What's happening now in this space)

## 5. FUTURE OUTLOOK
(Predictions and expectations)

## 6. CONCLUSIONS
(Summary and final thoughts)

## 7. SOURCES
{sources}

Make the report informative, well-organized, and professional."""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    report = response.text.strip()
    print(f"  [REPORT] Report generated ({len(report)} chars)")

    return {
        "report": report,
        "topic": topic
    }


async def run_research_demo():
    """
    Demonstrate web research and report generation.
    """
    print("=" * 60)
    print("hexDAG Web Research and Report Generator Demo")
    print("=" * 60)
    print()

    # Research topics
    topics = [
        "Latest trends in renewable energy 2024",
        "Artificial intelligence in healthcare",
    ]

    for i, topic in enumerate(topics, 1):
        print(f"[Topic {i}] {topic}")
        print("-" * 50)

        # Conduct research
        research_result = await conduct_research({"topic": topic})

        # Generate report
        report_input = {
            "topic": topic,
            "research_data": research_result["research_data"],
            "sources": research_result["sources"]
        }
        report_result = await generate_report(report_input)

        # Display
        print()
        print("RESEARCH SOURCES FOUND:")
        print("-" * 30)
        print(research_result["sources"] or "No sources found")
        print()
        print("GENERATED REPORT:")
        print("-" * 30)
        print(report_result["report"])
        print()
        print("=" * 60)
        print()


async def research_topic(topic: str):
    """
    Research a single custom topic.
    """
    print(f"Researching: {topic}")
    print("-" * 50)

    research_result = await conduct_research({"topic": topic})
    report_result = await generate_report({
        "topic": topic,
        "research_data": research_result["research_data"],
        "sources": research_result["sources"]
    })

    print()
    print("REPORT:")
    print("=" * 60)
    print(report_result["report"])

    return report_result


async def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1:
        # Custom topic from command line
        topic = " ".join(sys.argv[1:])
        await research_topic(topic)
    else:
        await run_research_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
