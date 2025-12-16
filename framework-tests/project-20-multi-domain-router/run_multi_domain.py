#!/usr/bin/env python
"""
hexDAG Multi-Domain Routing Agent Demo (Capstone)
Ported from LangGraph project-20-multi-domain-routing-agent

Pattern: Multi-Domain Routing
- Classify incoming queries by domain
- Route to specialized agents (Finance, HR, Marketing, General)
- Each agent handles domain-specific logic

This is a capstone project combining concepts from multiple previous projects.

Run with: ..\..\.venv\Scripts\python.exe run_multi_domain.py
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


# Domain-specific knowledge bases
HR_POLICIES = """
COMPANY HR POLICIES:
- Vacation: 20 days of paid time off per year
- Sick Leave: 10 days per year
- Health Insurance: Comprehensive plan, company pays 80%
- Remote Work: Up to 3 days per week allowed
- Parental Leave: 16 weeks for primary caregiver, 6 weeks for secondary
- 401(k): 4% company match
"""

BUDGET_DATA = {
    "Food": 200,
    "Rent": 1000,
    "Transport": 50,
    "Entertainment": 150,
    "Utilities": 100,
    "Savings": 300,
}

DOMAIN_KEYWORDS = {
    "finance": ["finance", "budget", "money", "expense", "cost", "spending", "savings", "income"],
    "hr": ["hr", "policy", "vacation", "leave", "benefits", "insurance", "401k", "remote", "pto"],
    "marketing": ["marketing", "slogan", "campaign", "brand", "advertisement", "promotion", "sales pitch"],
}


def classify_domain(query: str) -> str:
    """
    Classify query into a domain based on keywords.

    LangGraph version:
        def route_query(state):
            if "finance" in last_message: return "finance_agent"
            elif "hr" in last_message: return "hr_agent"
            ...
    """
    query_lower = query.lower()

    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                return domain

    return "general"


async def classify_query(inputs: dict) -> dict:
    """
    Classify the query and determine routing.
    """
    query = inputs.get("query", "")

    print(f"  [ROUTER] Classifying query...")

    domain = classify_domain(query)

    print(f"  [ROUTER] Detected domain: {domain.upper()}")

    return {
        "domain": domain,
        "query": query
    }


async def finance_agent(query: str) -> str:
    """
    Handle finance/budget queries.

    LangGraph version:
        df = pd.read_csv(io.StringIO(dummy_csv))
        summary = df.groupby("Category")["Amount"].sum().to_string()
        response = llm.invoke(f"Summarize this budget data: {summary}")
    """
    # Format budget data
    budget_lines = [f"- {cat}: ${amt}" for cat, amt in BUDGET_DATA.items()]
    total = sum(BUDGET_DATA.values())
    budget_summary = "\n".join(budget_lines) + f"\n\nTotal: ${total}"

    prompt = f"""You are a helpful finance assistant.

USER QUERY: {query}

BUDGET DATA:
{budget_summary}

Provide helpful financial advice based on the query and budget data.
Be specific and actionable."""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    return response.text.strip()


async def hr_agent(query: str) -> str:
    """
    Handle HR policy queries.

    LangGraph version:
        docs = hr_retriever.get_relevant_documents(query)
        response = llm.invoke(f"Answer based on HR policy: {context}")
    """
    prompt = f"""You are an HR assistant.

USER QUERY: {query}

HR POLICIES:
{HR_POLICIES}

Answer the question based on the company policies above.
If the answer isn't in the policies, say so clearly."""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    return response.text.strip()


async def marketing_agent(query: str) -> str:
    """
    Handle marketing queries.

    LangGraph version:
        response = llm.invoke("Generate a marketing slogan for a new product.")
    """
    prompt = f"""You are a creative marketing expert.

USER QUERY: {query}

Provide creative marketing assistance. If asked for slogans, provide 3 options.
Be creative, catchy, and memorable."""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    return response.text.strip()


async def general_agent(query: str) -> str:
    """
    Handle general queries.

    LangGraph version:
        response = llm.invoke(state["messages"])
    """
    prompt = f"""You are a helpful assistant.

USER QUERY: {query}

Provide a helpful response. Be friendly and informative."""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    return response.text.strip()


async def handle_query(inputs: dict) -> dict:
    """
    Route to appropriate agent based on domain classification.

    LangGraph uses conditional_edges:
        graph.add_conditional_edges("route_query", route_query, {...})

    hexDAG handles routing in a single node function.
    """
    domain = inputs.get("domain", "general")
    query = inputs.get("query", "")

    print(f"  [AGENT] Routing to {domain.upper()} agent...")

    # Route to appropriate agent
    if domain == "finance":
        response = await finance_agent(query)
    elif domain == "hr":
        response = await hr_agent(query)
    elif domain == "marketing":
        response = await marketing_agent(query)
    else:
        response = await general_agent(query)

    print(f"  [AGENT] Response generated")

    return {
        "domain": domain,
        "query": query,
        "response": response
    }


async def run_multi_domain_demo():
    """
    Demonstrate multi-domain routing with various query types.
    """
    print("=" * 60)
    print("hexDAG Multi-Domain Routing Agent (Capstone)")
    print("=" * 60)
    print()
    print("This unified assistant routes queries to specialized agents:")
    print("  - FINANCE: Budget, expenses, savings")
    print("  - HR: Policies, vacation, benefits")
    print("  - MARKETING: Slogans, campaigns, branding")
    print("  - GENERAL: Everything else")
    print()

    # Test queries (same as LangGraph example)
    test_queries = [
        ("Finance Query", "Can you give me a summary of my budget?"),
        ("HR Query", "What is the vacation policy?"),
        ("Marketing Query", "I need a new marketing slogan for a coffee shop."),
        ("General Query", "Tell me a joke about programming."),
    ]

    for query_type, query in test_queries:
        print(f"[{query_type}]")
        print(f"User: {query}")
        print("-" * 50)

        # Classify
        classify_result = await classify_query({"query": query})

        # Handle
        handle_result = await handle_query({
            "domain": classify_result["domain"],
            "query": query
        })

        print()
        print(f"Domain: {handle_result['domain'].upper()}")
        print(f"Response: {handle_result['response']}")
        print()
        print("=" * 60)
        print()


async def interactive_mode():
    """
    Interactive multi-domain assistant.
    """
    print("=" * 60)
    print("hexDAG Multi-Domain Assistant - Interactive Mode")
    print("=" * 60)
    print("Ask about finance, HR policies, marketing, or anything else.")
    print("Type 'quit' to exit.")
    print()

    while True:
        try:
            query = input("You: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not query:
                continue

            print()

            # Process query
            classify_result = await classify_query({"query": query})
            handle_result = await handle_query({
                "domain": classify_result["domain"],
                "query": query
            })

            print(f"\n[{handle_result['domain'].upper()} Agent]")
            print(f"Assistant: {handle_result['response']}")
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


async def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        await interactive_mode()
    else:
        await run_multi_domain_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
