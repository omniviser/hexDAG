#!/usr/bin/env python
"""
hexDAG Budget Planner Demo
Ported from LangGraph project-14-budget-data-summarization

Pattern: Data Processing + LLM Analysis
- Parse CSV budget data
- Aggregate expenses by category
- Generate natural language insights and recommendations

Run with: ..\..\.venv\Scripts\python.exe run_budget.py
"""
import asyncio
import io
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


# Sample budget data (more detailed than LangGraph example)
SAMPLE_BUDGETS = {
    "simple": """Category,Amount
Food,200
Rent,1000
Transport,50
Entertainment,150
""",
    "detailed": """Category,Amount,Date,Description
Housing,1500,2024-01-01,Monthly rent
Housing,150,2024-01-05,Utilities
Food,120,2024-01-03,Groceries
Food,85,2024-01-10,Groceries
Food,45,2024-01-15,Restaurant
Food,60,2024-01-22,Restaurant
Transport,100,2024-01-01,Gas
Transport,50,2024-01-15,Gas
Transport,25,2024-01-20,Parking
Entertainment,15,2024-01-05,Netflix
Entertainment,12,2024-01-05,Spotify
Entertainment,80,2024-01-12,Concert tickets
Healthcare,200,2024-01-08,Doctor visit
Healthcare,45,2024-01-20,Pharmacy
Shopping,250,2024-01-14,Clothing
Shopping,80,2024-01-25,Electronics
Savings,500,2024-01-01,Monthly savings
""",
    "overspending": """Category,Amount,Date
Rent,1200,2024-01-01
Food,800,2024-01-15
Entertainment,600,2024-01-10
Shopping,900,2024-01-20
Transport,300,2024-01-05
Subscriptions,150,2024-01-01
Dining Out,450,2024-01-18
"""
}


def parse_csv_data(csv_string: str) -> dict:
    """
    Parse CSV data without pandas (to avoid dependency).
    Returns category totals and statistics.
    """
    lines = csv_string.strip().split('\n')
    headers = [h.strip() for h in lines[0].split(',')]

    # Find column indices
    category_idx = headers.index('Category') if 'Category' in headers else 0
    amount_idx = headers.index('Amount') if 'Amount' in headers else 1

    # Aggregate by category
    category_totals = {}
    total_amount = 0

    for line in lines[1:]:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        category = parts[category_idx]
        try:
            amount = float(parts[amount_idx])
        except (ValueError, IndexError):
            continue

        category_totals[category] = category_totals.get(category, 0) + amount
        total_amount += amount

    return {
        "category_totals": category_totals,
        "total": total_amount,
        "num_categories": len(category_totals)
    }


async def parse_csv(inputs: dict) -> dict:
    """
    Parse CSV and aggregate expenses by category.

    LangGraph version:
        df = pd.read_csv(io.StringIO(state["csv_data"]))
        summary = df.groupby("Category")["Amount"].sum().to_string()

    hexDAG version: Manual parsing (no pandas dependency).
    """
    csv_data = inputs.get("csv_data", "")

    print(f"  [PARSE] Parsing budget data...")

    parsed = parse_csv_data(csv_data)

    # Format summary
    summary_lines = ["BUDGET SUMMARY", "=" * 40, ""]
    summary_lines.append("SPENDING BY CATEGORY:")

    # Sort by amount descending
    sorted_categories = sorted(
        parsed["category_totals"].items(),
        key=lambda x: x[1],
        reverse=True
    )

    for category, amount in sorted_categories:
        percentage = (amount / parsed["total"]) * 100 if parsed["total"] > 0 else 0
        summary_lines.append(f"  {category}: ${amount:,.2f} ({percentage:.1f}%)")

    summary_lines.append("")
    summary_lines.append(f"TOTAL SPENDING: ${parsed['total']:,.2f}")
    summary_lines.append(f"NUMBER OF CATEGORIES: {parsed['num_categories']}")

    budget_summary = "\n".join(summary_lines)

    print(f"  [PARSE] Found {parsed['num_categories']} categories, total: ${parsed['total']:,.2f}")

    return {
        "budget_summary": budget_summary,
        "total": parsed["total"],
        "category_totals": parsed["category_totals"]
    }


async def summarize_budget(inputs: dict) -> dict:
    """
    Generate natural language budget analysis.

    LangGraph version:
        response = llm.invoke(f"Summarize the following budget data:\n\n{state['summary']}")

    hexDAG version: More detailed analysis with recommendations.
    """
    budget_summary = inputs.get("budget_summary", "")
    total = inputs.get("total", 0)

    print(f"  [ANALYZE] Generating budget insights...")

    prompt = f"""You are a financial advisor. Analyze this budget data and provide helpful insights.

{budget_summary}

Provide a clear analysis including:

1. OVERVIEW: Brief summary of total spending and main categories

2. TOP SPENDING AREAS: Identify the largest expense categories

3. OBSERVATIONS:
   - Any concerning spending patterns?
   - Categories that seem high or low?
   - Balance between needs vs wants?

4. RECOMMENDATIONS:
   - Specific suggestions to optimize the budget
   - Areas where savings might be possible
   - Good habits to maintain

Keep the analysis practical and actionable. Be encouraging but honest."""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    analysis = response.text.strip()
    print(f"  [ANALYZE] Analysis complete")

    return {
        "analysis": analysis,
        "total_spending": total
    }


async def run_budget_demo():
    """
    Demonstrate budget analysis with different datasets.
    """
    print("=" * 60)
    print("hexDAG Budget Planner Demo")
    print("=" * 60)
    print()

    # Test with different budget scenarios
    scenarios = [
        ("Simple Budget", "simple"),
        ("Detailed Monthly Budget", "detailed"),
        ("Overspending Scenario", "overspending"),
    ]

    for scenario_name, scenario_key in scenarios:
        print(f"[Scenario] {scenario_name}")
        print("-" * 50)

        csv_data = SAMPLE_BUDGETS[scenario_key]

        # Parse CSV
        parse_result = await parse_csv({"csv_data": csv_data})

        # Generate analysis
        analyze_input = {
            "budget_summary": parse_result["budget_summary"],
            "total": parse_result["total"]
        }
        analyze_result = await summarize_budget(analyze_input)

        # Display
        print()
        print("PARSED DATA:")
        print("-" * 30)
        print(parse_result["budget_summary"])
        print()
        print("AI ANALYSIS:")
        print("-" * 30)
        print(analyze_result["analysis"])
        print()
        print("=" * 60)
        print()


async def analyze_custom_budget(csv_data: str):
    """
    Analyze custom budget data.
    """
    print("Analyzing your budget...")
    print("-" * 50)

    parse_result = await parse_csv({"csv_data": csv_data})
    analyze_result = await summarize_budget({
        "budget_summary": parse_result["budget_summary"],
        "total": parse_result["total"]
    })

    print()
    print("ANALYSIS:")
    print("=" * 50)
    print(analyze_result["analysis"])

    return analyze_result


async def main():
    """Main entry point."""
    await run_budget_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
