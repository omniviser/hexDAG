#!/usr/bin/env python
"""
hexDAG Marketing Slogan Generator Demo
Ported from LangGraph project-09-creative-content-generation

Pattern: Iterative refinement of creative content
- Generate initial slogan
- Refine multiple times
- Evaluate final result

LangGraph uses cycles (generate <-> refine with recursion_limit)
hexDAG uses linear DAG (generate -> refine_1 -> refine_2 -> evaluate)

Run with: ..\..\.venv\Scripts\python.exe run_marketing.py
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


async def generate_slogan(inputs: dict) -> dict:
    """
    Generate initial marketing slogan.

    LangGraph version:
        def generate_slogan(state):
            response = llm.invoke("Generate a marketing slogan...")
            return {"messages": [response]}
    """
    product = inputs.get("product_description", "a new coffee shop")

    print(f"  [GENERATE] Creating initial slogan for: {product}")

    prompt = f"""Generate a creative marketing slogan for: {product}

Requirements:
- Catchy and memorable
- Short (under 10 words)
- Highlights key benefits

Respond with ONLY the slogan, nothing else."""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    slogan = response.text.strip().strip('"')

    print(f"  [GENERATE] Initial: \"{slogan}\"")

    return {
        "initial_slogan": slogan,
        "iteration": 0
    }


async def refine_slogan(inputs: dict) -> dict:
    """
    Refine an existing slogan.

    LangGraph version:
        def refine_slogan(state):
            response = llm.invoke(f"Refine this slogan: {state['messages'][-1].content}")
            return {"messages": [response]}
    """
    # Get the slogan to refine (could be initial or previously refined)
    current_slogan = inputs.get("current_slogan") or inputs.get("initial_slogan", "")
    product = inputs.get("product_description", "a new coffee shop")
    iteration = inputs.get("iteration", 0) + 1

    print(f"  [REFINE {iteration}] Improving: \"{current_slogan}\"")

    prompt = f"""Refine this marketing slogan to make it better:
"{current_slogan}"

Product: {product}

Make it:
- More catchy
- More memorable
- Better rhythm

Respond with ONLY the refined slogan, nothing else."""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    refined = response.text.strip().strip('"')

    print(f"  [REFINE {iteration}] Result: \"{refined}\"")

    return {
        "current_slogan": refined,
        "iteration": iteration
    }


async def evaluate_slogan(inputs: dict) -> dict:
    """
    Evaluate the final slogan.
    """
    final_slogan = inputs.get("current_slogan", "")
    product = inputs.get("product_description", "a new coffee shop")

    print(f"  [EVALUATE] Analyzing final slogan...")

    prompt = f"""Evaluate this marketing slogan for {product}:

Final slogan: "{final_slogan}"

Provide a brief analysis:
1. Rating (1-10)
2. Key strengths
3. Why it works"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    return {
        "final_slogan": final_slogan,
        "evaluation": response.text.strip()
    }


async def run_marketing_demo():
    """
    Demonstrate iterative slogan refinement.

    LangGraph uses: generate <-> refine (cycle with recursion_limit=3)
    hexDAG uses: generate -> refine -> refine -> evaluate (linear DAG)
    """
    print("=" * 60)
    print("hexDAG Marketing Slogan Generator Demo")
    print("=" * 60)
    print()

    # Create graph with multiple refinement steps (DAG equivalent of cycles)
    graph = DirectedGraph()
    graph.add(NodeSpec("generate", generate_slogan))
    graph.add(NodeSpec("refine_1", refine_slogan, depends_on=["generate"]))
    graph.add(NodeSpec("refine_2", refine_slogan, depends_on=["refine_1"]))
    graph.add(NodeSpec("evaluate", evaluate_slogan, depends_on=["refine_2"]))

    orchestrator = Orchestrator()

    # Test with different products
    products = [
        "a new coffee shop called 'Morning Brew'",
        "an AI-powered fitness app",
        "eco-friendly reusable water bottles",
    ]

    for i, product in enumerate(products, 1):
        print(f"[Product {i}] {product}")
        print("-" * 50)

        # Run the pipeline
        # We need to pass data through the chain manually
        result = {}

        # Step 1: Generate
        gen_result = await generate_slogan({"product_description": product})
        result["generate"] = gen_result

        # Step 2: Refine 1
        refine1_input = {
            "product_description": product,
            "initial_slogan": gen_result["initial_slogan"],
            "iteration": gen_result["iteration"]
        }
        refine1_result = await refine_slogan(refine1_input)
        result["refine_1"] = refine1_result

        # Step 3: Refine 2
        refine2_input = {
            "product_description": product,
            "current_slogan": refine1_result["current_slogan"],
            "iteration": refine1_result["iteration"]
        }
        refine2_result = await refine_slogan(refine2_input)
        result["refine_2"] = refine2_result

        # Step 4: Evaluate
        eval_input = {
            "product_description": product,
            "current_slogan": refine2_result["current_slogan"]
        }
        eval_result = await evaluate_slogan(eval_input)
        result["evaluate"] = eval_result

        print()
        print("FINAL SLOGAN:")
        print(f'  "{eval_result["final_slogan"]}"')
        print()
        print("EVALUATION:")
        print(eval_result["evaluation"])
        print()
        print("=" * 60)
        print()


async def run_with_custom_iterations(product: str, num_refinements: int = 3):
    """
    Run with configurable number of refinements.

    This shows how hexDAG can handle variable iterations
    by building the DAG dynamically.
    """
    print(f"Generating slogan for: {product}")
    print(f"Refinement iterations: {num_refinements}")
    print("-" * 50)

    # Generate initial
    result = await generate_slogan({"product_description": product})
    current_slogan = result["initial_slogan"]

    # Refine N times
    for i in range(num_refinements):
        refine_input = {
            "product_description": product,
            "current_slogan": current_slogan,
            "iteration": i
        }
        refine_result = await refine_slogan(refine_input)
        current_slogan = refine_result["current_slogan"]

    # Evaluate
    eval_result = await evaluate_slogan({
        "product_description": product,
        "current_slogan": current_slogan
    })

    print()
    print(f'FINAL: "{eval_result["final_slogan"]}"')
    print()
    return eval_result


async def main():
    """Main entry point."""
    await run_marketing_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
