#!/usr/bin/env python
"""
hexDAG Candidate Assessment System Demo
Ported from LangGraph project-13-candidate-assessment-system

Pattern: RAG + Assessment Pipeline
- Retrieve assessment criteria for the job role
- Evaluate candidate responses against criteria
- Provide structured feedback with scores

Run with: ..\..\.venv\Scripts\python.exe run_assessment.py
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


# Assessment criteria knowledge base (by role)
ASSESSMENT_CRITERIA = {
    "software_engineer": {
        "role": "Software Engineer",
        "criteria": """
        TECHNICAL SKILLS (40%):
        - Programming proficiency (Python, JavaScript, etc.)
        - Problem-solving approach
        - System design thinking
        - Code quality awareness

        COMMUNICATION (25%):
        - Clear explanations of technical concepts
        - Structured thinking
        - Ability to ask clarifying questions

        EXPERIENCE (20%):
        - Relevant project experience
        - Understanding of software development lifecycle
        - Familiarity with tools and frameworks

        CULTURAL FIT (15%):
        - Collaboration mindset
        - Growth orientation
        - Alignment with company values
        """
    },
    "data_scientist": {
        "role": "Data Scientist",
        "criteria": """
        TECHNICAL SKILLS (40%):
        - Statistical knowledge
        - Machine learning understanding
        - Python/R proficiency
        - Data visualization skills

        PROBLEM-SOLVING (25%):
        - Analytical thinking
        - Hypothesis formulation
        - Experimental design

        COMMUNICATION (20%):
        - Explaining complex concepts simply
        - Storytelling with data
        - Business acumen

        DOMAIN KNOWLEDGE (15%):
        - Industry understanding
        - Relevant experience
        - Curiosity and learning ability
        """
    },
    "product_manager": {
        "role": "Product Manager",
        "criteria": """
        STRATEGIC THINKING (30%):
        - Vision and roadmap planning
        - Market understanding
        - Prioritization skills

        COMMUNICATION (30%):
        - Stakeholder management
        - Clear articulation of ideas
        - Written and verbal skills

        EXECUTION (25%):
        - Project management
        - Data-driven decisions
        - Cross-functional collaboration

        LEADERSHIP (15%):
        - Influence without authority
        - Team motivation
        - Conflict resolution
        """
    }
}

# Sample interview questions and responses
SAMPLE_INTERVIEWS = [
    {
        "role": "software_engineer",
        "question": "Tell me about a challenging technical problem you solved recently.",
        "response": """
        Recently, I worked on optimizing a slow database query that was causing
        timeouts in production. The query was joining multiple tables and
        processing millions of rows.

        First, I analyzed the query execution plan and identified missing indexes.
        Then I refactored the query to use CTEs for better readability and added
        appropriate indexes. I also implemented pagination to handle large result sets.

        The result was a 95% improvement in query time, from 30 seconds to under
        2 seconds. I documented the changes and shared the learnings with my team.
        """
    },
    {
        "role": "software_engineer",
        "question": "How would you design a URL shortening service?",
        "response": """
        I would use a hash function to generate short codes. Store mappings in a
        database. Use caching for popular URLs. That's basically it.
        """
    },
    {
        "role": "data_scientist",
        "question": "How would you approach a problem where you need to predict customer churn?",
        "response": """
        For customer churn prediction, I would follow a structured approach:

        1. Data Collection: Gather historical customer data including usage patterns,
           support tickets, billing history, and engagement metrics.

        2. Feature Engineering: Create relevant features like days since last login,
           average session duration, support ticket frequency, and payment delays.

        3. Model Selection: Start with logistic regression as a baseline, then try
           random forest and gradient boosting. I'd use cross-validation to compare.

        4. Evaluation: Focus on precision-recall since churn is typically imbalanced.
           Calculate the business impact of false positives vs false negatives.

        5. Deployment: Implement the model with monitoring for drift, and create
           a dashboard for the business team to act on predictions.
        """
    }
]


async def retrieve_criteria(inputs: dict) -> dict:
    """
    Retrieve assessment criteria for the job role.

    LangGraph version uses FAISS vector store.
    hexDAG version uses simple lookup (could use FAISS too).
    """
    role_key = inputs.get("job_role", "software_engineer").lower().replace(" ", "_")

    print(f"  [RETRIEVE] Getting criteria for: {role_key}")

    criteria_data = ASSESSMENT_CRITERIA.get(role_key, ASSESSMENT_CRITERIA["software_engineer"])

    print(f"  [RETRIEVE] Found criteria for: {criteria_data['role']}")

    return {
        "criteria": criteria_data["criteria"],
        "role_name": criteria_data["role"]
    }


async def assess_response(inputs: dict) -> dict:
    """
    Assess candidate response against criteria.

    LangGraph version:
        response = llm.invoke(f"Assess the candidate response based on these criteria: {context}")

    hexDAG version: Structured assessment with scoring.
    """
    criteria = inputs.get("criteria", "")
    role_name = inputs.get("role_name", "")
    question = inputs.get("question", "")
    candidate_response = inputs.get("candidate_response", "")

    print(f"  [ASSESS] Evaluating candidate for: {role_name}")

    prompt = f"""You are an experienced interview assessor for a {role_name} position.
Evaluate the candidate's response based on the assessment criteria.

ASSESSMENT CRITERIA:
{criteria}

INTERVIEW QUESTION:
{question}

CANDIDATE'S RESPONSE:
{candidate_response}

Provide a structured assessment:

1. SCORE: (1-10, where 10 is exceptional)

2. STRENGTHS:
   - List 2-3 specific strengths demonstrated

3. AREAS FOR IMPROVEMENT:
   - List 2-3 specific areas to improve

4. DETAILED FEEDBACK:
   - Brief analysis of the response quality

5. RECOMMENDATION: (Choose one)
   - Strong Yes: Exceptional candidate, move forward immediately
   - Yes: Good candidate, recommend next round
   - Maybe: Some concerns, needs additional evaluation
   - No: Does not meet requirements

Be fair, objective, and constructive in your assessment."""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    assessment = response.text.strip()
    print(f"  [ASSESS] Assessment complete")

    return {
        "assessment": assessment,
        "role": role_name,
        "question": question
    }


async def run_assessment_demo():
    """
    Demonstrate candidate assessment system.
    """
    print("=" * 60)
    print("hexDAG Candidate Assessment System Demo")
    print("=" * 60)
    print()

    for i, interview in enumerate(SAMPLE_INTERVIEWS, 1):
        print(f"[Candidate {i}] Role: {interview['role'].replace('_', ' ').title()}")
        print(f"Question: {interview['question']}")
        print("-" * 50)

        # Retrieve criteria
        retrieve_result = await retrieve_criteria({
            "job_role": interview["role"]
        })

        # Assess response
        assess_input = {
            "criteria": retrieve_result["criteria"],
            "role_name": retrieve_result["role_name"],
            "question": interview["question"],
            "candidate_response": interview["response"]
        }
        assess_result = await assess_response(assess_input)

        # Display
        print()
        print("CANDIDATE RESPONSE:")
        print("-" * 30)
        print(interview["response"].strip())
        print()
        print("ASSESSMENT:")
        print("-" * 30)
        print(assess_result["assessment"])
        print()
        print("=" * 60)
        print()


async def assess_single_response(role: str, question: str, response: str):
    """
    Assess a single candidate response.
    """
    retrieve_result = await retrieve_criteria({"job_role": role})

    assess_result = await assess_response({
        "criteria": retrieve_result["criteria"],
        "role_name": retrieve_result["role_name"],
        "question": question,
        "candidate_response": response
    })

    print()
    print("ASSESSMENT:")
    print("=" * 50)
    print(assess_result["assessment"])

    return assess_result


async def main():
    """Main entry point."""
    await run_assessment_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
