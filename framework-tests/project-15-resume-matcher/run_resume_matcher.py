#!/usr/bin/env python
"""
hexDAG Resume/Job Description Matching Demo
Ported from LangGraph project-15-resume-job-description-matching

Pattern: Document Comparison + LLM Analysis
- Compare resume to job description
- Calculate match score based on keywords/skills
- Generate actionable feedback for candidates

Run with: ..\..\.venv\Scripts\python.exe run_resume_matcher.py
"""
import asyncio
import os
import re
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


# Sample resumes and job descriptions
SAMPLE_DATA = [
    {
        "name": "Strong Match",
        "resume": """
        JOHN DOE - Software Engineer

        SKILLS:
        - Python (5 years)
        - Machine Learning (PyTorch, TensorFlow)
        - Data Analysis (Pandas, NumPy)
        - SQL and NoSQL databases
        - REST API development
        - Git, Docker, AWS

        EXPERIENCE:
        Senior Software Engineer at TechCorp (2020-2024)
        - Built ML pipelines processing 1M+ records daily
        - Developed Python microservices
        - Led team of 3 developers

        EDUCATION:
        BS Computer Science, State University
        """,
        "job": """
        POSITION: Python Developer - AI Team

        REQUIREMENTS:
        - 3+ years Python experience
        - Experience with machine learning frameworks
        - Strong data analysis skills
        - Knowledge of cloud platforms (AWS preferred)
        - Team collaboration experience

        NICE TO HAVE:
        - Docker/Kubernetes experience
        - REST API development
        """
    },
    {
        "name": "Partial Match",
        "resume": """
        JANE SMITH - Web Developer

        SKILLS:
        - JavaScript, TypeScript
        - React, Vue.js
        - Node.js
        - HTML/CSS
        - Basic Python
        - MongoDB

        EXPERIENCE:
        Frontend Developer at WebAgency (2021-2024)
        - Built responsive web applications
        - Worked with REST APIs

        EDUCATION:
        Bootcamp Certificate, Code Academy
        """,
        "job": """
        POSITION: Python Developer - AI Team

        REQUIREMENTS:
        - 3+ years Python experience
        - Experience with machine learning frameworks
        - Strong data analysis skills
        - Knowledge of cloud platforms (AWS preferred)

        NICE TO HAVE:
        - Docker/Kubernetes experience
        """
    },
    {
        "name": "Weak Match",
        "resume": """
        BOB WILSON - Marketing Manager

        SKILLS:
        - Marketing strategy
        - Social media management
        - Content creation
        - Google Analytics
        - Basic Excel

        EXPERIENCE:
        Marketing Manager at AdCo (2019-2024)
        - Managed marketing campaigns
        - Increased engagement by 50%

        EDUCATION:
        BA Marketing, Business University
        """,
        "job": """
        POSITION: Python Developer - AI Team

        REQUIREMENTS:
        - 3+ years Python experience
        - Experience with machine learning frameworks
        - Strong data analysis skills
        - Knowledge of cloud platforms (AWS preferred)
        """
    }
]


def extract_keywords(text: str) -> set:
    """Extract relevant keywords from text (simple approach)."""
    # Common tech keywords to look for
    tech_keywords = {
        'python', 'java', 'javascript', 'typescript', 'react', 'vue', 'angular',
        'node', 'sql', 'nosql', 'mongodb', 'postgresql', 'mysql',
        'aws', 'azure', 'gcp', 'cloud', 'docker', 'kubernetes',
        'machine learning', 'ml', 'ai', 'artificial intelligence',
        'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit',
        'data analysis', 'data science', 'analytics',
        'rest', 'api', 'microservices', 'git', 'agile',
        'team', 'lead', 'senior', 'junior', 'experience'
    }

    text_lower = text.lower()
    found = set()

    for keyword in tech_keywords:
        if keyword in text_lower:
            found.add(keyword)

    return found


def calculate_match_score(resume_keywords: set, job_keywords: set) -> dict:
    """Calculate match score between resume and job keywords."""
    if not job_keywords:
        return {"score": 0, "matched": set(), "missing": set()}

    matched = resume_keywords.intersection(job_keywords)
    missing = job_keywords - resume_keywords
    score = len(matched) / len(job_keywords) * 100

    return {
        "score": round(score, 1),
        "matched": matched,
        "missing": missing
    }


async def compare_documents(inputs: dict) -> dict:
    """
    Compare resume to job description.

    LangGraph version uses embeddings:
        resume_vec = embeddings.embed_query(state["resume_text"])
        job_vec = embeddings.embed_query(state["job_description"])
        score = sum(a * b for a, b in zip(resume_vec, job_vec))

    hexDAG version uses keyword matching (simpler, no embedding API needed).
    """
    resume_text = inputs.get("resume_text", "")
    job_description = inputs.get("job_description", "")

    print(f"  [COMPARE] Analyzing resume vs job description...")

    # Extract keywords
    resume_keywords = extract_keywords(resume_text)
    job_keywords = extract_keywords(job_description)

    # Calculate match
    match_result = calculate_match_score(resume_keywords, job_keywords)

    # Format analysis
    analysis_lines = [
        f"KEYWORD MATCH SCORE: {match_result['score']}%",
        "",
        "MATCHING KEYWORDS:",
        f"  {', '.join(sorted(match_result['matched'])) or 'None'}",
        "",
        "MISSING FROM RESUME:",
        f"  {', '.join(sorted(match_result['missing'])) or 'None'}",
    ]

    match_analysis = "\n".join(analysis_lines)

    print(f"  [COMPARE] Match score: {match_result['score']}%")

    return {
        "match_analysis": match_analysis,
        "match_score": match_result["score"],
        "matched_keywords": list(match_result["matched"]),
        "missing_keywords": list(match_result["missing"])
    }


async def generate_feedback(inputs: dict) -> dict:
    """
    Generate detailed feedback for the candidate.

    LangGraph version:
        response = llm.invoke(f"Based on a match score of {state['match_score']}, provide feedback...")

    hexDAG version: More detailed analysis with specific recommendations.
    """
    resume_text = inputs.get("resume_text", "")
    job_description = inputs.get("job_description", "")
    match_analysis = inputs.get("match_analysis", "")
    match_score = inputs.get("match_score", 0)

    print(f"  [FEEDBACK] Generating detailed feedback...")

    prompt = f"""You are an expert career advisor. Analyze how well this resume matches the job description.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

PRELIMINARY ANALYSIS:
{match_analysis}

Provide a helpful assessment:

1. OVERALL MATCH: Rate as Strong Match / Partial Match / Weak Match

2. STRENGTHS: What qualifications does the candidate have that match the job?

3. GAPS: What key requirements is the candidate missing?

4. RECOMMENDATIONS: Specific advice for the candidate to improve their chances
   - Skills to develop
   - Experience to highlight
   - Resume improvements

5. VERDICT: Should this candidate apply? (Yes definitely / Yes with reservations / Consider other roles)

Be constructive and encouraging while being honest about gaps."""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    feedback = response.text.strip()
    print(f"  [FEEDBACK] Feedback generated")

    return {
        "feedback": feedback,
        "match_score": match_score
    }


async def run_resume_matcher_demo():
    """
    Demonstrate resume/job matching with different scenarios.
    """
    print("=" * 60)
    print("hexDAG Resume/Job Description Matcher Demo")
    print("=" * 60)
    print()

    for i, data in enumerate(SAMPLE_DATA, 1):
        print(f"[Scenario {i}] {data['name']}")
        print("-" * 50)

        # Compare documents
        compare_result = await compare_documents({
            "resume_text": data["resume"],
            "job_description": data["job"]
        })

        # Generate feedback
        feedback_input = {
            "resume_text": data["resume"],
            "job_description": data["job"],
            "match_analysis": compare_result["match_analysis"],
            "match_score": compare_result["match_score"]
        }
        feedback_result = await generate_feedback(feedback_input)

        # Display
        print()
        print("MATCH ANALYSIS:")
        print("-" * 30)
        print(compare_result["match_analysis"])
        print()
        print("CAREER ADVISOR FEEDBACK:")
        print("-" * 30)
        print(feedback_result["feedback"])
        print()
        print("=" * 60)
        print()


async def match_resume_to_job(resume: str, job: str):
    """
    Match a single resume to a job description.
    """
    print("Analyzing resume match...")
    print("-" * 50)

    compare_result = await compare_documents({
        "resume_text": resume,
        "job_description": job
    })

    feedback_result = await generate_feedback({
        "resume_text": resume,
        "job_description": job,
        "match_analysis": compare_result["match_analysis"],
        "match_score": compare_result["match_score"]
    })

    print()
    print("FEEDBACK:")
    print("=" * 50)
    print(feedback_result["feedback"])

    return feedback_result


async def main():
    """Main entry point."""
    await run_resume_matcher_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
