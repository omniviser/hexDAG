# Project 13: Candidate Assessment System - hexDAG Port

## Overview
This folder contains the hexDAG port of LangGraph's `project-13-candidate-assessment-system`.

## What It Tests

**Interview coach that evaluates candidate responses against role-specific criteria.**

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Candidate  │     │  Retrieve   │     │   Assess    │
│  Response   │ --> │  Criteria   │ --> │  Response   │ --> Feedback
└─────────────┘     └─────────────┘     └─────────────┘
                          │                    │
                          v                    v
                    Role-specific         Score + Strengths
                    rubrics               + Improvements
```

### Example Flow:
```
Role: Software Engineer
Question: "Tell me about a challenging technical problem you solved."

[RETRIEVE] Getting criteria for: software_engineer
[RETRIEVE] Found: Technical Skills (40%), Communication (25%)...

[ASSESS] Evaluating candidate...

ASSESSMENT:
1. SCORE: 8/10
2. STRENGTHS:
   - Clear problem-solving methodology
   - Quantified results (95% improvement)
   - Team collaboration mentioned
3. AREAS FOR IMPROVEMENT:
   - Could discuss alternative approaches
   - More detail on technical decisions
4. RECOMMENDATION: Yes - Move to next round
```

### Why This Matters:
- **Consistency** - Same criteria applied to all candidates
- **Objectivity** - Reduces interviewer bias
- **Efficiency** - Faster initial screening
- **Scalability** - Handle many candidates

## Files
- `assessment_pipeline.yaml` - hexDAG YAML pipeline
- `run_assessment.py` - Python runner with sample interviews
- `README.md` - This file

## The Difference

| | LangGraph | hexDAG |
|---|-----------|--------|
| **Criteria Storage** | FAISS vector store | Simple dictionary (could use FAISS) |
| **Assessment** | Basic prompt | Structured scoring rubric |
| **Output** | Free text | Score + Strengths + Recommendation |

### LangGraph (basic):
```python
# Single hardcoded criteria string
assessment_criteria = "Key skills: Python, LangGraph, problem-solving."

def assess_response(state):
    docs = retriever.get_relevant_documents("assessment criteria")
    response = llm.invoke(f"Assess based on: {docs}")
    return {"messages": [response]}
```

### hexDAG (enhanced):
```python
# Role-specific criteria with weighted categories
ASSESSMENT_CRITERIA = {
    "software_engineer": {
        "criteria": """
        TECHNICAL SKILLS (40%): ...
        COMMUNICATION (25%): ...
        EXPERIENCE (20%): ...
        """
    }
}

async def assess_response(inputs):
    # Structured prompt with scoring rubric
    prompt = f"""
    Provide:
    1. SCORE: (1-10)
    2. STRENGTHS: ...
    3. AREAS FOR IMPROVEMENT: ...
    4. RECOMMENDATION: (Strong Yes / Yes / Maybe / No)
    """
```

## Verdict: WORKS PERFECTLY

Simple linear pipeline - perfect for hexDAG.

**hexDAG version is enhanced with:**
- Multiple role types (Software Engineer, Data Scientist, Product Manager)
- Weighted assessment categories
- Structured output (score, strengths, improvements, recommendation)
- Multiple sample interviews for testing

## Supported Roles

| Role | Key Criteria |
|------|--------------|
| Software Engineer | Technical (40%), Communication (25%), Experience (20%) |
| Data Scientist | Technical (40%), Problem-Solving (25%), Communication (20%) |
| Product Manager | Strategic (30%), Communication (30%), Execution (25%) |

## How to Run

```bash
cd framework-tests/project13-candidate-assessment
..\..\.venv\Scripts\python.exe run_assessment.py
```

Expected output:
```
============================================================
hexDAG Candidate Assessment System Demo
============================================================

[Candidate 1] Role: Software Engineer
Question: Tell me about a challenging technical problem you solved.
--------------------------------------------------
  [RETRIEVE] Getting criteria for: software_engineer
  [RETRIEVE] Found criteria for: Software Engineer
  [ASSESS] Evaluating candidate for: Software Engineer
  [ASSESS] Assessment complete

CANDIDATE RESPONSE:
------------------------------
Recently, I worked on optimizing a slow database query...

ASSESSMENT:
------------------------------
1. SCORE: 8/10

2. STRENGTHS:
   - Systematic problem-solving approach
   - Quantified results (95% improvement)
   - Knowledge sharing with team

3. AREAS FOR IMPROVEMENT:
   - Could mention testing strategy
   - Discuss trade-offs considered

4. RECOMMENDATION: Yes - Good candidate, recommend next round
```

## Test Results
[To be filled after testing]

## Observations
[To be filled after testing]
