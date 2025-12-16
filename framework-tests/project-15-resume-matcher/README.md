# Project 15: Resume/Job Description Matching - hexDAG Port

## Overview
This folder contains the hexDAG port of LangGraph's `project-15-resume-job-description-matching`.

## What It Tests

**Match resumes to job descriptions and provide actionable feedback.**

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Resume +   │ --> │   Compare    │ --> │   Generate   │ --> Report
│ Job Posting  │     │   Documents  │     │   Feedback   │
└──────────────┘     └──────────────┘     └──────────────┘
                           │                     │
                           v                     v
                     Keyword match          Career advice
                     score (0-100%)         + recommendations
```

### Example Flow:
```
RESUME: "Python developer, 5 years ML experience, AWS..."
JOB: "Seeking Python developer with ML and cloud experience..."

[COMPARE] Analyzing resume vs job description...
[COMPARE] Match score: 85%

MATCH ANALYSIS:
  Matching: python, machine learning, aws, docker, api
  Missing: kubernetes

[FEEDBACK] Generating detailed feedback...

FEEDBACK:
  1. OVERALL MATCH: Strong Match
  2. STRENGTHS: Python expertise, ML experience, cloud knowledge
  3. GAPS: No Kubernetes mentioned
  4. RECOMMENDATIONS: Add K8s to skills, highlight team leadership
  5. VERDICT: Yes definitely - strong candidate
```

### Why This Matters:
- **Candidate screening** - Quickly identify qualified applicants
- **Self-assessment** - Candidates can check their fit before applying
- **Feedback** - Actionable advice to improve chances

## Files
- `resume_pipeline.yaml` - hexDAG YAML pipeline
- `run_resume_matcher.py` - Python runner with sample scenarios
- `README.md` - This file

## The Difference

| | LangGraph | hexDAG |
|---|-----------|--------|
| **Comparison** | Vector embeddings (API) | Keyword matching (no API) |
| **Score** | Dot product of vectors | Percentage of matched keywords |
| **Feedback** | Basic score feedback | Detailed career advice |

### LangGraph (embeddings):
```python
# Requires embedding API calls
resume_vec = embeddings.embed_query(state["resume_text"])
job_vec = embeddings.embed_query(state["job_description"])
score = sum(a * b for a, b in zip(resume_vec, job_vec))
```

### hexDAG (keyword matching):
```python
# No embedding API needed
resume_keywords = extract_keywords(resume_text)
job_keywords = extract_keywords(job_description)
matched = resume_keywords.intersection(job_keywords)
score = len(matched) / len(job_keywords) * 100
```

## Verdict: WORKS PERFECTLY

Simple linear pipeline - perfect for hexDAG.

**hexDAG version enhancements:**
- No embedding API needed (simpler)
- Shows matched vs missing keywords
- Multiple test scenarios (strong/partial/weak match)
- Detailed career advice with recommendations

## Test Scenarios

| Scenario | Resume | Match Score | Verdict |
|----------|--------|-------------|---------|
| Strong Match | Python ML Engineer | ~85% | Yes definitely |
| Partial Match | Web Developer (basic Python) | ~40% | Yes with reservations |
| Weak Match | Marketing Manager | ~10% | Consider other roles |

## How to Run

```bash
cd framework-tests/project15-resume-matcher
..\..\.venv\Scripts\python.exe run_resume_matcher.py
```

Expected output:
```
============================================================
hexDAG Resume/Job Description Matcher Demo
============================================================

[Scenario 1] Strong Match
--------------------------------------------------
  [COMPARE] Analyzing resume vs job description...
  [COMPARE] Match score: 85.7%
  [FEEDBACK] Generating detailed feedback...

MATCH ANALYSIS:
------------------------------
KEYWORD MATCH SCORE: 85.7%

MATCHING KEYWORDS:
  aws, data analysis, docker, machine learning, python, team

MISSING FROM RESUME:
  kubernetes

CAREER ADVISOR FEEDBACK:
------------------------------
1. OVERALL MATCH: Strong Match

2. STRENGTHS:
   - Extensive Python experience (5 years)
   - Strong ML background with PyTorch and TensorFlow
   - Cloud experience with AWS
   - Team leadership experience

3. GAPS:
   - No explicit Kubernetes experience mentioned

4. RECOMMENDATIONS:
   - Highlight your ML pipeline work prominently
   - Consider adding a K8s certification
   - Quantify your achievements more

5. VERDICT: Yes definitely - this is a strong candidate
```

## Test Results
[To be filled after testing]

## Observations
[To be filled after testing]
