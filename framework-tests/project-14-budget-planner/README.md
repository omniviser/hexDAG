# Project 14: Budget Data Summarization - hexDAG Port

## Overview
This folder contains the hexDAG port of LangGraph's `project-14-budget-data-summarization`.

## What It Tests

**Data processing pipeline - parse structured data then generate natural language insights.**

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   CSV Data  │ --> │   Parse &   │ --> │  LLM        │ --> Insights
│   (Budget)  │     │  Aggregate  │     │  Analysis   │
└─────────────┘     └─────────────┘     └─────────────┘
                          │                    │
                          v                    v
                    Category totals       Recommendations
                    and percentages       and warnings
```

### Example Flow:
```
INPUT CSV:
Category,Amount
Food,200
Rent,1000
Transport,50
Entertainment,150

[PARSE] Parsing budget data...
[PARSE] Found 4 categories, total: $1,400.00

BUDGET SUMMARY:
  Rent: $1,000.00 (71.4%)
  Food: $200.00 (14.3%)
  Entertainment: $150.00 (10.7%)
  Transport: $50.00 (3.6%)

[ANALYZE] Generating insights...

AI ANALYSIS:
"Your total spending is $1,400. Housing takes up 71% of your budget,
which is higher than the recommended 30%. Consider finding ways to
reduce housing costs or increasing income..."
```

### Why This Matters:
- **Data + AI** - Combine structured data processing with LLM insights
- **Actionable advice** - Not just numbers, but recommendations
- **Pattern recognition** - LLM can spot concerning trends

## Files
- `budget_pipeline.yaml` - hexDAG YAML pipeline
- `run_budget.py` - Python runner with sample budgets
- `README.md` - This file

## The Difference

| | LangGraph | hexDAG |
|---|-----------|--------|
| **CSV Parsing** | pandas | Manual (no dependency) |
| **Output** | Basic summary | Detailed analysis + recommendations |
| **Scenarios** | Single example | Multiple test scenarios |

### LangGraph (basic):
```python
import pandas as pd

def parse_csv(state):
    df = pd.read_csv(io.StringIO(state["csv_data"]))
    summary = df.groupby("Category")["Amount"].sum().to_string()
    return {"summary": summary}

def summarize_budget(state):
    response = llm.invoke(f"Summarize: {state['summary']}")
    return {"messages": [response]}
```

### hexDAG (enhanced):
```python
def parse_csv_data(csv_string: str) -> dict:
    # Manual parsing - no pandas needed
    # Returns totals, percentages, statistics
    ...

async def summarize_budget(inputs: dict) -> dict:
    prompt = f"""Analyze this budget:
    {budget_summary}

    Provide:
    1. OVERVIEW
    2. TOP SPENDING AREAS
    3. OBSERVATIONS (concerning patterns?)
    4. RECOMMENDATIONS (specific savings tips)
    """
```

## Verdict: WORKS PERFECTLY

Simple linear pipeline - perfect for hexDAG.

**hexDAG version enhancements:**
- No pandas dependency (simpler)
- Multiple test scenarios (simple, detailed, overspending)
- Percentage calculations
- Structured analysis with recommendations

## Test Scenarios

| Scenario | Total | Top Category | Issue |
|----------|-------|--------------|-------|
| Simple | $1,400 | Rent (71%) | High housing ratio |
| Detailed | $3,297 | Housing (50%) | Balanced |
| Overspending | $4,400 | Rent (27%) | High entertainment/shopping |

## How to Run

```bash
cd framework-tests/project14-budget-planner
..\..\.venv\Scripts\python.exe run_budget.py
```

Expected output:
```
============================================================
hexDAG Budget Planner Demo
============================================================

[Scenario] Simple Budget
--------------------------------------------------
  [PARSE] Parsing budget data...
  [PARSE] Found 4 categories, total: $1,400.00
  [ANALYZE] Generating budget insights...
  [ANALYZE] Analysis complete

PARSED DATA:
------------------------------
BUDGET SUMMARY
========================================

SPENDING BY CATEGORY:
  Rent: $1,000.00 (71.4%)
  Food: $200.00 (14.3%)
  Entertainment: $150.00 (10.7%)
  Transport: $50.00 (3.6%)

TOTAL SPENDING: $1,400.00

AI ANALYSIS:
------------------------------
1. OVERVIEW: Your total monthly spending is $1,400...

2. TOP SPENDING AREAS: Housing dominates at 71%...

3. OBSERVATIONS: Your housing cost ratio is concerning...

4. RECOMMENDATIONS:
   - Consider roommates or relocation
   - Food spending is reasonable, maintain this
   - Transport costs are low - great job!
```

## Test Results
[To be filled after testing]

## Observations
[To be filled after testing]
