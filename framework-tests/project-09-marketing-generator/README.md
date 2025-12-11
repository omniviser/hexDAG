# Project 9: Marketing Slogan Generator - hexDAG Port

## Overview
This folder contains the hexDAG port of LangGraph's `project-09-creative-content-generation`.

## What It Tests

**Iterative refinement of creative content using cycles.**

```
LangGraph (cycles):
  ┌──────────┐     ┌──────────┐
  │ Generate │ <-> │  Refine  │  (loops until recursion_limit)
  └──────────┘     └──────────┘

hexDAG (linear DAG):
  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ Generate │ -> │ Refine 1 │ -> │ Refine 2 │ -> │ Evaluate │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### Example Flow:
```
Product: "a new coffee shop"

[GENERATE] "Wake up to Morning Brew"
    ↓
[REFINE 1] "Morning Brew: Where Every Sip Sparks Your Day"
    ↓
[REFINE 2] "Spark Your Day at Morning Brew"
    ↓
[EVALUATE] Rating: 8/10 - Catchy, memorable, action-oriented
```

### Why This Matters:
- **Creative iteration** - First draft is rarely the best
- **Quality improvement** - Each pass polishes the output
- **Automated refinement** - No manual back-and-forth

## Files
- `marketing_pipeline.yaml` - hexDAG YAML pipeline
- `run_marketing.py` - Python runner
- `README.md` - This file

## The Difference

| | LangGraph | hexDAG |
|---|-----------|--------|
| **Pattern** | Cycles (generate <-> refine) | Linear DAG |
| **Iterations** | Dynamic (recursion_limit) | Fixed (pre-defined nodes) |
| **Completion** | May need limit to stop | Always completes |

### LangGraph (cycles):
```python
graph.add_edge("generate", "refine")
graph.add_edge("refine", "generate")  # <- CYCLE!

app.invoke(None, {"recursion_limit": 3})  # Must limit or infinite loop
```

### hexDAG (linear DAG):
```python
graph.add(NodeSpec("generate", generate_slogan))
graph.add(NodeSpec("refine_1", refine_slogan, depends_on=["generate"]))
graph.add(NodeSpec("refine_2", refine_slogan, depends_on=["refine_1"]))
graph.add(NodeSpec("evaluate", evaluate_slogan, depends_on=["refine_2"]))
```

## Verdict: DESIGN DIFFERENCE (Not a Gap)

This is the same pattern as Project 5 - hexDAG intentionally doesn't support cycles.

### The Core Problem

```
LangGraph (cycles):
┌──────────┐     ┌──────────┐     ┌─────────────┐
│ Generate │ --> │  Refine  │ --> │ Good enough?│
└──────────┘     └──────────┘     └─────────────┘
                      ^                  │
                      │    NO            │ YES
                      └──────────────────┴────> Done!

hexDAG (DAG):
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Generate │ -> │ Refine 1 │ -> │ Refine 2 │ -> Done!
└──────────┘    └──────────┘    └──────────┘
                                 (always 2, even if
                                  first was perfect)
```

| Scenario | LangGraph | hexDAG |
|----------|-----------|--------|
| Perfect on 1st try | Stops after 1 | Does 2 more anyway (waste) |
| Needs 5 refinements | Keeps going | Stops at 2 (not good enough) |
| Needs 100 refinements | Keeps going | Stops at 2 (fails) |

### Why This Can't Be "Fixed" (By Design)

hexDAG is a **DAG** - Directed **Acyclic** Graph. The "Acyclic" part means:

```
This is IMPOSSIBLE in a true DAG:

    A --> B --> C
    ^           |
    |           |
    └───────────┘  (cycle back - FORBIDDEN)
```

hexDAG **chose** to be acyclic for safety/predictability. True cycles will never be allowed - this is a design choice, not a technical limitation.

### Current Workarounds

1. **Pre-define iterations** - Add refine_1, refine_2, refine_3 nodes
2. **Dynamic DAG building** - Build graph with N refine nodes at runtime
3. **Loop in Python** - Call refine function N times manually

### Trade-offs

| Cycles (LangGraph) | DAG (hexDAG) |
|--------------------|--------------|
| Flexible iterations | Fixed iterations |
| Can loop forever | Always completes |
| Harder to debug | Predictable flow |
| Dynamic stopping | Pre-planned steps |

**When DAG is better:**
- Production systems (guaranteed completion)
- Testing (predictable behavior)
- Cost control (known LLM calls)

**When Cycles are better:**
- Unknown iteration count
- "Keep trying until good enough"
- Research/experimentation

## How to Run

```bash
cd framework-tests/project9-marketing-generator
..\..\.venv\Scripts\python.exe run_marketing.py
```

Expected output:
```
============================================================
hexDAG Marketing Slogan Generator Demo
============================================================

[Product 1] a new coffee shop called 'Morning Brew'
--------------------------------------------------
  [GENERATE] Creating initial slogan for: a new coffee shop...
  [GENERATE] Initial: "Wake Up to Morning Brew"
  [REFINE 1] Improving: "Wake Up to Morning Brew"
  [REFINE 1] Result: "Morning Brew: Start Your Day Right"
  [REFINE 2] Improving: "Morning Brew: Start Your Day Right"
  [REFINE 2] Result: "Brew Your Best Morning"
  [EVALUATE] Analyzing final slogan...

FINAL SLOGAN:
  "Brew Your Best Morning"

EVALUATION:
Rating: 8/10
Strengths: Short, memorable, action-oriented...
```

## Test Results
[To be filled after testing]

## Observations
[To be filled after testing]
