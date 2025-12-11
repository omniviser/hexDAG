# Project 6: Map-Reduce (Iterative Processing) - hexDAG Port

## Overview
This folder contains the hexDAG port of LangGraph's `project-06-iterative-processing-workflow`.

## What It Does

**Map-Reduce pattern:**
1. **MAP**: Process a list of items (e.g., ["apple", "banana", "cherry"])
2. **REDUCE**: Aggregate all results into a summary

```
Input: ["apple", "banana", "cherry"]
         ↓
    [MAP PHASE]
    ├── Process "apple" → "Apple is a fruit..."
    ├── Process "banana" → "Banana is yellow..."
    └── Process "cherry" → "Cherry is small..."
         ↓
    [REDUCE PHASE]
    Aggregate: "3 items processed"
         ↓
Output: Summary of all results
```

## Files
- `map_reduce_pipeline.yaml` - hexDAG YAML pipeline
- `run_map_reduce.py` - Python runner
- `README.md` - This file

## The Difference

| | LangGraph | hexDAG |
|---|-----------|--------|
| **Parallel processing** | Lambda in node | `asyncio.gather()` |
| **Graph structure** | process → aggregate | process → aggregate |
| **Complexity** | Similar | Similar |

### LangGraph:
```python
graph.add_node("process", lambda state: [process_item(i) for i in state["items"]])
graph.add_node("aggregate", aggregate_results)
graph.add_edge("process", "aggregate")
```

### hexDAG:
```python
graph.add(NodeSpec("process_items", process_items_node))
graph.add(NodeSpec("aggregate", aggregate_node).after("process_items"))
```

## Verdict: GOOD MATCH ✅

hexDAG handles Map-Reduce well:
- Clean DAG structure (process → aggregate)
- Parallel processing via `asyncio.gather()`
- No workarounds needed

## How to Run

```bash
cd framework-tests/project6-map-reduce
..\..\.venv\Scripts\python.exe run_map_reduce.py
```

Enter items or press Enter for default: `["apple", "banana", "cherry"]`

## Test Results
[To be filled after testing]

## Observations
[To be filled after testing]
