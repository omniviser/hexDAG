# Project 3: Human-in-the-Loop Chatbot - hexDAG Port

## Overview
This folder contains the hexDAG port of LangGraph's `project-03-human-in-the-loop-chatbot`.

**IMPORTANT:** This project reveals a **hexDAG GAP** - no native interrupt/resume support.

## Files
- `run_hitl.py` - Python runner with human approval workflow
- `README.md` - This file

## Original LangGraph Version
Located at: `reference_examples/langgraph-tutorials/project-03-human-in-the-loop-chatbot/main.py`

## Key Differences

| Aspect | LangGraph | hexDAG |
|--------|-----------|--------|
| Interrupt | `interrupt()` function | **NOT SUPPORTED** |
| Resume | `Command(resume=...)` | **NOT SUPPORTED** |
| Workaround | N/A | Split into multiple DAGs |
| Checkpointing | `MemorySaver` | Manual state passing |

## hexDAG GAP IDENTIFIED

### Human-in-the-Loop Pattern

**LangGraph approach:**
```python
from langgraph.types import interrupt, Command

def approve_node(state):
    res = interrupt({"draft": state["draft"]})  # Pauses here!
    return {"approved": res["data"]}

# Later, resume with:
cmd = Command(resume={"data": "yes"})
graph.stream(cmd, config)
```

**hexDAG workaround:**
```python
# Must split workflow into multiple phases
# Phase 1: Search + Draft (DAG 1)
# Phase 2: Human approval (plain Python)
# Phase 3: Finalize (DAG 2)
```

### Impact
- Cannot pause DAG execution mid-flow
- Must manually orchestrate multi-phase workflows
- State must be passed between phases explicitly

### Recommendation
hexDAG should consider adding:
1. `interrupt()` primitive for pausing execution
2. Checkpoint/resume mechanism
3. Or document the multi-phase workaround pattern

## How to Run

### Install dependency:
```bash
pip install duckduckgo-search
```

### hexDAG Version:
```bash
cd framework-tests/project3-human-in-the-loop
..\..\.venv\Scripts\python.exe run_hitl.py
```

## Test Workflow
1. Ask a question (e.g., "What is quantum computing?")
2. Agent searches web and drafts answer
3. **YOU** review and approve/reject/edit
4. Final answer is shown

## Test Results
[To be filled after testing]

## Observations
[To be filled after testing]

### Severity: MAJOR GAP
This is a significant limitation for production workflows that require human oversight.
