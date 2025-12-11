# Project 5: Conditional Edges - hexDAG Port

## Overview
This folder contains the hexDAG port of LangGraph's `project-05-multi-agent-collaboration`.

**Note:** Despite the folder name, this project demonstrates **conditional edges** (looping), not multi-agent collaboration.

## Files
- `conditional_pipeline.yaml` - hexDAG YAML pipeline definition
- `run_conditional.py` - Python runner with loop implementation
- `README.md` - This file

## Original LangGraph Version
Located at: `reference_examples/langgraph-tutorials/project-05-multi-agent-collaboration/main.py`

## What This Project Does

The chatbot loops until the message count exceeds a threshold (5 messages):

```
User: Hello!
Chatbot: Hi there!
User: Tell me more
Chatbot: Sure...
... (continues until 5+ messages)
[Threshold reached, exit loop]
```

## Key Architectural Difference: Cycles vs DAGs

### LangGraph Approach (Cycles Allowed)
```python
# LangGraph allows cycles in the graph
graph.add_conditional_edges(
    "chatbot",
    check_threshold,
    {True: "end", False: "chatbot"}  # Loops back to chatbot!
)
```
- Graph can have cycles
- Loop is part of the graph structure
- Execution continues until condition is True

### hexDAG Approach (DAG = No Cycles)
```python
# hexDAG = Directed ACYCLIC Graph (no cycles by definition)
# Loop must be implemented externally
while not check_threshold(message_count):
    result = await orchestrator.run(graph, inputs)
    # Update state and continue
```
- DAGs cannot have cycles
- Loop is external Python code
- Each iteration is a separate DAG execution

## Is This a Gap?

**Debatable.** This is a **design philosophy difference**:

| Aspect | LangGraph | hexDAG |
|--------|-----------|--------|
| Graph Type | StateGraph (allows cycles) | DirectedGraph (acyclic) |
| Loops | In-graph via conditional edges | External via Python/LoopNode |
| Philosophy | Graph defines all control flow | DAG for dependencies, Python for control |
| Predictability | Harder to reason about cycles | Clear execution order |
| Flexibility | More dynamic | More structured |

### hexDAG Alternative: LoopNode
hexDAG does have a `LoopNode` for iteration:
```yaml
- kind: loop_node
  metadata:
    name: chat_loop
  spec:
    max_iterations: 10
    condition: "message_count <= 5"
    body: [chatbot]
```

## How to Run

```bash
cd framework-tests/project5-conditional-edges
..\..\.venv\Scripts\python.exe run_conditional.py
```

Choose mode:
1. **Auto-run** - Simulates LangGraph behavior (automatic loop)
2. **Interactive** - You provide input each iteration

## Test Results
[To be filled after testing]

## Observations
[To be filled after testing]

### Classification: DESIGN DIFFERENCE (not a gap)
hexDAG's DAG-based approach is intentional for predictability and testability.
Loops can still be achieved via LoopNode or external iteration.
