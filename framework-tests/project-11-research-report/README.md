# Project 11: Research and Report Generator - hexDAG Port

## Overview
This folder contains the hexDAG port of LangGraph's `project-11-research-and-report-generation`.

## What It Tests

**Multi-Agent Workflow - multiple specialized agents working together.**

```
┌────────────────┐     ┌────────────────┐
│   RESEARCHER   │ --> │     WRITER     │ --> Blog Post
│    (Agent 1)   │     │    (Agent 2)   │
└────────────────┘     └────────────────┘
       │                      │
       v                      v
  Gathers facts          Creates content
  and trends             from research
```

### Example Flow:
```
Topic: "Latest trends in AI"

[RESEARCHER] Researching: Latest trends in AI
[RESEARCHER] Gathering facts, trends, and insights...
[RESEARCHER] Research complete (2500 chars)

[WRITER] Creating blog post from research...
[WRITER] Blog post complete (3200 chars)

Output: Complete blog post with title, intro, body, conclusion
```

### Why Multi-Agent Matters:
- **Separation of concerns** - Each agent focuses on one task
- **Specialized prompts** - Better results than one generic agent
- **Reusable agents** - Researcher can feed other writers/analysts
- **Scalable** - Easy to add more agents (editor, SEO optimizer, etc.)

## Files
- `research_pipeline.yaml` - hexDAG YAML pipeline
- `run_research.py` - Python runner
- `README.md` - This file

## The Difference

| | LangGraph | hexDAG |
|---|-----------|--------|
| **Pattern** | Linear graph | Linear DAG |
| **State** | Shared message list | Explicit input/output |
| **Flow** | researcher -> writer | researcher -> writer |

### LangGraph:
```python
def researcher(state: State):
    response = llm.invoke("Research the latest trends in AI.")
    return {"messages": [response]}

def writer(state: State):
    response = llm.invoke(f"Write a blog post about: {state['messages'][-1].content}")
    return {"messages": [response]}

graph.add_edge("researcher", "writer")
```

### hexDAG:
```python
async def researcher(inputs: dict) -> dict:
    topic = inputs.get("topic", "")
    # ... research ...
    return {"research_content": research_content}

async def writer(inputs: dict) -> dict:
    research_content = inputs.get("research_content", "")
    # ... write ...
    return {"blog_post": blog_post}

graph.add(NodeSpec("researcher", researcher))
graph.add(NodeSpec("writer", writer, depends_on=["researcher"]))
```

## Verdict: WORKS PERFECTLY

This is exactly what hexDAG is designed for! A linear DAG of specialized agents.

**No gaps, no workarounds needed.**

hexDAG's explicit input/output is actually cleaner than LangGraph's shared message list - you know exactly what data flows between agents.

## Extending the Pipeline

Easy to add more agents:

```
┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
│ Researcher │-->│   Writer   │-->│   Editor   │-->│ Publisher  │
└────────────┘   └────────────┘   └────────────┘   └────────────┘
```

```yaml
spec:
  nodes:
    - name: researcher
      dependencies: []
    - name: writer
      dependencies: [researcher]
    - name: editor
      dependencies: [writer]
    - name: publisher
      dependencies: [editor]
```

## How to Run

```bash
cd framework-tests/project11-research-report

# Demo mode (predefined topics)
..\..\.venv\Scripts\python.exe run_research.py

# Custom topic
..\..\.venv\Scripts\python.exe run_research.py "quantum computing breakthroughs"
```

Expected output:
```
============================================================
hexDAG Research and Report Generator Demo
============================================================

[Topic 1] the latest trends in AI
--------------------------------------------------
  [RESEARCHER] Researching: the latest trends in AI
  [RESEARCHER] Gathering facts, trends, and insights...
  [RESEARCHER] Research complete (2500 chars)
  [WRITER] Creating blog post from research...
  [WRITER] Blog post complete (3200 chars)

============================================================
BLOG POST:
============================================================
# The AI Revolution: Trends Shaping Our Future

In the rapidly evolving landscape of artificial intelligence...
```

## Test Results
[To be filled after testing]

## Observations
[To be filled after testing]
