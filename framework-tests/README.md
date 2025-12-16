# hexDAG Framework Tests

## Overview

This folder contains ports of 22 LangGraph tutorial projects to hexDAG, providing a rigorous evaluation of hexDAG's capabilities compared to LangGraph.

**Purpose:** Identify gaps, design differences, and strengths of hexDAG for real-world AI workflows.

## Executive Summary

| Category | Count | Percentage |
|----------|-------|------------|
| **Works Perfectly** | 15 | 68% |
| **Design Differences** | 5 | 23% |
| **Major Gaps** | 2 | 9% |

**Overall Score: 8/10** - Production-ready for ~70% of use cases.

---

## Complete Project Verdicts

### ✅ WORKS PERFECTLY (15 Projects)

| # | Project | Pattern | Notes |
|---|---------|---------|-------|
| 1 | Basic Chatbot | Simple LLM | Perfect fit |
| 2 | Agent with Search | Tool + LLM | Uses DuckDuckGo (free) vs Tavily |
| 6 | Map-Reduce | Parallel processing | Excellent DAG use case |
| 8 | Finance Assistant | Search + Analysis | Manual tool integration |
| 10 | HR Policy Helper | RAG | Perfect fit |
| 11 | Research Report | Multi-agent sequential | Perfect fit |
| 12 | Email Automation | Parse + Respond | Perfect fit |
| 13 | Candidate Assessment | Linear evaluation | Perfect fit |
| 14 | Budget Planner | Data + Analysis | Perfect fit |
| 15 | Resume Matcher | Document comparison | Perfect fit |
| 16 | RAG Chatbot | Retrieval + Generation | Perfect fit |
| 17 | Log Analyzer | Parse + Summarize | Perfect fit |
| 18 | Web Research | Search + Report | Perfect fit |
| 19 | Shell Command | Execute + Explain | Cross-platform safe |
| 20 | Multi-Domain Router | Classify + Route | Perfect fit |
| 21 | Invoice Reader | PDF Extract + Validate | JSON vs Pydantic |
| 22 | Blog Writer | Multi-agent pipeline | Perfect fit |

**Key Insight:** All linear, deterministic workflows are handled excellently. hexDAG's explicit input/output model is often superior to LangGraph's shared state approach.

---

### ⚠️ DESIGN DIFFERENCES (5 Projects)

These are **intentional architectural choices**, not bugs.

#### Project 4: Shared State Management
| LangGraph | hexDAG |
|-----------|--------|
| `TypedDict` with `add_messages` reducer | Explicit dict passing between nodes |
| Implicit state mutations | No hidden mutations |
| Less boilerplate | More explicit, easier to debug |

**Verdict:** hexDAG's approach is arguably **better** for production - clearer data flow, easier testing.

---

#### Project 5: Conditional Looping
| LangGraph | hexDAG |
|-----------|--------|
| Graph cycles allowed | DAG = no cycles by design |
| `add_conditional_edges` with loop-back | External loop or LoopNode |
| Dynamic termination | Predictable execution |

```python
# LangGraph - cycle in graph
graph.add_conditional_edges("router", check, {True: "end", False: "chatbot"})

# hexDAG - external loop
while not done:
    result = await orchestrator.run(graph, inputs)
```

**Verdict:** hexDAG chose **predictability over flexibility**. DAGs guarantee termination - no infinite loops possible.

---

#### Project 9: Iterative Refinement
| LangGraph | hexDAG |
|-----------|--------|
| "Refine until quality is good" | Fixed refinement count |
| Dynamic iteration | refine_1 → refine_2 → done |
| Unknown LLM calls | Known cost upfront |

**Trade-off:**
- LangGraph: More flexible, but unpredictable cost
- hexDAG: Predictable, but may under/over-refine

---

### ❌ MAJOR GAPS (2 Projects)

#### Project 3: Human-in-the-Loop - CRITICAL GAP

**LangGraph Native Support:**
```python
from langgraph.types import interrupt, Command

def approve_node(state):
    res = interrupt({"draft": state["draft"]})  # PAUSES HERE
    return {"approved": res["data"]}

# Resume later:
graph.stream(Command(resume={"data": "approved"}), config)
```

**hexDAG Limitation:**
- ❌ No `interrupt()` primitive
- ❌ No `Command(resume=...)` mechanism
- ❌ Cannot pause DAG mid-execution

**Workaround Required:**
```
# Instead of one DAG with pause:
Search → Draft → [PAUSE] → Finalize

# Must split into two DAGs:
DAG 1: Search → Draft → Save to DB
[Human reviews externally]
DAG 2: Load from DB → Finalize
```

**Impact:** Cannot easily build:
- Content approval workflows
- Financial transaction authorization
- Healthcare decision validation
- Any compliance-heavy process

**Severity: CRITICAL** - Requires external orchestration for human approval flows.

---

#### Project 7: Caching/Checkpointing - PARTIAL GAP

**LangGraph Native Support:**
```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")
app = graph.compile(checkpointer=memory)
```

**hexDAG Limitation:**
- ❌ No built-in checkpointer
- ❌ No automatic state persistence
- ❌ Must implement caching manually

**Manual Implementation Required:**
```python
class SimpleCache:
    def __init__(self):
        self._cache = {}

    def get(self, key):
        return self._cache.get(key)

    def set(self, key, value):
        self._cache[key] = value
```

**Severity: MODERATE** - Solvable with custom code, but requires effort.

---

## Capability Comparison

| Capability | hexDAG | LangGraph |
|------------|--------|-----------|
| Linear Pipelines | **10/10** | 10/10 |
| Multi-Agent Workflows | **10/10** | 10/10 |
| RAG/Retrieval | **10/10** | 10/10 |
| Routing/Classification | **10/10** | 10/10 |
| Parallel Execution | **10/10** | 8/10 |
| Predictability | **10/10** | 6/10 |
| Human-in-the-Loop | **2/10** | 10/10 |
| Dynamic Looping | **5/10** | 10/10 |
| Built-in Caching | **4/10** | 9/10 |
| Testability | **10/10** | 7/10 |
| Production Safety | **10/10** | 7/10 |

---

## When to Use Each Framework

### Use hexDAG When:
- ✅ Linear pipelines (A → B → C → D)
- ✅ Deterministic workflows with known steps
- ✅ Production systems requiring guaranteed termination
- ✅ Multi-agent sequential processing
- ✅ RAG workflows (retrieve → generate)
- ✅ Cost sensitivity (known LLM call count)
- ✅ Enterprise deployments (predictability matters)

### Use LangGraph When:
- ✅ Human-in-the-loop approval workflows
- ✅ Dynamic looping ("keep trying until good")
- ✅ Complex shared state management
- ✅ Built-in checkpointing needed
- ✅ Flexible runtime control flow

---

## hexDAG Strengths

### 1. Predictability
- DAGs guarantee acyclic execution
- No infinite loops possible
- Always terminates in finite steps
- Cost is known upfront

### 2. Testability
- Each node is independently testable
- Mock inputs/outputs easily
- No global state to mock
- Deterministic results

### 3. Observability
- Explicit data flow is traceable
- Clear error boundaries
- Natural audit trail

### 4. Production Safety
- Guaranteed termination
- No recursion limits needed
- Clear resource consumption

---

## Recommended hexDAG Improvements

### Priority 1: Critical
1. **Add interrupt/resume support** - Essential for human-in-the-loop
2. **Add caching framework** - Built-in cache port with backends

### Priority 2: Nice-to-Have
1. **Dynamic DAG building** - Build nodes at runtime
2. **LoopNode improvements** - Better docs and options
3. **Conditional node type** - Simpler routing

### Priority 3: Documentation
1. **Migration guide from LangGraph**
2. **Design philosophy document** (why DAGs)
3. **Patterns catalog** (22 examples)

---

## Project Structure

```
framework-tests/
├── README.md                    # This file
├── project1-chatbot/            # Basic LLM chat
├── project2-agent-with-search/  # Tool integration
├── project3-human-in-the-loop/  # ❌ MAJOR GAP
├── project4-stateful-agent/     # ⚠️ Design difference
├── project5-looping/            # ⚠️ Design difference (cycles)
├── project6-map-reduce/         # Parallel processing
├── project7-caching/            # ❌ Partial gap
├── project8-finance-assistant/  # Search + analysis
├── project9-marketing-generator/# ⚠️ Design difference (iteration)
├── project10-hr-policy-helper/  # RAG pattern
├── project11-research-report/   # Multi-agent
├── project12-email-automation/  # Parse + respond
├── project13-candidate-assessment/
├── project14-budget-planner/
├── project15-resume-matcher/
├── project16-rag-chatbot/
├── project17-log-analyzer/
├── project18-web-research-report/
├── project19-shell-command/
├── project20-multi-domain-router/
├── project21-invoice-reader/
└── project22-blog-writer/
```

---

## Running the Tests

Each project can be run independently:

```bash
cd framework-tests/project<N>-<name>
..\..\venv\Scripts\python.exe run_<name>.py
```

**Requirements:**
- Python 3.10+
- Google Gemini API key in `.env`
- Project-specific dependencies (see each README)

---

## Conclusion

**hexDAG is production-ready for 68% of common AI workflow patterns.**

It excels at linear pipelines, multi-agent systems, RAG, and routing - covering most enterprise use cases. The DAG architecture provides superior predictability and testability compared to LangGraph's more flexible but less deterministic approach.

**The main gaps are:**
1. Human-in-the-loop (requires workaround)
2. Built-in caching (requires manual implementation)

For workflows requiring human approval or dynamic iteration, either use the documented workarounds or consider LangGraph for those specific cases.

---

*Last Updated: December 2024*
*Tests conducted against hexDAG 0.5.x and LangGraph tutorials*
