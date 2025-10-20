# AI Development Workflow for hexDAG

This guide outlines the optimal way to use AI assistants (Cursor, Claude Code) in the hexDAG project. By adhering to these practices, we ensure code quality, architectural integrity, and compliance with the project's strict standards (Python 3.12+, Ruff, Pydantic).

---

## 1. AI Context Priority

All AI tools are configured to prioritize project standards through three key files:

| File | Purpose | What It Contains |
|------|---------|------------------|
| `.cursorrules` | Auto-enforcement rules | Terse rules Cursor applies automatically |
| `CLAUDE.md` | Project context | Architecture, commands, development patterns |
| `AI_WORKFLOW.md` | Usage guide | How to effectively prompt AI tools |

hexDAG is an **enterprise-ready AI agent orchestration framework** that transforms complex AI workflows into deterministic, testable systems through declarative YAML configurations and DAG-based orchestration.

AI tools automatically enforce:

1. **Style**: Line length 100, Ruff formatting (from `.cursorrules` and `pyproject.toml`)
2. **Typing**: Modern Python 3.12+ syntax (`list[str]`, `X | Y`) and full type hinting (MyPy/Pyright compliance)
3. **Architecture**: Hexagonal principles (separation of Domain/Application from Adapters)

⚠️ **NOTE**: AI-generated code should still be reviewed by the developer for logic and architectural placement before being committed.

---

## 2. Recommended Usage Patterns

### A. Implementing New Features (`feat/*` or `experiment/*`)

When asking the AI to generate a new component, **always specify the architectural layer and required Pydantic validation.**

| Goal | Prompt Example | Why it Works |
|------|----------------|--------------|
| **New Core Node** | "Create a new `ConditionalNode` in `hexai/core/application/nodes/`. It must accept a Pydantic `ConditionalSpec` and be **async-first**." | Directs AI to correct folder and enforces core pillars: **Pydantic** and **Async-First** |
| **New Adapter** | "Implement a new LLM adapter that fulfills the `LLM` Protocol interface in `hexai/core/ports/llm.py`. Place it in `hexai/adapters/`. The adapter should implement the `async def aresponse(self, messages: MessageList) -> str | None` method." | AI automatically retrieves `LLM` Protocol definition and generates required async method with proper type hints |
| **YAML Pipeline** | "Create a YAML pipeline for a research workflow with an agent node that uses tools, followed by an LLM summarizer." | AI uses YAML structure rules from `.cursorrules` |

### B. Refactoring and Fixing Bugs (`refactor/*` or `fix/*`)

AI is highly effective for enforcing type correctness and modern syntax.

| Goal | Prompt Example | Why it Works |
|------|----------------|--------------|
| **Type Modernization** | "Refactor all `typing.Optional` and `typing.Dict` in this file to use modern Python 3.12+ syntax." | Enforces `.cursorrules` instantly, saving manual work with Pyupgrade |
| **Complexity Reduction** | "Refactor function `process_dag` to reduce its Radon code complexity score below B." | Directs AI to focus on `[tool.radon]` requirement from `pyproject.toml` |
| **Async Migration** | "Convert this function to async and use `asyncio.gather()` for parallel execution." | Enforces async-first architecture |

### C. Writing Tests (`test/*`)

| Goal | Prompt Example | Why it Works |
|------|----------------|--------------|
| **Unit Tests** | "Write pytest tests for `DirectedGraph.topological_sort()` covering: happy path, cycles, missing dependencies." | AI follows Arrange-Act-Assert pattern from `.cursorrules` |
| **Async Tests** | "Write async integration tests for `Orchestrator.execute()` with mocked LLM adapter." | AI uses `@pytest.mark.asyncio` and mock patterns |

---

## 3. Using Extended Context (`.claude/` Directory)

For complex tasks requiring deep architectural understanding, use the **`@/`** reference feature in Cursor (or similar context inclusion in Claude Code):

### Strategic Context Files

Create these files in `.claude/` for specialized guidance:

```
.claude/
├── architecture_detail.md    # Deep dive into hexagonal architecture
├── yaml_guide.md            # YAML pipeline syntax and examples
├── node_types.md            # All node types and their schemas
└── event_system.md          # Event emission patterns
```

### Usage Examples

- **Architecture Review**: "Review this module structure against `@.claude/architecture_detail.md` principles."
- **YAML Syntax**: "Build a pipeline using the template in `@.claude/yaml_guide.md`."
- **Node Implementation**: "Implement a new node type following patterns in `@.claude/node_types.md`."

By making these documents accessible through directory references, the AI avoids generating non-compliant code based only on external knowledge.

---

## 4. Effective Prompting Strategies

### ✅ Good Prompts (Specific, Context-Rich)

```
"Create an async LLM node in hexai/core/application/nodes/llm_node.py that:
1. Accepts a Pydantic LLMNodeSpec with prompt_template field
2. Uses the LLM Protocol interface (hexai/core/ports/llm.py) for model calls
3. Implements async def aresponse(messages: MessageList) -> str | None
4. Emits NodeStarted and NodeCompleted events
5. Includes NumPy-style docstrings
6. Has corresponding tests in tests/hexai/core/application/nodes/
7. Follows the pattern of existing nodes in the same directory"
```

### ❌ Poor Prompts (Vague, Missing Context)

```
"Make an LLM node"
```

### Prompt Template

Use this structure for complex requests:

```
"Create [component type] in [exact file path] that:
1. [Architectural requirement]
2. [Validation requirement]
3. [Interface/protocol requirement]
4. [Documentation requirement]
5. [Testing requirement]

Follow the patterns in [reference file/module]."
```

---

## 5. AI-Assisted Code Review

### Pre-Commit Checklist

Before committing AI-generated code, ask AI to review against standards:

```
"Review this code for:
1. Modern Python 3.12+ type hints (no List, Dict, Optional)
2. Pydantic validation on all data
3. Async-first patterns for I/O
4. NumPy-style docstrings
5. Comprehensive error handling
6. Event emission for observability
7. Corresponding test coverage"
```

### Quality Gate Commands

The developer **must** run these before committing:

```bash
# 1. Format and lint
uv run ruff format hexai/
uv run ruff check hexai/ --fix

# 2. Type check
uv run mypy hexai/
uv run pyright hexai/

# 3. Run tests
uv run pytest

# 4. All pre-commit hooks
uv run pre-commit run --all-files
```

---

## 6. Common AI Pitfalls and Solutions

| Pitfall | Solution |
|---------|----------|
| AI uses legacy type hints (`List`, `Dict`) | Explicitly request: "Use Python 3.12+ built-in generics" |
| AI creates synchronous I/O functions | Request: "Make this async and use asyncio.gather()" |
| AI forgets Pydantic validation | Request: "Add Pydantic Field() validation with constraints" |
| AI generates placeholder implementations | Request: "Provide complete, working implementation" |
| AI omits docstrings | Request: "Add NumPy-style docstrings with Parameters, Returns, Raises" |
| AI doesn't emit events | Request: "Add event emission for NodeStarted/NodeCompleted" |

---

## 7. Workflow Examples

### Example 1: New Feature End-to-End

```bash
# 1. Create feature branch
git checkout -b feat/conditional-node-TASK-123

# 2. Prompt AI
"Create ConditionalNode in hexai/core/application/nodes/conditional_node.py 
following the pattern of existing nodes. Include Pydantic validation, 
async execution, event emission, and tests."

# 3. Review AI output
- Check type hints are modern (list[str], not List[str])
- Verify Pydantic models have Field() validation
- Ensure async/await is used correctly
- Confirm events are emitted

# 4. Run quality checks
uv run pre-commit run --all-files

# 5. Test
uv run pytest tests/hexai/core/application/nodes/test_conditional_node.py

# 6. Commit
git add hexai/core/application/nodes/conditional_node.py tests/...
git commit -m "feat: add ConditionalNode for branching logic"
```

### Example 2: Bug Fix with AI

```bash
# 1. Create fix branch
git checkout -b fix/cycle-detection-TASK-456

# 2. Prompt AI
"The DirectedGraph.detect_cycles() method fails when there are 
self-referencing nodes. Fix this edge case and add tests."

# 3. Review fix
- Verify logic correctness
- Check test coverage for edge case
- Ensure error messages are clear

# 4. Run tests
uv run pytest tests/hexai/core/domain/test_directed_graph.py -v

# 5. Commit
git commit -m "fix: handle self-referencing nodes in cycle detection"
```

---

## 8. Advanced: Teaching AI Project Patterns

When AI generates code that doesn't match project patterns, create reusable examples:

### Pattern Document Template

Create `.claude/patterns/[pattern_name].md`:

```markdown
# Pattern: Async Node Implementation

## Structure
\`\`\`python
from typing import Any
from pydantic import BaseModel, Field
from hexai.core.domain.node import NodeSpec

class MyNodeSpec(BaseModel):
    """Specification for MyNode."""
    param: str = Field(..., min_length=1)

async def my_node(spec: MyNodeSpec, context: dict[str, Any]) -> Any:
    """Execute node logic."""
    # Emit start event
    await emit_event(NodeStarted(node_id=spec.id))
    
    try:
        # Node logic here
        result = await process(spec.param)
        
        # Emit completion event
        await emit_event(NodeCompleted(node_id=spec.id, result=result))
        return result
    except Exception as e:
        # Emit failure event
        await emit_event(NodeFailed(node_id=spec.id, error=str(e)))
        raise
\`\`\`

## Usage
Reference this pattern when implementing new nodes:
"Implement [NodeType] following the pattern in @.claude/patterns/async_node.md"
```

---

## 9. Measuring AI Effectiveness

Track these metrics to optimize AI usage:

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Pre-commit pass rate** | >90% | `uv run pre-commit run --all-files` succeeds first time |
| **Type hint compliance** | 100% | No mypy/pyright errors on AI-generated code |
| **Test coverage** | >80% | AI generates tests alongside features |
| **Code review cycles** | <2 | AI code requires minimal human revision |

---

## 10. Quick Reference: AI Commands

### Code Generation
```
"Create [component] in [path] with [requirements]"
"Implement [interface] in [adapter]"
"Add [validation] to [model]"
```

### Refactoring
```
"Modernize type hints in [file]"
"Convert [function] to async"
"Reduce complexity in [function]"
```

### Testing
```
"Write tests for [component] covering [scenarios]"
"Add integration tests for [workflow]"
"Mock [dependency] in tests"
```

### Documentation
```
"Add NumPy docstrings to [module]"
"Create architecture diagram for [system]"
"Document [API] usage"
```

---

## Summary: AI-First Development Checklist

- [ ] Branch name follows convention
- [ ] AI prompt includes architectural context
- [ ] Generated code uses Python 3.12+ type hints
- [ ] All data validated with Pydantic
- [ ] I/O operations are async
- [ ] Events emitted for observability
- [ ] NumPy docstrings present
- [ ] Tests written and passing
- [ ] Pre-commit hooks pass
- [ ] Code reviewed by human

**Remember**: AI is a powerful assistant, but the developer remains responsible for architectural decisions, logic correctness, and code quality. Use AI to enforce standards and accelerate implementation, not to replace critical thinking.