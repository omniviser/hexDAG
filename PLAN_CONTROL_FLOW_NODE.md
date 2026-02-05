# Plan: Unified ControlFlowNode for hexDAG

## Summary

Replace both `LoopNode` and `ConditionalNode` with a single unified `ControlFlowNode` that supports all control flow patterns through a `mode` parameter:

- **5 modes**: `while`, `for-each`, `times`, `if-else`, `switch`
- **2 execution patterns per mode**:
  - **Inline body**: When `body` or `body_pipeline` specified → execute within the node
  - **Yield to downstream**: When no body → yield state to downstream nodes (routing/iteration)
- **4 body options** (when inline): function string, `!py` inline Python, inline nodes (sub-DAG), pipeline reference

## Design Decisions

| Decision | Choice |
|----------|--------|
| LoopNode + ConditionalNode | **Merge** into unified `ControlFlowNode` |
| Body execution pattern | **Unified** - function, `!py` inline Python, inline nodes, or pipeline reference |
| Backward compatibility | **Breaking change** - both deprecated in favor of ControlFlowNode |
| Inline vs Yield pattern | **Unified** - all modes support both; `body` present → inline execution; `body` absent → yield to downstream |

## ControlFlowNode Design

### Five Modes

| Mode | Description | Key Parameters |
|------|-------------|----------------|
| `while` | Condition-based loop | `condition`, `initial_state` |
| `for-each` | Collection iteration | `items`, `item_var`, `concurrency` |
| `times` | Fixed count iteration | `count`, `concurrency` |
| `if-else` | Single condition branch | `condition`, `else_body` |
| `switch` | Multi-branch conditions | `branches`, `else_body` |

### Unified Execution Pattern: Body vs Yield-to-Downstream

**All modes support BOTH patterns** based on whether `body`/`body_pipeline` is provided:

| Pattern | When | Behavior |
|---------|------|----------|
| **Inline Body** | `body` or `body_pipeline` specified | Execute body within the control flow node |
| **Yield to Downstream** | No `body` specified | Yield state/iteration data to downstream nodes |

This eliminates the need for separate `route` and `iterate` modes - they're just `switch` and loop modes without a body.

### YAML Examples

```yaml
# Mode 1: WHILE - condition-based loop (replaces LoopNode)
- kind: control_flow_node
  metadata:
    name: retry_loop
  spec:
    mode: while
    condition: "state.retry_count < 3 and not state.success"
    initial_state:
      retry_count: 0
      success: false
    body: "myapp.attempt_operation"
    state_update:
      retry_count: "state.retry_count + 1"
      success: "$body.success"
    collect: last

# Mode 2: FOR-EACH - collection iteration (new capability)
- kind: control_flow_node
  metadata:
    name: process_threads
  spec:
    mode: for-each
    items: "$input.threads"
    item_var: "thread"
    concurrency: 5
    body:
      - kind: port_call_node
        spec:
          port: database
          method: get_context
          input_mapping:
            thread_id: "$item.id"
      - kind: llm_node
        spec:
          prompt_template: "Analyze: {{$item.content}}"
    collect: list

# Mode 3: TIMES - fixed count iteration
- kind: control_flow_node
  metadata:
    name: generate_samples
  spec:
    mode: times
    count: 5
    concurrency: 3
    body_pipeline: "./generate_sample.yaml"
    collect: list

# Mode 4: IF-ELSE - single condition (simple conditional)
- kind: control_flow_node
  metadata:
    name: handle_urgent
  spec:
    mode: if-else
    condition: "priority == 'urgent'"
    body:
      - kind: llm_node
        spec:
          prompt_template: "URGENT: {{input}}"
    else_body:
      - kind: function_node
        spec:
          fn: "myapp.standard_process"

# Mode 5: SWITCH - multi-branch conditions (executes body inline)
- kind: control_flow_node
  metadata:
    name: action_handler
  spec:
    mode: switch
    branches:
      - condition: "action == 'ACCEPT'"
        body:
          - kind: function_node
            spec:
              fn: "myapp.approve"
      - condition: "action == 'REJECT'"
        body:
          - kind: function_node
            spec:
              fn: "myapp.reject"
      - condition: "confidence < 0.5"
        body:
          - kind: llm_node
            spec:
              prompt_template: "Low confidence review: {{input}}"
    else_body:
      - kind: function_node
        spec:
          fn: "myapp.manual_review"

# ============================================================
# YIELD-TO-DOWNSTREAM PATTERN (no body specified)
# ============================================================
# Same modes, but WITHOUT body - yields to external downstream nodes

# FOR-EACH without body - yields each item to downstream nodes
- kind: control_flow_node
  metadata:
    name: batch_iterator
  spec:
    mode: for-each
    items: "$input.batches"
    item_var: "batch"
    # No body! Yields to downstream nodes
# Output per iteration: {"batch": {...}, "index": 0, "total": 5, "is_last": false}

# Downstream nodes process each iteration (external to the control_flow_node):
- kind: port_call_node
  metadata:
    name: fetch_context
  spec:
    port: database
    method: get_batch_context
    input_mapping:
      batch_id: "batch_iterator.batch.id"
  dependencies: [batch_iterator]

- kind: llm_node
  metadata:
    name: analyze_batch
  spec:
    prompt_template: |
      Batch {{batch_iterator.index + 1}} of {{batch_iterator.total}}:
      Context: {{fetch_context.result}}
      Data: {{batch_iterator.batch}}
  dependencies: [batch_iterator, fetch_context]

# WHILE without body - condition-based iteration yielding to downstream
- kind: control_flow_node
  metadata:
    name: retry_iterator
  spec:
    mode: while
    condition: "state.attempts < 3 and not state.success"
    initial_state:
      attempts: 0
      success: false
    # No body! state_update happens AFTER downstream nodes complete
    state_update:
      attempts: "state.attempts + 1"
      success: "downstream_result.success"  # References downstream output

# SWITCH without body - routing pattern (replaces ConditionalNode)
- kind: control_flow_node
  metadata:
    name: path_router
  spec:
    mode: switch
    branches:
      - condition: "status == 'urgent'"
        action: "urgent_path"    # action instead of body
      - condition: "status == 'normal'"
        action: "normal_path"
    else_action: "default_path"
# Output: {"result": "urgent_path", "metadata": {...}}
# Downstream nodes use: when: "path_router.result == 'urgent_path'"
```

### Yield-to-Downstream Pattern (No Body)

When `body` or `body_pipeline` is **not specified**, all modes yield control to downstream nodes. This enables:

1. **Separate concerns** - Control flow logic is decoupled from processing
2. **Multiple downstream pipelines** - Different nodes can process the same iteration/branch
3. **Complex workflows** - Full DAG capabilities within each iteration/branch

**How it works (loop modes without body):**

```
┌─────────────────────────────────────────────────────────────────────┐
│  for-each/while/times node (no body)                                │
│  (controls iteration, yields state)                                  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ yields: {$item, $index, $total, state}
                                ▼
        ┌───────────────────────┴────────────────────────┐
        │                                                │
        ▼                                                ▼
┌───────────────────┐                        ┌───────────────────┐
│  fetch_data       │                        │  validate_item    │
│  (port_call_node) │                        │  (function_node)  │
└────────┬──────────┘                        └────────┬──────────┘
         │                                            │
         └─────────────────┬──────────────────────────┘
                           ▼
                ┌───────────────────┐
                │  process_item     │
                │  (llm_node)       │
                └────────┬──────────┘
                         │ returns result
                         ▼
         ┌───────────────────────────────────┐
         │  control flow node continues      │
         │  (updates state, checks condition)│
         └───────────────────────────────────┘
```

**State update from downstream outputs (while mode):**

```yaml
- kind: control_flow_node
  metadata:
    name: batch_processor
  spec:
    mode: while
    condition: "state.attempts < 3 and not state.success"
    initial_state:
      attempts: 0
      success: false
    # No body - yields to downstream
    # state_update references downstream node outputs
    state_update:
      attempts: "state.attempts + 1"
      success: "process_item.success"  # References downstream node
```

**How it works (switch mode without body = routing):**

```
┌─────────────────────────────────────────────────────────────────────┐
│  switch node (no body, has action instead)                          │
│  evaluates conditions, returns matched action label                  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ returns: {"result": "urgent_path"}
                                ▼
        ┌───────────────────────┴────────────────────────┐
        │                                                │
        ▼                                                ▼
┌───────────────────────────┐              ┌───────────────────────────┐
│  urgent_handler           │              │  normal_handler           │
│  when: "router.result ==  │              │  when: "router.result ==  │
│        'urgent_path'"     │              │        'normal_path'"     │
└───────────────────────────┘              └───────────────────────────┘
```

### Unified Body Execution Pattern

When `body` or `body_pipeline` **is specified**, all modes support these body specifications:

```yaml
# Option A: Function string (module path)
body: "myapp.process_item"

# Option B: Inline Python (full Python via !py tag)
body: !py |
  async def process(item, index, state, **ports):
      """Full Python function - executed once per iteration."""
      db = ports.get('database')
      if db:
          result = await db.aquery(item['id'])
      else:
          result = {"id": item['id'], "processed": True}
      return result

# Option C: Inline nodes (sub-DAG)
body:
  - kind: function_node
    metadata:
      name: "{{name}}_step1"
    spec:
      fn: "myapp.extract"
  - kind: llm_node
    metadata:
      name: "{{name}}_step2"
    spec:
      when: "step1.success"  # Can be skipped!
      prompt_template: "Process: {{$item}}"

# Option D: Pipeline reference (external file)
body_pipeline: "./pipelines/process_item.yaml"
```

### `!py` Tag Details

The `!py` YAML custom tag allows inline Python functions:

```yaml
# Function signature for !py body
async def process(
    item: Any,           # Current item (for-each) or iteration data
    index: int,          # Current iteration index
    state: dict,         # State dict (for while mode)
    **ports: Any         # Injected ports (database, llm, memory, etc.)
) -> Any:
    """Return value becomes the body result."""
    ...
```

**Security considerations:**
- `!py` executes arbitrary Python - use only with trusted YAML
- For untrusted input, use `body: "module.path"` or inline nodes instead
- The function runs in the same process as the orchestrator

**Example use cases:**
```yaml
# Complex transformation
body: !py |
  async def process(item, index, state, **ports):
      from decimal import Decimal
      rate = Decimal(str(item['rate']))
      discount = Decimal('0.10') if state.get('vip') else Decimal('0.05')
      return {"final_rate": float(rate * (1 - discount))}

# Database operations with error handling
body: !py |
  async def process(item, index, state, **ports):
      db = ports.get('database')
      try:
          result = await db.aexecute_query(
              "UPDATE items SET processed = true WHERE id = ?",
              [item['id']]
          )
          return {"success": True, "id": item['id']}
      except Exception as e:
          return {"success": False, "error": str(e)}
```

### Configuration Options

```yaml
spec:
  # Mode (required)
  mode: "while" | "for-each" | "times" | "if-else" | "switch"

  # --- WHILE mode ---
  condition: "state.attempts < 3"
  initial_state: {...}
  state_update: {...}
  max_iterations: 100

  # --- FOR-EACH mode ---
  items: "$input.collection"
  item_var: "item"
  index_var: "index"

  # --- TIMES mode ---
  count: 10

  # --- IF-ELSE mode ---
  condition: "status == 'active'"
  else_body: [...] | "..." | null

  # --- SWITCH mode (executes body) ---
  branches:
    - condition: "..."
      body: [...] | "..."
  else_body: [...] | "..." | null

  # --- Common to all modes ---
  # Body specification (exactly one required):
  body: "myapp.func"            # Option A: Module path string
  body: !py |                   # Option B: Inline Python function
    async def process(item, index, state, **ports):
        return item
  body:                         # Option C: Inline nodes (sub-DAG)
    - kind: expression_node
      spec: ...
  body_pipeline: "./file.yaml"  # Option D: Pipeline reference

  # Concurrency (for-each, times only)
  concurrency: 5

  # Result collection
  collect: "list" | "last" | "reduce" | "dict" | "first"
  key_field: "id"
  reducer: "myapp.reduce_fn"

  # Error handling
  error_handling: "fail_fast" | "continue" | "collect"
```

### Output Structure

All modes return a consistent structure:

```python
{
    "result": ...,              # The body output(s) based on collect mode
    "metadata": {
        "mode": "for-each",
        "iterations": 10,       # For loop modes
        "matched_branch": 2,    # For switch mode
        "condition_met": True,  # For if-else mode
        "successful": 9,
        "failed": 1,
        "stopped_by": "exhausted" | "condition" | "limit" | "else" | "no_match",
        "errors": [...],        # If error_handling != "fail_fast"
        "final_state": {...},   # For while mode
        "duration_ms": 1234.5
    }
}
```

### Execution Context

Each body execution receives:

```python
{
    # Current iteration info (loop modes)
    "$item": current_item,
    "$index": 0,
    "$total": 10,
    "$is_first": True,
    "$is_last": False,

    # State (while mode)
    "state": {...},

    # Branch info (switch mode)
    "$branch_index": 2,
    "$condition": "confidence < 0.5",

    # Parent context
    **dependency_outputs
}
```

### Node Skipping in Inline Bodies

When an inline node has a `when` condition that evaluates to false:

1. **NodeSkipped event** is emitted (existing hexDAG behavior)
2. **Node returns** `{"_skipped": True, "reason": "..."}`
3. **Downstream nodes** in the body that depend on the skipped node will receive the skip marker
4. **Body result** is the **last non-skipped result** from the inline nodes

```yaml
# Example: step2 depends on step1, step1 might be skipped
body:
  - kind: function_node
    metadata:
      name: step1
    spec:
      when: "status == 'active'"  # May be skipped!
      fn: "myapp.process"

  - kind: llm_node
    metadata:
      name: step2
    spec:
      # step2 receives {"_skipped": True} if step1 was skipped
      prompt_template: "Analyze: {{step1.result}}"

# If step1 is skipped:
#   - NodeSkipped event emitted for step1
#   - step2 receives step1 = {"_skipped": True, "reason": "..."}
#   - step2 can check for this and handle gracefully
#   - Body result = step2's output (last non-skipped)
```

**Skip-aware templates:**
```yaml
body:
  - kind: expression_node
    metadata:
      name: check_skip
    spec:
      expressions:
        # Handle potential skip from previous node
        data: "step1.result if not step1.get('_skipped') else 'default'"
```

## Implementation Architecture

### File Structure

```
hexdag/
├── builtin/
│   └── nodes/
│       ├── control_flow_node.py  # NEW: Unified ControlFlowNode
│       ├── loop_node.py          # DEPRECATE: Keep for backward compat
│       └── __init__.py           # Export ControlFlowNode
├── core/
│   └── orchestration/
│       └── body_executor.py      # NEW: Shared body execution logic
└── tests/
    └── hexdag/
        └── builtin/
            └── nodes/
                └── test_control_flow_node.py  # NEW: Comprehensive tests
```

### Core Components

#### 1. BodyExecutor (`hexdag/core/orchestration/body_executor.py`)

```python
class BodyExecutor:
    """Execute node body in function, inline-nodes, or pipeline mode."""

    async def execute(
        self,
        body: str | list[dict] | None,
        body_pipeline: str | None,
        input_data: dict,
        context: NodeExecutionContext,
        ports: dict[str, Any],
    ) -> Any:
        """Execute body and return result."""
        if body_pipeline:
            return await self._execute_pipeline(...)
        elif isinstance(body, list):
            return await self._execute_inline_nodes(...)
        elif isinstance(body, str):
            return await self._execute_function(...)
        else:
            raise ValueError("body or body_pipeline required")
```

#### 2. ControlFlowNode (`hexdag/builtin/nodes/control_flow_node.py`)

```python
class ControlFlowNode(BaseNodeFactory):
    """Unified control flow node supporting while, for-each, times, if-else, switch.

    All modes support two execution patterns:
    - Inline body: When body/body_pipeline specified → execute within node
    - Yield to downstream: When no body → yield state to dependent nodes
    """

    def __call__(
        self,
        name: str,
        mode: Literal["while", "for-each", "times", "if-else", "switch"],
        # Body specification (optional - if omitted, yields to downstream)
        body: str | list[dict] | Callable | None = None,  # Callable for !py
        body_pipeline: str | None = None,
        # Mode-specific params
        condition: str | None = None,          # while, if-else, switch branches
        items: str | None = None,              # for-each
        count: int | None = None,              # times
        branches: list[dict] | None = None,    # switch (each has condition + body OR action)
        else_body: str | list[dict] | None = None,   # if-else, switch (inline)
        else_action: str | None = None,        # switch without body (routing)
        # Loop state management
        initial_state: dict | None = None,     # while
        state_update: dict | None = None,      # while (can reference downstream outputs)
        max_iterations: int = 100,             # while safety limit
        # Concurrency (for-each, times only)
        concurrency: int = 1,
        # Result collection
        collect: str = "list",                 # list, last, first, dict, reduce
        key_field: str | None = None,          # for collect: dict
        reducer: str | None = None,            # for collect: reduce
        # Error handling
        error_handling: str = "fail_fast",     # fail_fast, continue, collect
        **kwargs,
    ) -> NodeSpec:
        ...
```

## Implementation Steps

### Step 1: Create BodyExecutor
**File:** `hexdag/core/orchestration/body_executor.py`

- Implement function resolution via `hexdag.core.resolver.resolve()`
- Implement `!py` inline Python compilation and execution
- Implement inline node building (use YamlPipelineBuilder patterns)
- Implement pipeline loading and sub-orchestration
- Add error handling and logging

```python
class BodyExecutor:
    """Execute node body in one of four modes."""

    async def execute(
        self,
        body: str | list[dict] | Callable | None,
        body_pipeline: str | None,
        input_data: dict,
        context: NodeExecutionContext,
        ports: dict[str, Any],
    ) -> Any:
        if body_pipeline:
            return await self._execute_pipeline(...)
        elif isinstance(body, list):
            return await self._execute_inline_nodes(...)
        elif callable(body):
            # !py compiled function
            return await self._execute_py_function(body, input_data, context, ports)
        elif isinstance(body, str):
            return await self._execute_function(...)
        else:
            raise ValueError("body or body_pipeline required")
```

### Step 1b: Add !py YAML Tag Handler
**File:** `hexdag/core/pipeline_builder/py_tag.py`

Create a YAML custom tag constructor for `!py`:

```python
import yaml

def py_constructor(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> Callable:
    """Compile !py tagged Python code into a callable function."""
    source_code = loader.construct_scalar(node)

    # Compile and extract the function
    namespace: dict[str, Any] = {}
    exec(compile(source_code, '<yaml-!py>', 'exec'), namespace)

    # Find the defined function (first callable in namespace)
    for name, obj in namespace.items():
        if callable(obj) and not name.startswith('_'):
            return obj

    raise ValueError("!py block must define a function")

# Register with YAML loader
yaml.SafeLoader.add_constructor('!py', py_constructor)
```

### Step 2: Create ControlFlowNode Base
**File:** `hexdag/builtin/nodes/control_flow_node.py`

- Base structure extending `BaseNodeFactory`
- Mode validation and dispatch
- `_yaml_schema` for documentation

### Step 3: Implement Loop Modes
**In:** `control_flow_node.py`

- `_execute_while_mode()` - condition-based with state
- `_execute_foreach_mode()` - collection iteration with parallelism
- `_execute_times_mode()` - fixed count iteration
- Use `asyncio.Semaphore` for concurrency (pattern from WaveExecutor)

### Step 4: Implement Conditional Modes
**In:** `control_flow_node.py`

- `_execute_if_else_mode()` - single condition with optional else
- `_execute_switch_mode()` - multi-branch with optional else
- Both execute bodies, not just return labels

### Step 5: Implement Collect Modes
**In:** `control_flow_node.py`

- `list` - all results in array
- `last` - only final result
- `first` - first successful result
- `dict` - keyed by key_field
- `reduce` - custom reducer function

### Step 6: Register Node
**File:** `hexdag/builtin/nodes/__init__.py`

```python
from .control_flow_node import ControlFlowNode

__all__ = [
    ...
    "ControlFlowNode",
]
```

### Step 7: Deprecate Old Nodes
**Files:** `loop_node.py`

```python
import warnings

class LoopNode(BaseNodeFactory):
    def __init__(self, ...):
        warnings.warn(
            "LoopNode is deprecated. Use ControlFlowNode with mode='while' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        ...

class ConditionalNode(BaseNodeFactory):
    def __init__(self, ...):
        warnings.warn(
            "ConditionalNode is deprecated. Use ControlFlowNode with mode='switch' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        ...
```

### Step 8: Add Tests
**File:** `tests/hexdag/builtin/nodes/test_control_flow_node.py`

Test cases for each mode with inline body:
- **while**: condition evaluation, state management, max_iterations
- **for-each**: collection iteration, concurrency, empty collection
- **times**: fixed count, concurrent execution
- **if-else**: true branch, false branch, no else_body
- **switch**: multiple branches, first match wins, else_body fallback

Test cases for each mode with yield-to-downstream (no body):
- **for-each/while/times**: yields iteration data, state_update from downstream outputs
- **switch**: returns action label, downstream nodes use `when` conditions

Test cases for body modes:
- Function string resolution
- `!py` inline Python execution (sync and async)
- Inline nodes execution (with skip handling)
- Pipeline reference loading

Test cases for collect modes and error handling.

Test cases for `!py` tag:
- Basic function compilation
- Async function support
- Access to ports
- Error handling in inline Python

### Step 9: Update Documentation
**File:** `docs/reference/nodes.md`

Document ControlFlowNode with examples for all modes.

## Migration Guide

### From LoopNode

```yaml
# Before
- kind: loop_node
  spec:
    while_condition: "state.iteration < 3"
    body: "myapp.process"
    initial_state: {iteration: 0}
    collect_mode: last

# After
- kind: control_flow_node
  spec:
    mode: while
    condition: "state.iteration < 3"
    body: "myapp.process"
    initial_state: {iteration: 0}
    collect: last
```

### From ConditionalNode

**Option A: Keep routing pattern (switch mode without body)**
```yaml
# Before
- kind: conditional_node
  spec:
    branches:
      - condition: "status == 'urgent'"
        action: "priority"
    else_action: "standard"

# After (same behavior - no body, returns label)
- kind: control_flow_node
  spec:
    mode: switch
    branches:
      - condition: "status == 'urgent'"
        action: "priority"      # action instead of body → routing
    else_action: "standard"
# Downstream nodes still use: when: "router.result == 'priority'"
```

**Option B: Inline execution (switch mode with body)**
```yaml
# Before (routing + separate handler nodes)
- kind: conditional_node
  metadata:
    name: router
  spec:
    branches:
      - condition: "status == 'urgent'"
        action: "priority"

- kind: function_node
  metadata:
    name: priority_handler
  spec:
    when: "router.result == 'priority'"
    fn: "myapp.priority_handler"
  dependencies: [router]

# After (all in one node)
- kind: control_flow_node
  spec:
    mode: switch
    branches:
      - condition: "status == 'urgent'"
        body: "myapp.priority_handler"
    else_body: "myapp.standard_handler"
```

### New Capability: Inline Bodies

```yaml
# Not possible before - now supported
- kind: control_flow_node
  spec:
    mode: for-each
    items: "$input.threads"
    concurrency: 5
    body:
      - kind: port_call_node
        spec:
          port: database
          method: get_thread_context
      - kind: llm_node
        spec:
          prompt_template: "Analyze: {{$item}}"
```

## Critical Files

### Create
| File | Purpose |
|------|---------|
| `hexdag/core/orchestration/body_executor.py` | Shared body execution logic |
| `hexdag/core/pipeline_builder/py_tag.py` | `!py` YAML custom tag handler |
| `hexdag/builtin/nodes/control_flow_node.py` | Unified ControlFlowNode |
| `tests/hexdag/builtin/nodes/test_control_flow_node.py` | Test suite |

### Modify
| File | Purpose |
|------|---------|
| `hexdag/builtin/nodes/__init__.py` | Export ControlFlowNode |
| `hexdag/builtin/nodes/loop_node.py` | Add deprecation warnings |
| `docs/reference/nodes.md` | Add documentation |

### Reference (Read Only)
| File | Purpose |
|------|---------|
| `hexdag/builtin/nodes/loop_node.py` | Patterns for state, collect modes |
| `hexdag/core/orchestration/components/wave_executor.py` | Semaphore concurrency |
| `hexdag/builtin/macros/tool_macro.py` | Subgraph building pattern |
| `hexdag/core/expression_parser.py` | Expression compilation |
| `hexdag/core/resolver.py` | Module path resolution |

## Verification Plan

1. **Unit Tests**
   ```bash
   uv run pytest tests/hexdag/builtin/nodes/test_control_flow_node.py -v
   ```

2. **Type Checking**
   ```bash
   uv run pyright hexdag/builtin/nodes/control_flow_node.py
   uv run pyright hexdag/core/orchestration/body_executor.py
   ```

3. **Linting**
   ```bash
   uv run ruff check hexdag/builtin/nodes/control_flow_node.py --fix
   ```

4. **Integration Test**: Create YAML pipeline using all modes
   ```yaml
   apiVersion: hexdag/v1
   kind: Pipeline
   metadata:
     name: control-flow-test
   spec:
     nodes:
       - kind: data_node
         metadata:
           name: input
         spec:
           data:
             items: [1, 2, 3]
             status: "urgent"

       - kind: control_flow_node
         metadata:
           name: process_items
         spec:
           mode: for-each
           items: "$input.items"
           concurrency: 2
           body:
             - kind: expression_node
               spec:
                 expressions:
                   doubled: "$item * 2"
         dependencies: [input]

       - kind: control_flow_node
         metadata:
           name: check_status
         spec:
           mode: if-else
           condition: "status == 'urgent'"
           body: "myapp.handle_urgent"
           else_body: "myapp.handle_normal"
         dependencies: [input]
   ```

5. **Manual Verification**
   ```bash
   uv run python -c "
   from hexdag.builtin.nodes import ControlFlowNode
   print('ControlFlowNode:', ControlFlowNode)
   "
   ```
