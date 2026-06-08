# hexdag-plugins[langfuse]

Langfuse LLM observability for hexDAG pipelines. Maps hexDAG's execution model to Langfuse traces, spans, and generations — zero kernel changes, pure dependency injection.

## Install

```bash
pip install hexdag-plugins[langfuse]
```

Set environment variables (or pass to `Langfuse()` constructor):

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://cloud.langfuse.com  # or self-hosted
```

## Quick Start

```python
from langfuse import Langfuse
from hexdag import System
from hexdag.drivers.observer_manager import LocalObserverManager
from hexdag_plugins.langfuse import LangfuseObserver

observer_mgr = LocalObserverManager()
observer_mgr.register(
    LangfuseObserver(Langfuse(), session_id="user-abc"),
    keep_alive=True,
)

system = System.from_yaml("system.yaml", observer_manager=observer_mgr)
result = await system.run({"query": "Hello"})
```

## Trace Hierarchy

```
System              →  trace
  Process           →    span
    Pipeline        →      span
      Node          →        span
        LLM call    →          generation (model, messages, response, tokens)
        Tool call   →          span (tool name, params, result)
```

Standalone pipelines (no System) auto-create their own trace.

## API

```python
LangfuseObserver(
    client: Langfuse,                       # required
    session_id: str | None = None,          # group traces in Langfuse UI
    trace_metadata: dict | None = None,     # extra metadata on every trace
    flush_on_complete: bool = True,          # auto-flush when outermost scope ends
)
```

## What Gets Traced

| hexDAG Event | Langfuse | Data |
|---|---|---|
| `SystemStarted` | trace | name, process count |
| `ProcessStarted` | span | process name, index |
| `PipelineStarted` | span | name, node/wave counts |
| `NodeStarted` | span | name, wave, dependencies |
| `LLMPortCall` | generation | model, messages, response, token usage |
| `ToolRouterPortCall` | span | tool name, params, result |
| `NodeCompleted` | span.end() | result, duration |
| `NodeFailed` | span.end(ERROR) | error message |
| `ProcessCompleted` | span.end() | status, duration |
| `SystemCompleted` | trace.update() | results, status |

## Examples

### FastAPI — per-request tracing

```python
from fastapi import FastAPI
from langfuse import Langfuse
from hexdag import System
from hexdag.drivers.observer_manager import LocalObserverManager
from hexdag_plugins.langfuse import LangfuseObserver

app = FastAPI()
langfuse = Langfuse()

@app.post("/chat")
async def chat(message: str, session_id: str = "anon"):
    mgr = LocalObserverManager()
    mgr.register(
        LangfuseObserver(langfuse, session_id=session_id),
        keep_alive=True,
    )
    system = System.from_yaml("system.yaml", observer_manager=mgr)
    return await system.run_process("chat_agent", {"message": message})
```

### Alongside other observers

```python
from hexdag.stdlib.observers import SimpleLoggingObserver

mgr = LocalObserverManager()
mgr.register(SimpleLoggingObserver(verbose=True).handle, keep_alive=True)
mgr.register(LangfuseObserver(Langfuse()), keep_alive=True)
```

### Standalone pipeline (no System)

```python
from hexdag.kernel.pipeline_runner import PipelineRunner

mgr = LocalObserverManager()
mgr.register(LangfuseObserver(Langfuse(), session_id="run-1"), keep_alive=True)

runner = PipelineRunner(port_overrides={"observer_manager": mgr})
await runner.run("pipeline.yaml", input_data={"text": "hello"})
```

## How It Works

hexDAG's kernel emits events during execution (`LLMPortCall`, `NodeStarted`, etc.). The `ObservableLLM` middleware (auto-stacked) captures full LLM call data. `LangfuseObserver` is a read-only observer that translates these events to Langfuse SDK calls.

```
Your App                          hexDAG Kernel
─────────                         ─────────────
Langfuse()                        Orchestrator emits events
    │                                    │
LangfuseObserver                  ObservableLLM emits LLMPortCall
    │                                    │
ObserverManager.register()        ObserverManager.notify(event)
    │                                    │
System.from_yaml(obs_mgr) ──────► LangfuseObserver.handle(event)
                                         │
                                  langfuse.trace/span/generation
```

No langfuse dependency in hexDAG's kernel — the application owns the client.
