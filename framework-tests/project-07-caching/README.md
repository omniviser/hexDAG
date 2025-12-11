# Project 7: Node-Level Caching - hexDAG Port

## Overview
This folder contains the hexDAG port of LangGraph's `project-07-document-summarization-pipeline`.

**Note:** The folder name is misleading - this is actually about **caching**, not document summarization.

## What It Tests

**Caching LLM responses to avoid redundant API calls.**

```
Call 1: "What is the capital of France?"
        → [CACHE MISS] → Call LLM → "Paris" → Save to cache
        → Time: ~800ms

Call 2: "What is the capital of France?" (same question)
        → [CACHE HIT] → Return cached "Paris"
        → Time: ~1ms (no API call!)
```

### Why This Matters:
- **Save money** - Don't pay for duplicate LLM calls
- **Faster responses** - Cached results return instantly
- **Reduce API rate limits** - Fewer calls to external services

## Files
- `caching_pipeline.yaml` - hexDAG YAML pipeline
- `run_caching.py` - Python runner with cache implementation
- `README.md` - This file

## The Difference

| | LangGraph | hexDAG |
|---|-----------|--------|
| **Caching** | Built-in `SqliteSaver` | Manual implementation |
| **Config** | `checkpointer=memory` | Custom cache class |
| **Thread ID** | `config={"thread_id": "1"}` | Passed as input |

### LangGraph (built-in):
```python
memory = SqliteSaver.from_conn_string(":memory:")
app = graph.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}
app.invoke(input, config=config)  # Cached!
```

### hexDAG (manual):
```python
class SimpleCache:
    def get(self, key): ...
    def set(self, key, value): ...

# Must implement caching in node function
cached = cache.get(key)
if cached:
    return cached
```

## Verdict: PARTIAL GAP ⚠️

hexDAG doesn't have built-in checkpointing/caching like LangGraph's `SqliteSaver`.

**Workaround:** Implement caching manually (as shown in `run_caching.py`).

**Recommendation:** hexDAG could add a caching decorator or built-in cache port.

## Cache Limitations & Resolutions

### The Problem: Memory Constraints

The current `SimpleCache` implementation stores everything in RAM (Python dictionary). This has serious limitations:

| Issue | Impact |
|-------|--------|
| **Memory exhaustion** | Cache grows unbounded until system runs out of RAM |
| **No persistence** | Cache is lost when program restarts |
| **No eviction** | Old/unused entries never get removed |
| **Single process** | Cache not shared between multiple instances |

### Example: Memory Growth

```
Call 1: Cache size = 1 entry (~1KB)
Call 100: Cache size = 100 entries (~100KB)
Call 10,000: Cache size = 10,000 entries (~10MB)
Call 1,000,000: Cache size = 1M entries (~1GB) ← PROBLEM!
```

### Solutions

#### 1. LRU (Least Recently Used) Cache
Remove the oldest unused entries when cache is full.

```python
from functools import lru_cache

@lru_cache(maxsize=1000)  # Keep only 1000 most recent
def cached_llm_call(prompt: str) -> str:
    return call_llm(prompt)
```

#### 2. TTL (Time-To-Live) Expiration
Entries expire after a set time period.

```python
class TTLCache:
    def __init__(self, ttl_seconds=3600):  # 1 hour default
        self._cache = {}
        self._timestamps = {}
        self.ttl = ttl_seconds

    def get(self, key):
        if key in self._cache:
            if time.time() - self._timestamps[key] < self.ttl:
                return self._cache[key]
            else:
                del self._cache[key]  # Expired
        return None
```

#### 3. Max Size with Eviction
Hard limit on cache size.

```python
class BoundedCache:
    def __init__(self, max_size=1000):
        self._cache = {}
        self.max_size = max_size

    def set(self, key, value):
        if len(self._cache) >= self.max_size:
            # Remove oldest entry (FIFO)
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = value
```

#### 4. Disk-Based Cache (Like LangGraph)
Store cache on disk for persistence and larger capacity.

```python
import sqlite3

class DiskCache:
    def __init__(self, db_path="cache.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache
            (key TEXT PRIMARY KEY, value TEXT)
        """)

    def get(self, key):
        result = self.conn.execute(
            "SELECT value FROM cache WHERE key = ?", (key,)
        ).fetchone()
        return result[0] if result else None

    def set(self, key, value):
        self.conn.execute(
            "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
            (key, value)
        )
        self.conn.commit()
```

#### 5. Redis/External Cache
For distributed systems and production use.

```python
import redis

class RedisCache:
    def __init__(self):
        self.client = redis.Redis(host='localhost', port=6379)

    def get(self, key):
        return self.client.get(key)

    def set(self, key, value, ttl=3600):
        self.client.setex(key, ttl, value)
```

### Comparison: LangGraph vs hexDAG Caching

| Feature | LangGraph | hexDAG (current) | hexDAG (recommended) |
|---------|-----------|------------------|---------------------|
| **Storage** | SQLite (disk) | RAM (dict) | Disk or Redis |
| **Persistence** | Yes | No | Yes |
| **Max size** | Disk limit | RAM limit | Configurable |
| **TTL support** | Yes | No | Add it |
| **Built-in** | Yes | No | Should add |
| **Shared cache** | Via file | No | Via Redis |

### Recommendation for hexDAG

hexDAG should consider adding:

1. **`@cached` decorator** for node functions
2. **Built-in cache port** with configurable backends (memory, disk, Redis)
3. **TTL and max-size options** in YAML configuration

Example future YAML syntax:
```yaml
spec:
  nodes:
    - kind: llm_node
      metadata:
        name: chatbot
      spec:
        cache:
          enabled: true
          backend: sqlite  # or: memory, redis
          ttl: 3600        # seconds
          max_size: 10000  # entries
```

## How to Run

```bash
cd framework-tests/project7-caching
..\..\.venv\Scripts\python.exe run_caching.py
```

Expected output:
```
[Call 1] CACHE MISS - calls LLM (~800ms)
[Call 2] CACHE HIT - returns cached (~1ms)
[Call 3] CACHE MISS - different question (~800ms)
```

## Test Results
[To be filled after testing]

## Observations
[To be filled after testing]
