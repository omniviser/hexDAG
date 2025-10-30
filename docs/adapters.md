# Adapters Reference

This page documents all registered adapters in HexDAG.

## Overview

Total adapters: **2**

## Namespace: `plugin`

### `local_observer_manager`

Local standalone implementation of observer manager.

    This implementation provides:
    - Weak reference support to prevent memory leaks
    - Event type filtering for efficiency
    - Concurrent observer execution with limits
    - Fault isolation - observer failures don't crash the pipeline
    - Timeout handling for slow observers
    - Thread pool for sync observers to avoid blocking

**Metadata:**

- **Type:** `adapter`
- **Namespace:** `plugin`
- **Name:** `local_observer_manager`

---

### `local_policy_manager`

Local policy manager using WeakSet and heapq for efficient management.

    This implementation uses Python's built-in weak reference containers for
    automatic memory management and heapq for priority-based execution.

    Key Features:
    - WeakSet for automatic cleanup of USER/TEMPORARY policies
    - WeakKeyDictionary for metadata that cleans up with policies
    - Strong references for CORE/PLUGIN policies that shouldn't be GC'd
    - Heapq for efficient O(log n) priority queue operations
    - Type-based filtering and management

**Metadata:**

- **Type:** `adapter`
- **Namespace:** `plugin`
- **Name:** `local_policy_manager`

---
