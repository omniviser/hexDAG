# hexDAG Architecture Detail: Hexagonal Layers

This document provides an expanded view of the Hexagonal Architecture applied to hexDAG, emphasizing the flow of control and dependencies.

## I. Dependency Rule (Ports and Adapters)
The dependency flow must ALWAYS be directional: **Adapters → Ports ← Application → Domain**.
The core principle is: **Business logic must not depend on external infrastructure or I/O.**

## II. Core Modules Breakdown

### 1. `hexai/core/domain/`
- **Role:** Pure business logic and core entity definitions (DirectedGraph, NodeSpec, Event base classes).
- **Rule:** May only import Pydantic and Python standard libraries. **Strictly no I/O, database, or LLM imports.**

### 2. `hexai/core/application/`
- **Role:** Use cases and orchestration logic (Orchestrator, NodeFactory).
- **Rule:** Interacts with the outside world only via the defined interfaces in `ports/`.

### 3. `hexai/core/ports/`
- **Role:** Interface definitions (Protocols). Confirmed in `llm.py` by using `@port` and `Protocol`.
- **Content:** Python `Protocol` definitions for `LLM`, `Database`, `Memory`, and `ToolRouter`.
- **Rule:** Contains protocols **ONLY**. No implementation logic allowed.

## III. Infrastructure and Compilation
### 4. `hexai/adapters/`
- **Role:** Implementation of Ports using external technologies (e.g., OpenAIAdapter, MockAdapter).
- **Rule:** Must depend on and implement protocols from `ports/`. Use `hexai/adapters/mock/` for development testing.

### 5. `hexai/agent_factory/`
- **Role:** Compilation layer. Translates declarative YAML files into executable `DirectedGraph` objects.
- **Rule:** Handles configuration parsing and object construction.