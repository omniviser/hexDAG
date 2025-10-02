# 🚀 hexDAG Architecture Roadmap: Distributed + Dynamic DAGs

**Status:** Planning Phase
**Target:** Multi-agent systems with dynamic DAG modification during execution
**Last Updated:** 2025-10-02

---

## 🎯 **Vision**

Enable hexDAG to support:
1. **Distributed Node Execution** - Nodes run on different machines/workers
2. **Dynamic DAG Modification** - Agents can add/modify the graph during execution
3. **Multi-Agent Coordination** - Multiple agents collaboratively build and execute workflows
4. **Fault Tolerance** - Graceful handling of node/worker failures
5. **Hot Reloading** - Update node implementations without stopping execution

---

## 📊 **Current Architecture Analysis**

### ✅ **What We Have**

1. **Checkpoint System** ([models.py:70-100](hexai/core/orchestration/models.py#L70-L100))
   - `CheckpointState` with `graph_snapshot`
   - Serialization/deserialization ready
   - Resume from checkpoint support

2. **Event System** ([events.py](hexai/core/application/events/events.py))
   - Comprehensive observability
   - Observer pattern with async notifications
   - Already distributed-friendly

3. **Context Propagation** ([execution_context.py](hexai/core/context/execution_context.py))
   - `ContextVar` for async-safe state
   - Immutable `MappingProxyType` for thread safety
   - Ready for distributed context passing

4. **Wave-Based Execution** ([wave_executor.py](hexai/core/orchestration/components/wave_executor.py))
   - Natural parallelism boundaries
   - Semaphore-based concurrency control
   - Easy to map to worker pools

### ⚠️ **Current Limitations**

1. **Mutable DAG State** - `DirectedGraph.add()` mutates in-place
   - Race conditions in multi-agent scenarios
   - No version tracking
   - Cache invalidation is local-only

2. **In-Memory Only** - No distributed state management
   - All nodes in single process
   - No worker coordination
   - No distributed cache

3. **Static Wave Planning** - Waves computed once at start
   - Can't add nodes mid-execution
   - No dynamic rescheduling

4. **Single Orchestrator** - One orchestrator per DAG
   - Can't split across workers
   - No master/worker pattern

---

## 🏗️ **Proposed Architecture**

### 1️⃣ **Immutable DAG with Versioning (Foundation)**

#### **Problem:** Multi-agent modifications create race conditions

```python
# Current (mutable)
dag.add(node)  # Modifies in-place - dangerous!
waves = dag.waves()  # Might be stale

# Proposed (immutable)
dag_v2 = dag.add_node(node)  # Returns new version
dag_v2.version  # 2 (incremented)
dag.version     # 1 (unchanged)
```

#### **Design:**

```python
from dataclasses import dataclass, replace
from collections.abc import Hashable

@dataclass(frozen=True)
class DirectedGraphV2:
    """Immutable DirectedGraph with version tracking.

    Uses structural sharing (persistent data structures) for memory efficiency.
    Copy-on-write semantics ensure thread/process safety.
    """

    version: int
    nodes: MappingProxyType[str, NodeSpec]  # Immutable view
    _forward_edges: FrozenDict[str, frozenset[str]]
    _reverse_edges: FrozenDict[str, frozenset[str]]

    # Cryptographic hash for distributed consistency
    _content_hash: str = field(init=False)

    def __post_init__(self):
        # Compute content hash for distributed validation
        import hashlib
        content = f"{self.version}:{sorted(self.nodes.keys())}"
        object.__setattr__(self, '_content_hash',
                         hashlib.sha256(content.encode()).hexdigest()[:16])

    def add_node(self, node_spec: NodeSpec) -> "DirectedGraphV2":
        """Return new version with added node (copy-on-write)."""
        new_nodes = {**self.nodes, node_spec.name: node_spec}
        return replace(
            self,
            version=self.version + 1,
            nodes=MappingProxyType(new_nodes),
            # Update edges...
        )

    @cached_property
    def waves(self) -> tuple[tuple[str, ...], ...]:
        """Cached waves (immutable tuple for hashability)."""
        # Compute waves - result is immutable
        return tuple(tuple(wave) for wave in self._compute_waves())
```

**Benefits:**
- ✅ Thread-safe by design (no locks needed)
- ✅ Version tracking for conflict resolution
- ✅ Content hashing for distributed validation
- ✅ Structural sharing (memory efficient)
- ✅ `cached_property` works perfectly (immutable)

---

### 2️⃣ **Distributed Worker Pool Architecture**

#### **Design:**

```
┌─────────────────────────────────────────────────────────┐
│                   Master Orchestrator                    │
│  - DAG version management                                │
│  - Wave scheduling                                       │
│  - Worker coordination                                   │
│  - Event aggregation                                     │
└─────────────┬───────────────────────────────────────────┘
              │
    ┌─────────┼─────────┬─────────────┬─────────────┐
    │         │         │             │             │
┌───▼───┐ ┌──▼────┐ ┌──▼────┐    ┌──▼────┐    ┌──▼────┐
│Worker1│ │Worker2│ │Worker3│ .. │WorkerN│    │WorkerM│
│Node A │ │Node B │ │Node C │    │Node D │    │Hot    │
│       │ │       │ │       │    │       │    │Standby│
└───┬───┘ └───┬───┘ └───┬───┘    └───┬───┘    └───────┘
    │         │         │            │
    └─────────┴─────────┴────────────┘
              │
    ┌─────────▼─────────────────────────┐
    │   Distributed State Store          │
    │   (Redis/etcd/Consul)              │
    │   - DAG versions                   │
    │   - Node results                   │
    │   - Worker health                  │
    │   - Locks for critical sections    │
    └────────────────────────────────────┘
```

#### **Implementation Sketch:**

```python
from dataclasses import dataclass
from enum import Enum

class WorkerStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    FAILED = "failed"

@dataclass
class Worker:
    """Distributed worker for node execution."""

    worker_id: str
    hostname: str
    status: WorkerStatus
    current_node: str | None
    capabilities: set[str]  # e.g., {"gpu", "llm", "db"}

    async def execute_node(
        self,
        node_spec: NodeSpec,
        inputs: dict[str, Any],
        context: ExecutionContext,
    ) -> Any:
        """Execute node and return result."""
        # Node execution with telemetry
        pass

class DistributedOrchestrator:
    """Master orchestrator for distributed execution."""

    def __init__(
        self,
        state_store: StateStore,  # Redis/etcd backend
        worker_pool: WorkerPool,
    ):
        self.state_store = state_store
        self.worker_pool = worker_pool
        self.dag_versions: dict[str, DirectedGraphV2] = {}

    async def run_distributed(
        self,
        dag: DirectedGraphV2,
        initial_input: Any,
    ) -> dict[str, Any]:
        """Execute DAG across worker pool."""

        # 1. Publish DAG version to state store
        await self.state_store.set_dag(dag.version, dag.serialize())

        # 2. Execute waves with worker assignment
        for wave in dag.waves:
            # Assign nodes to workers based on capabilities
            assignments = await self.worker_pool.assign_nodes(wave, dag)

            # Execute wave across workers
            results = await asyncio.gather(
                *[worker.execute_node(dag.nodes[node], inputs, ctx)
                  for node, worker in assignments.items()]
            )

            # 3. Store results in distributed state
            await self.state_store.store_wave_results(dag.version, wave, results)

        return await self.state_store.get_all_results(dag.version)
```

---

### 3️⃣ **Dynamic DAG Modification Protocol**

#### **Problem:** Agents need to add nodes during execution

**Use Case:**
```python
# Agent discovers it needs additional analysis
while executing_node:
    if needs_deeper_analysis:
        # Agent proposes new subgraph
        new_dag = current_dag.add_node(
            NodeSpec("deep_analysis", analyze_deeply)
            .after("current_node")
        )
        # Submit modification proposal
        await orchestrator.propose_dag_modification(new_dag)
```

#### **Design: DAG Modification as State Machine**

```python
class DAGModificationProposal(BaseModel):
    """Proposal to modify DAG during execution."""

    proposal_id: str
    source_agent: str
    current_version: int  # DAG version when proposed
    proposed_dag: DirectedGraphV2  # New version
    modification_type: Literal["add_node", "add_subgraph", "replace_node"]
    justification: str
    priority: int  # For conflict resolution

class DAGModificationState(Enum):
    PROPOSED = "proposed"
    VALIDATING = "validating"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"

class DynamicDAGController:
    """Manages dynamic DAG modifications during execution."""

    async def propose_modification(
        self,
        proposal: DAGModificationProposal,
    ) -> str:
        """Agent proposes DAG modification."""

        # 1. Validate proposal doesn't break running nodes
        if not await self._validate_safe(proposal):
            raise ValueError("Unsafe modification")

        # 2. Check for conflicts with other proposals
        conflicts = await self._check_conflicts(proposal)
        if conflicts:
            # Conflict resolution (CRDT-style)
            resolved = await self._resolve_conflicts(proposal, conflicts)
            proposal = resolved

        # 3. Merge proposal into next version
        new_dag = self._merge_proposal(self.current_dag, proposal)

        # 4. Recompute waves for remaining execution
        remaining_waves = new_dag.waves_after(self.current_wave_index)

        # 5. Publish new version to workers
        await self.state_store.publish_dag_version(new_dag)

        return proposal.proposal_id

    def _validate_safe(self, proposal: DAGModificationProposal) -> bool:
        """Ensure modification doesn't break running nodes."""
        # Check:
        # - No cycles introduced
        # - No modification to completed nodes
        # - New dependencies are resolvable
        # - Type compatibility maintained
        pass
```

---

### 4️⃣ **Multi-Agent Coordination Layer**

#### **Design: Agent Roles + DAG Ownership**

```python
class AgentRole(Enum):
    ORCHESTRATOR = "orchestrator"  # Can modify DAG structure
    EXECUTOR = "executor"          # Executes assigned nodes
    OBSERVER = "observer"          # Monitors, no modifications

@dataclass
class Agent:
    """Multi-agent participant in DAG execution."""

    agent_id: str
    role: AgentRole
    capabilities: set[str]
    trust_level: float  # 0.0-1.0 for proposal approval

    async def propose_subgraph(
        self,
        parent_node: str,
        subgraph: DirectedGraphV2,
        reason: str,
    ) -> DAGModificationProposal:
        """Propose adding subgraph to DAG."""
        proposal = DAGModificationProposal(
            proposal_id=str(uuid.uuid4()),
            source_agent=self.agent_id,
            current_version=self.current_dag_version,
            proposed_dag=self.current_dag.merge_subgraph(parent_node, subgraph),
            modification_type="add_subgraph",
            justification=reason,
            priority=self._calculate_priority(reason),
        )
        return await self.controller.propose_modification(proposal)

class MultiAgentCoordinator:
    """Coordinates multiple agents modifying DAG."""

    def __init__(self):
        self.agents: dict[str, Agent] = {}
        self.consensus_protocol = ConsensusProtocol()

    async def vote_on_proposal(
        self,
        proposal: DAGModificationProposal,
    ) -> bool:
        """Agents vote on proposal approval."""

        # Weighted voting based on trust levels
        votes = await asyncio.gather(*[
            agent.vote(proposal)
            for agent in self.agents.values()
            if agent.role == AgentRole.ORCHESTRATOR
        ])

        # Consensus algorithm (e.g., 2/3 majority)
        weighted_score = sum(
            vote * agent.trust_level
            for vote, agent in zip(votes, self.agents.values())
        )

        threshold = 0.66 * sum(a.trust_level for a in self.agents.values())
        return weighted_score >= threshold
```

---

### 5️⃣ **Conflict-Free Replicated Data Types (CRDTs) for DAG**

#### **Problem:** Multiple agents modify DAG concurrently

**Solution:** Use CRDT principles for automatic conflict resolution

```python
class DAG_CRDT:
    """CRDT-based DAG for concurrent modifications.

    Uses Lamport timestamps and vector clocks for causality tracking.
    Automatically merges concurrent modifications without coordination.
    """

    def __init__(self):
        self.nodes: dict[str, tuple[NodeSpec, VectorClock]] = {}
        self.edges: dict[tuple[str, str], VectorClock] = {}
        self.tombstones: set[str] = set()  # Deleted nodes
        self.vector_clock: VectorClock = VectorClock()

    def add_node_crdt(
        self,
        node: NodeSpec,
        agent_id: str,
    ) -> "DAG_CRDT":
        """Add node with CRDT semantics (idempotent)."""

        # Increment vector clock for this agent
        new_clock = self.vector_clock.increment(agent_id)

        # Add node with timestamp
        new_nodes = {
            **self.nodes,
            node.name: (node, new_clock)
        }

        return DAG_CRDT(
            nodes=new_nodes,
            edges=self.edges,
            tombstones=self.tombstones,
            vector_clock=new_clock,
        )

    def merge(self, other: "DAG_CRDT") -> "DAG_CRDT":
        """Merge two CRDT DAGs (commutative & associative)."""

        # Merge nodes: keep node with higher vector clock
        merged_nodes = {}
        all_node_names = set(self.nodes) | set(other.nodes)

        for name in all_node_names:
            if name in self.tombstones or name in other.tombstones:
                continue  # Node was deleted

            node1, clock1 = self.nodes.get(name, (None, VectorClock()))
            node2, clock2 = other.nodes.get(name, (None, VectorClock()))

            # Keep node with higher causality
            if clock1 > clock2:
                merged_nodes[name] = (node1, clock1)
            else:
                merged_nodes[name] = (node2, clock2)

        # Merge edges similarly...

        return DAG_CRDT(
            nodes=merged_nodes,
            edges=self._merge_edges(other),
            tombstones=self.tombstones | other.tombstones,
            vector_clock=self.vector_clock.merge(other.vector_clock),
        )
```

---

### 6️⃣ **Hot Node Reloading & Migration**

#### **Design: Update node implementation without stopping execution**

```python
class HotReloadManager:
    """Manages hot reloading of node implementations."""

    async def reload_node_impl(
        self,
        node_name: str,
        new_impl: Callable,
        version: int,
    ):
        """Hot reload node implementation."""

        # 1. Validate new implementation signature matches
        old_node = self.dag.nodes[node_name]
        if not self._signature_compatible(old_node.func, new_impl):
            raise ValueError("Incompatible signature")

        # 2. Create new node spec with updated impl
        new_node = old_node.replace(func=new_impl)

        # 3. Update DAG with new version
        new_dag = self.dag.replace_node(node_name, new_node)

        # 4. Notify workers to update cached implementations
        await self.worker_pool.broadcast_node_update(node_name, new_impl, version)

        # 5. Nodes executing old version complete normally
        # 6. New executions use new version
        self.dag = new_dag

class NodeMigration:
    """Migrate running node to different worker."""

    async def migrate_node(
        self,
        node_name: str,
        from_worker: Worker,
        to_worker: Worker,
    ):
        """Live migrate node execution to new worker."""

        # 1. Checkpoint current node state
        state = await from_worker.checkpoint_node(node_name)

        # 2. Transfer state to new worker
        await to_worker.restore_node(node_name, state)

        # 3. Resume execution on new worker
        await to_worker.resume_node(node_name)

        # 4. Cleanup old worker
        await from_worker.cleanup_node(node_name)
```

---

## 🎯 **Implementation Roadmap**

### Phase 1: Foundation (Week 1-2)
- [ ] Immutable `DirectedGraphV2` with versioning
- [ ] Content hashing for distributed validation
- [ ] DAG serialization/deserialization improvements
- [ ] Version compatibility checking

### Phase 2: Distributed Execution (Week 3-4)
- [ ] Worker pool abstraction
- [ ] Distributed state store (Redis adapter)
- [ ] Master orchestrator with worker coordination
- [ ] Wave-based work distribution

### Phase 3: Dynamic Modifications (Week 5-6)
- [ ] `DAGModificationProposal` system
- [ ] Safe modification validator
- [ ] Dynamic wave recomputation
- [ ] Conflict detection

### Phase 4: Multi-Agent (Week 7-8)
- [ ] Agent roles and capabilities
- [ ] Consensus protocol for modifications
- [ ] CRDT-based DAG merging
- [ ] Trust-weighted voting

### Phase 5: Advanced Features (Week 9-10)
- [ ] Hot node reloading
- [ ] Node migration between workers
- [ ] Distributed event aggregation
- [ ] Multi-region support

---

## 💡 **Key Design Decisions**

### ✅ **What to Keep from Current Architecture**

1. **Event System** - Already distributed-friendly, just needs aggregation
2. **Checkpoint System** - Perfect for fault tolerance
3. **Context Propagation** - `ContextVar` works across workers via serialization
4. **Wave-Based Execution** - Natural unit of work for distribution

### 🔄 **What to Change**

1. **Mutable DAG** → Immutable versioned DAG
2. **Single-process** → Distributed worker pool
3. **Static waves** → Dynamic wave recomputation
4. **No coordination** → Multi-agent consensus

### 🆕 **What to Add**

1. **CRDT semantics** for conflict-free concurrent modifications
2. **Vector clocks** for causality tracking
3. **Distributed locks** for critical sections
4. **Worker health monitoring** and failover
5. **Hot reloading** for node implementations

---

## 🧪 **Example: Multi-Agent Dynamic DAG**

```python
# Agent 1: Orchestrator agent discovers need for analysis
async def research_agent(input_data: dict) -> dict:
    results = await do_research(input_data)

    if results["complexity"] > 0.8:
        # Need deeper analysis - propose modification
        deep_analysis_node = NodeSpec(
            name="deep_analysis",
            func=deep_analyze,
            in_model=DeepAnalysisInput,
        ).after("research_agent")

        proposal = await agent.propose_node_addition(deep_analysis_node)
        await wait_for_approval(proposal)

    return results

# Agent 2: Executor agent reviews proposal
async def executor_agent_vote(proposal: DAGModificationProposal) -> bool:
    # Validate proposal
    is_safe = validate_proposal(proposal)
    estimate_cost = estimate_execution_cost(proposal)

    return is_safe and estimate_cost < budget

# System automatically merges approved modifications
# DAG evolves: v1 → v2 → v3 during execution
# Workers seamlessly pick up new versions
```

---

## 🎓 **Research Papers & Techniques**

1. **CRDTs**: Shapiro et al., "A comprehensive study of CRDTs"
2. **Vector Clocks**: Lamport, "Time, Clocks, and Ordering of Events"
3. **Distributed Consensus**: Raft/Paxos for coordination
4. **Workflow Scheduling**: DAG scheduling in Apache Airflow/Prefect
5. **Dynamic Graphs**: Incremental graph algorithms

---

## 🚀 **Next Steps**

1. **Prototype `DirectedGraphV2`** with immutability
2. **Design state store interface** (Redis/etcd)
3. **Implement worker pool** abstraction
4. **Create modification proposal** system
5. **Test with simple multi-agent** scenario

---

**This is a living document. Update as architecture evolves!**
