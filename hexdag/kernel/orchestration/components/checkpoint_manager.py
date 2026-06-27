"""CheckpointManager component for orchestrator state persistence.

Storage-agnostic checkpoint manager using Memory Port for maximum flexibility.
Supports any backend: SQL databases, files (JSON/YAML), Redis, S3, etc.
"""

from hexdag.kernel.domain.dag import DirectedGraph, NodeSpec
from hexdag.kernel.orchestration.models import CheckpointState
from hexdag.kernel.ports.data_store import SupportsKeyValue


class CheckpointManager:
    """Manages orchestrator checkpoints using Memory Port abstraction.

    This implementation is storage-agnostic and works with any Memory backend:
    - SQL databases (via SQLiteMemoryAdapter)
    - File storage (JSON, text via FileMemoryAdapter)
    - In-memory storage (for testing)
    - Redis, S3, etc.

    Responsibilities:
    - Save/restore execution state
    - Filter graphs for resume
    - Automatic serialization via Pydantic

    Parameters
    ----------
    storage : SupportsKeyValue
        Memory port implementation for storage backend
    key_prefix : str, default="checkpoint:"
        Prefix for checkpoint keys (useful for namespacing)
    auto_checkpoint : bool, default=True
        Auto-save after nodes complete

    Examples
    --------
    In-memory storage (testing)::

        storage = InMemoryMemory()
        mgr = CheckpointManager(storage=storage)
        await mgr.save(state)
        restored = await mgr.load("run-123")

    File-based storage (production)::

        storage = FileMemoryAdapter(base_path="./checkpoints", format="json")
        mgr = CheckpointManager(storage=storage)

    Database storage (enterprise)::

        db = SQLiteAdapter(db_path="hexdag.db")
        storage = SQLiteMemoryAdapter(database=db)
        mgr = CheckpointManager(storage=storage)
    """

    def __init__(
        self,
        storage: SupportsKeyValue,
        key_prefix: str = "checkpoint:",
        auto_checkpoint: bool = True,
    ):
        self.storage = storage
        self.key_prefix = key_prefix
        self.auto_checkpoint = auto_checkpoint

    def _make_key(self, run_id: str) -> str:
        """Generate storage key for a run_id."""
        return f"{self.key_prefix}{run_id}"

    async def save(self, state: CheckpointState) -> None:
        """Save checkpoint state.

        Uses Pydantic's model_dump_json() for automatic serialization.
        All complex types (datetime, nested models) are handled automatically.

        Parameters
        ----------
        state : CheckpointState
            Complete checkpoint state to persist
        """
        key = self._make_key(state.run_id)
        # Pydantic handles all serialization including datetime, nested models, etc.
        serialized = state.model_dump_json()
        await self.storage.aset(key, serialized)

    async def load(self, run_id: str) -> CheckpointState | None:
        """Load checkpoint state by run_id.

        Uses Pydantic's model_validate_json() for automatic deserialization.

        Parameters
        ----------
        run_id : str
            Run identifier to load

        Returns
        -------
        CheckpointState | None
            Restored checkpoint state, or None if not found
        """
        key = self._make_key(run_id)
        serialized = await self.storage.aget(key)

        if serialized is None:
            return None

        # Pydantic handles all deserialization and validation
        return CheckpointState.model_validate_json(serialized)

    async def alist_runs(self) -> list[CheckpointState]:
        """List all persisted checkpoints (crash recovery discovery).

        Requires the storage backend to support key listing
        (``alist_keys``). Checkpoints that fail to deserialize are skipped.

        Returns
        -------
        list[CheckpointState]
            All checkpoints under this manager's key prefix, most
            recently updated first.

        Raises
        ------
        NotImplementedError
            If the storage backend does not support key listing.
        """
        list_keys = getattr(self.storage, "alist_keys", None)
        if list_keys is None:
            raise NotImplementedError(
                f"Storage backend {type(self.storage).__name__} does not support "
                f"key listing (alist_keys) — cannot enumerate runs"
            )

        states: list[CheckpointState] = []
        for key in await list_keys(prefix=self.key_prefix):
            serialized = await self.storage.aget(key)
            if serialized is None:
                continue
            try:
                states.append(CheckpointState.model_validate_json(serialized))
            except ValueError:  # noqa: PERF203 — skip corrupt entries, keep listing
                continue

        states.sort(key=lambda s: s.updated_at, reverse=True)
        return states

    async def load_for_resume(self, run_id: str) -> CheckpointState | None:
        """Load a checkpoint and mark it as "resuming" to block concurrent resumes.

        If the checkpoint is already in "resuming" status, raises
        ``ValueError`` to prevent concurrent resume of the same run.

        Parameters
        ----------
        run_id : str
            Run identifier to load.

        Returns
        -------
        CheckpointState | None
            Checkpoint state, or None if not found.

        Raises
        ------
        ValueError
            If the checkpoint is already being resumed.
        """
        checkpoint = await self.load(run_id)
        if checkpoint is None:
            return None

        if checkpoint.status == "resuming":
            msg = (
                f"Checkpoint '{run_id}' is already being resumed. Concurrent resume is not allowed."
            )
            raise ValueError(msg)

        # Mark as resuming and save back
        updated = checkpoint.model_copy(update={"status": "resuming"})
        await self.save(updated)
        return updated

    def filter_completed(self, graph: DirectedGraph, completed: set[str]) -> DirectedGraph:
        """Create graph with only pending nodes.

        Parameters
        ----------
        graph : DirectedGraph
            Original DAG
        completed : set[str]
            Set of completed node names

        Returns
        -------
        DirectedGraph
            New graph with only pending nodes and updated dependencies
        """
        pending = DirectedGraph()
        for spec in graph:  # Using iterator instead of .nodes.items()
            if spec.name not in completed:
                pending += NodeSpec(  # Using += operator instead of .add()
                    name=spec.name,
                    fn=spec.fn,
                    deps=frozenset(d for d in spec.deps if d not in completed),
                    timeout=spec.timeout,
                )
        return pending
