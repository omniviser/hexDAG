"""CheckpointNode factory for declarative mid-pipeline save/restore."""

import re
from datetime import UTC, datetime
from typing import Any, Literal

from hexdag.kernel.context import get_user_ports
from hexdag.kernel.domain.dag import NodeSpec
from hexdag.kernel.logging import get_logger
from hexdag.kernel.orchestration.components.checkpoint_manager import CheckpointManager
from hexdag.kernel.orchestration.models import CheckpointState
from hexdag.kernel.utils.node_timer import node_timer

from .base_node_factory import BaseNodeFactory

logger = get_logger(__name__)


class CheckpointNode(BaseNodeFactory, yaml_alias="checkpoint_node"):
    """Declarative mid-pipeline checkpoint save/restore node.

    Allows explicit checkpointing at business milestones so that expensive
    work is not repeated on retry.  Uses the ``checkpoint`` port (any
    ``SupportsKeyValue`` adapter) for storage.

    Actions
    -------
    save
        Persist the selected upstream node results under ``run_id``.
        Returns ``{"saved": True, "run_id": run_id, "keys": [...]}``.
    restore
        Load a previously saved checkpoint.
        Returns the saved payload dict, or ``{"found": False}`` if none exists.

    Examples
    --------
    YAML::

        ports:
          checkpoint:
            adapter: in_memory_adapter  # or sqlite_adapter, redis_adapter, etc.

        nodes:
          - kind: checkpoint_node
            metadata:
              name: maybe_restore
            spec:
              action: restore
              run_id: "{{ metadata.run_id }}"
            dependencies: []

          - kind: checkpoint_node
            metadata:
              name: save_after_extract
            spec:
              action: save
              run_id: "{{ metadata.run_id }}"
              keys: [extract_data, transform_data]
            dependencies: [transform_data]
    """

    # Studio UI metadata
    _hexdag_icon = "Database"
    _hexdag_color = "#8b5cf6"  # violet-500

    def __call__(
        self,
        name: str,
        action: Literal["save", "restore"],
        run_id: str,
        keys: list[str] | None = None,
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a checkpoint save or restore node.

        Parameters
        ----------
        name : str
            Node name
        action : "save" | "restore"
            Whether to save or restore a checkpoint
        run_id : str
            Unique identifier for the checkpoint.  Supports template syntax:
            ``"{{ metadata.run_id }}"``
        keys : list[str] | None
            For ``save``: list of upstream node-result keys to include.
            If None, all node results available at execution time are saved.
        deps : list[str] | None
            Explicit dependency node names
        **kwargs : Any
            Additional kwargs (ignored)
        """
        _action = action
        _run_id = run_id
        _keys = keys

        async def checkpoint_fn(input_data: Any, **ports: Any) -> dict[str, Any]:
            if not ports:
                ports = get_user_ports()

            storage = ports.get("checkpoint")
            if storage is None:
                logger.warning(
                    "No 'checkpoint' port configured — checkpoint node '{}' skipped", name
                )
                return {"skipped": True, "reason": "No checkpoint port configured"}

            mgr = CheckpointManager(storage=storage)

            # Resolve run_id — support simple template {{ key }} from input_data
            resolved_run_id = _run_id
            if "{{" in _run_id and isinstance(input_data, dict):
                for match in re.finditer(r"\{\{\s*([\w.]+)\s*\}\}", _run_id):
                    key_path = match.group(1).split(".")
                    value: Any = input_data
                    for part in key_path:
                        if isinstance(value, dict):
                            value = value.get(part)
                        else:
                            value = None
                            break
                    if value is not None:
                        resolved_run_id = resolved_run_id.replace(match.group(0), str(value))

            with node_timer():
                if _action == "save":
                    return await _do_save(input_data, mgr, resolved_run_id, _keys, name)
                return await _do_restore(mgr, resolved_run_id)

        return NodeSpec(
            name=name,
            fn=checkpoint_fn,
            deps=frozenset(deps or []),
        )


async def _do_save(
    input_data: Any,
    mgr: CheckpointManager,
    run_id: str,
    keys: list[str] | None,
    node_name: str,
) -> dict[str, Any]:
    """Persist selected keys from input_data as a checkpoint."""
    if isinstance(input_data, dict):
        payload = {k: input_data[k] for k in keys if k in input_data} if keys else dict(input_data)
    else:
        payload = {"_value": input_data}

    state = CheckpointState(
        run_id=run_id,
        dag_id=node_name,
        graph_snapshot={},
        initial_input=None,
        node_results=payload,
        completed_node_ids=list(payload.keys()),
        failed_node_ids=[],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        metadata={"source_node": node_name, "action": "save"},
    )
    await mgr.save(state)
    logger.info("Checkpoint saved: run_id={}, keys={}", run_id, list(payload.keys()))
    return {"saved": True, "run_id": run_id, "keys": list(payload.keys())}


async def _do_restore(mgr: CheckpointManager, run_id: str) -> dict[str, Any]:
    """Load and return a previously saved checkpoint."""
    state = await mgr.load(run_id)
    if state is None:
        logger.debug("No checkpoint found for run_id={}", run_id)
        return {"found": False, "run_id": run_id}
    logger.info("Checkpoint restored: run_id={}", run_id)
    return {"found": True, "run_id": run_id, **state.node_results}
