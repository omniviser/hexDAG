"""Integration tests for automatic run journaling + crash recovery.

Verifies that PipelineRunner with checkpoint_storage:

1. Journals a checkpoint after every completed wave (status="running")
2. Exposes run_id on every PipelineResult
3. Can discover interrupted runs via list_runs()
4. Can resume an interrupted run from the journal, skipping completed nodes
"""

from __future__ import annotations

import sys
import textwrap
from typing import TYPE_CHECKING

import pytest

from hexdag.kernel.orchestration.components.checkpoint_manager import CheckpointManager
from hexdag.kernel.orchestration.components.lifecycle_manager import PostDagHookConfig
from hexdag.kernel.pipeline_runner import PipelineRunner
from hexdag.stdlib.adapters.memory.in_memory_memory import InMemoryMemory

if TYPE_CHECKING:
    from pathlib import Path

# Pipeline with two sequential function nodes.  step_two fails while a
# "fail flag" file exists — simulating a crash mid-run (the journal from
# wave 1 is the only persisted state when the final checkpoint is disabled).
JOURNAL_YAML = """\
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: journal-test
spec:
  nodes:
    - kind: function_node
      metadata:
        name: step_one
      spec:
        fn: journal_helpers.step_one
      dependencies: []
    - kind: function_node
      metadata:
        name: step_two
      spec:
        fn: journal_helpers.step_two
        input_mapping:
          data: step_one
      dependencies: [step_one]
"""

HELPER_MODULE = '''\
"""Helpers for journal integration test (importable by module path)."""

import json
from pathlib import Path

STATE_FILE = Path(__file__).parent / "journal_state.json"


def _bump(name: str) -> int:
    state = json.loads(STATE_FILE.read_text())
    state["calls"][name] = state["calls"].get(name, 0) + 1
    STATE_FILE.write_text(json.dumps(state))
    return state["calls"][name]


def step_one(value) -> dict:
    _bump("step_one")
    raw = value["value"] if isinstance(value, dict) else value
    return {"processed": raw.upper()}


def step_two(data) -> dict:
    _bump("step_two")
    state = json.loads(STATE_FILE.read_text())
    if state["fail_step_two"]:
        raise RuntimeError("simulated crash in step_two")
    if hasattr(data, "model_dump"):  # pydantic-wrapped on resume
        data = data.model_dump()
    inner = data.get("data", data)
    return {"final": inner["processed"] + "!"}
'''


@pytest.fixture()
def helpers(tmp_path: Path):
    """Write the helper module + state file and make it importable."""
    import json

    (tmp_path / "journal_helpers.py").write_text(HELPER_MODULE)
    state_file = tmp_path / "journal_state.json"
    state_file.write_text(json.dumps({"calls": {}, "fail_step_two": True}))

    sys.path.insert(0, str(tmp_path))
    try:
        yield state_file
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("journal_helpers", None)


@pytest.fixture()
def pipeline_file(tmp_path: Path) -> Path:
    path = tmp_path / "journal_pipeline.yaml"
    path.write_text(JOURNAL_YAML)
    return path


def _read_state(state_file: Path) -> dict:
    import json

    return json.loads(state_file.read_text())


def _set_fail(state_file: Path, fail: bool) -> None:
    import json

    state = json.loads(state_file.read_text())
    state["fail_step_two"] = fail
    state_file.write_text(json.dumps(state))


class TestRunIdExposure:
    """Every PipelineResult carries the run_id."""

    @pytest.mark.asyncio()
    async def test_run_id_populated_and_stripped_from_results(self) -> None:
        runner = PipelineRunner()
        result = await runner.run_from_string(
            textwrap.dedent("""\
                apiVersion: hexdag/v1
                kind: Pipeline
                metadata:
                  name: run-id-test
                spec:
                  nodes:
                    - kind: data_node
                      metadata:
                        name: start
                      spec:
                        output:
                          value: "hello"
                      dependencies: []
            """)
        )
        assert result.run_id  # non-empty UUID


class TestRunJournal:
    """Wave-level journaling enables crash recovery."""

    @pytest.mark.asyncio()
    async def test_journal_saved_after_completed_wave(
        self, helpers: Path, pipeline_file: Path
    ) -> None:
        """A run that dies mid-execution leaves a 'running' journal entry."""
        storage = InMemoryMemory()
        runner = PipelineRunner(
            checkpoint_storage=storage,
            # Disable the final (post-DAG) checkpoint so the only persisted
            # state is the wave journal — simulates a hard process crash.
            post_hook_config=PostDagHookConfig(
                enable_checkpoint_save=False,
                enable_incremental_checkpoint=True,
            ),
        )

        with pytest.raises(Exception, match="step_two"):
            await runner.run(pipeline_file, input_data={"value": "abc"})

        runs = await runner.list_runs(status="running")
        assert len(runs) == 1
        assert runs[0]["pipeline"] == "journal-test"
        assert runs[0]["completed_nodes"] == 1

        mgr = CheckpointManager(storage=storage)
        checkpoint = await mgr.load(runs[0]["run_id"])
        assert checkpoint is not None
        assert checkpoint.node_results["step_one"] == {"processed": "ABC"}
        assert "step_two" not in checkpoint.node_results

    @pytest.mark.asyncio()
    async def test_resume_from_journal_skips_completed_nodes(
        self, helpers: Path, pipeline_file: Path
    ) -> None:
        """resume(run_id) re-runs only the nodes after the last journal."""
        storage = InMemoryMemory()
        runner = PipelineRunner(
            checkpoint_storage=storage,
            post_hook_config=PostDagHookConfig(
                enable_checkpoint_save=False,
                enable_incremental_checkpoint=True,
            ),
        )

        with pytest.raises(Exception, match="step_two"):
            await runner.run(pipeline_file, input_data={"value": "abc"})

        runs = await runner.list_runs(status="running")
        run_id = runs[0]["run_id"]

        # "Fix the bug" and resume
        _set_fail(helpers, False)
        result = await runner.resume(pipeline_file, run_id)

        assert result.node_results["step_two"] == {"final": "ABC!"}
        calls = _read_state(helpers)["calls"]
        assert calls["step_one"] == 1  # NOT re-executed on resume
        assert calls["step_two"] == 2  # failed once, succeeded on resume

    @pytest.mark.asyncio()
    async def test_journal_enabled_by_default_with_checkpoint_storage(
        self, helpers: Path, pipeline_file: Path
    ) -> None:
        """PipelineRunner(checkpoint_storage=...) journals without extra config."""
        storage = InMemoryMemory()
        runner = PipelineRunner(checkpoint_storage=storage)

        _set_fail(helpers, False)
        result = await runner.run(pipeline_file, input_data={"value": "xyz"})
        assert result.run_id

        # Final checkpoint overwrites the journal with status="saved"
        runs = await runner.list_runs()
        assert len(runs) == 1
        assert runs[0]["status"] == "saved"
        assert runs[0]["run_id"] == result.run_id
        assert runs[0]["completed_nodes"] == 2

    @pytest.mark.asyncio()
    async def test_list_runs_requires_storage(self) -> None:
        from hexdag.kernel.exceptions import PipelineRunnerError

        runner = PipelineRunner()
        with pytest.raises(PipelineRunnerError, match="no checkpoint_storage"):
            await runner.list_runs()
