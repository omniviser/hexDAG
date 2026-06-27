"""End-to-end tests for the `transaction` macro (savepoint partial rollback)."""

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
pytest.importorskip("greenlet")

from hexdag.kernel.pipeline_runner import PipelineRunner
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

import hexdag_plugins.database.macros.transaction  # noqa: F401 - register macro alias

_MACRO = "hexdag_plugins.database.macros.TransactionMacro"


def _pipeline_yaml(dsn: str) -> str:
    return f"""
apiVersion: hexdag/v1
kind: Pipeline
metadata: {{ name: tx-macro-test }}
spec:
  services:
    database:
      class: hexdag_plugins.database.service.DatabaseService
      config: {{ dsn: "{dsn}" }}
  nodes:
    - kind: service_call_node
      metadata: {{ name: outside }}
      spec:
        service: database
        method: insert
        input_mapping: {{ table: "'items'", data: "$input.r_outside" }}
      dependencies: []
    - kind: macro_invocation
      metadata: {{ name: scope }}
      spec:
        macro: {_MACRO}
        config:
          service: database
          nodes:
            - kind: service_call_node
              metadata: {{ name: in1 }}
              spec:
                service: database
                method: insert
                input_mapping: {{ table: "'items'", data: "$input.r_in1" }}
            - kind: service_call_node
              metadata: {{ name: in2 }}
              spec:
                service: database
                method: insert
                input_mapping: {{ table: "'items'", data: "$input.r_in2" }}
                dependencies: [in1]
        dependencies: [outside]
"""


async def _setup(tmp_path):
    dsn = f"sqlite+aiosqlite:///{tmp_path / 'tx.db'}"
    eng = create_async_engine(dsn)
    async with eng.begin() as conn:
        await conn.execute(text("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)"))
    await eng.dispose()
    return dsn


async def _rows(dsn):
    eng = create_async_engine(dsn)
    async with eng.connect() as conn:
        result = await conn.execute(text("SELECT id, name FROM items ORDER BY id"))
        rows = [tuple(r) for r in result.fetchall()]
    await eng.dispose()
    return rows


@pytest.mark.asyncio
async def test_savepoint_commits_on_success(tmp_path):
    dsn = await _setup(tmp_path)
    inputs = {
        "r_outside": {"id": 1, "name": "outside"},
        "r_in1": {"id": 2, "name": "in1"},
        "r_in2": {"id": 3, "name": "in2"},
    }
    res = await PipelineRunner().run_from_string(_pipeline_yaml(dsn), inputs)
    assert res.status == "completed"
    assert await _rows(dsn) == [(1, "outside"), (2, "in1"), (3, "in2")]


@pytest.mark.asyncio
async def test_savepoint_partial_rollback_on_failure(tmp_path):
    """A failing step rolls back only its savepoint scope; the run still commits."""
    dsn = await _setup(tmp_path)
    inputs = {
        "r_outside": {"id": 1, "name": "outside"},
        "r_in1": {"id": 2, "name": "in1"},
        "r_in2": {"id": 1, "name": "dup"},  # duplicate PK → in2 fails
    }
    res = await PipelineRunner().run_from_string(_pipeline_yaml(dsn), inputs)
    assert res.status == "completed"
    # Scope (in1, in2) rolled back; out-of-scope `outside` persists.
    assert await _rows(dsn) == [(1, "outside")]


def _two_scope_yaml(dsn: str) -> str:
    """Two sibling transaction scopes in one run — unsupported, must fail fast."""
    return f"""
apiVersion: hexdag/v1
kind: Pipeline
metadata: {{ name: tx-two-scopes }}
spec:
  services:
    database:
      class: hexdag_plugins.database.service.DatabaseService
      config: {{ dsn: "{dsn}" }}
  nodes:
    - kind: macro_invocation
      metadata: {{ name: scopeA }}
      spec:
        macro: {_MACRO}
        config:
          service: database
          nodes:
            - kind: service_call_node
              metadata: {{ name: a1 }}
              spec:
                service: database
                method: insert
                input_mapping: {{ table: "'items'", data: "$input.r_a1" }}
        dependencies: []
    - kind: macro_invocation
      metadata: {{ name: scopeB }}
      spec:
        macro: {_MACRO}
        config:
          service: database
          nodes:
            - kind: service_call_node
              metadata: {{ name: b1 }}
              spec:
                service: database
                method: insert
                input_mapping: {{ table: "'items'", data: "$input.r_b1" }}
        dependencies: []
"""


@pytest.mark.asyncio
async def test_second_scope_in_one_run_fails_fast(tmp_path):
    """A second transaction scope in the same run is rejected with a clear error
    (not a SQLAlchemy IllegalStateChangeError, not a silent partial commit)."""
    from hexdag.kernel.exceptions import HexDAGError

    dsn = await _setup(tmp_path)
    inputs = {"r_a1": {"id": 1, "name": "a1"}, "r_b1": {"id": 2, "name": "b1"}}
    with pytest.raises(HexDAGError) as exc:
        await PipelineRunner().run_from_string(_two_scope_yaml(dsn), inputs)
    msg = str(exc.value)
    assert "one `transaction` scope" in msg or "System" in msg
    # Nothing committed — the run aborted before any commit.
    assert await _rows(dsn) == []
