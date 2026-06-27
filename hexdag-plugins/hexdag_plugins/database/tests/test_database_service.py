"""Tests for DatabaseService — CRUD/query + whole-run-atomic transaction."""

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
pytest.importorskip("greenlet")

from hexdag.kernel.context.execution_context import set_run_id
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from hexdag_plugins.database.service import DatabaseService


@pytest.fixture(autouse=True)
def _clear_run_id():
    set_run_id(None)
    yield
    set_run_id(None)


async def _make_db(tmp_path):
    dsn = f"sqlite+aiosqlite:///{tmp_path / 'svc.db'}"
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
async def test_standalone_insert_commits_immediately(tmp_path):
    dsn = await _make_db(tmp_path)
    svc = DatabaseService(dsn=dsn)
    await svc.insert("items", {"id": 1, "name": "a"})
    await svc.ateardown(success=True)
    assert await _rows(dsn) == [(1, "a")]


@pytest.mark.asyncio
async def test_crud_roundtrip(tmp_path):
    dsn = await _make_db(tmp_path)
    svc = DatabaseService(dsn=dsn)
    await svc.insert("items", {"id": 1, "name": "a"})
    assert (await svc.get("items", "1"))["name"] == "a"
    await svc.update("items", "1", {"name": "b"})
    assert (await svc.get("items", "1"))["name"] == "b"
    assert (await svc.query("SELECT count(*) AS n FROM items"))[0]["n"] == 1
    await svc.delete("items", "1")
    assert await svc.get("items", "1") is None
    await svc.ateardown(success=True)


@pytest.mark.asyncio
async def test_run_scoped_commit_on_success(tmp_path):
    dsn = await _make_db(tmp_path)
    svc = DatabaseService(dsn=dsn)
    set_run_id("run-commit")
    await svc.insert("items", {"id": 1, "name": "a"})
    await svc.insert("items", {"id": 2, "name": "b"})
    # Nothing committed yet (deferred to teardown).
    assert await _rows(dsn) == []
    await svc.ateardown(success=True)
    assert await _rows(dsn) == [(1, "a"), (2, "b")]


@pytest.mark.asyncio
async def test_run_scoped_rollback_on_failure(tmp_path):
    """Whole-run-atomic default: a step exception rolls back the entire run."""
    dsn = await _make_db(tmp_path)
    svc = DatabaseService(dsn=dsn)
    set_run_id("run-fail")
    await svc.insert("items", {"id": 1, "name": "a"})
    with pytest.raises(Exception):  # noqa: B017 - duplicate PK
        await svc.insert("items", {"id": 1, "name": "dup"})
    await svc.ateardown(success=True)  # success flag ignored — a step failed
    assert await _rows(dsn) == []


@pytest.mark.asyncio
async def test_stray_rollback_savepoint_does_not_clear_failure(tmp_path):
    """rollback_savepoint for a scope that was never opened must NOT clear the
    run's failure flag (else a genuinely-failed run would wrongly commit)."""
    dsn = await _make_db(tmp_path)
    svc = DatabaseService(dsn=dsn)
    set_run_id("run-stray")
    await svc.insert("items", {"id": 1, "name": "a"})
    with pytest.raises(Exception):  # noqa: B017 - duplicate PK marks the run failed
        await svc.insert("items", {"id": 1, "name": "dup"})
    # No savepoint was ever opened for this scope.
    await svc.rollback_savepoint("never-opened")
    await svc.ateardown(success=True)  # would commit if the flag were cleared
    assert await _rows(dsn) == []


@pytest.mark.asyncio
async def test_requires_dsn_or_factory():
    with pytest.raises(ValueError, match="dsn.*session_factory"):
        DatabaseService()
