"""Unit tests for SQLAlchemyAdapter's transaction-aware data methods.

These run against real file-based SQLite (so independent sessions see each
other's commits) and prove the fix: query/execute methods route through the
run-scoped session, so inside a pipeline run they read the run's uncommitted
writes and their own writes join the run transaction.
"""

import asyncio

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("greenlet")

from hexdag.kernel.context.execution_context import set_run_id
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from hexdag_plugins.database.adapters.sqlalchemy_adapter import SQLAlchemyAdapter


@pytest.fixture(autouse=True)
def _clear_run_id():
    set_run_id(None)
    yield
    set_run_id(None)


async def make_db(tmp_path):
    """File-based SQLite so separate sessions see each other's commits."""
    dsn = f"sqlite+aiosqlite:///{tmp_path}/test.db"
    engine = create_async_engine(dsn)
    async with engine.begin() as conn:
        await conn.execute(
            text("CREATE TABLE orders (id TEXT PRIMARY KEY, status TEXT, amount INTEGER)")
        )
        await conn.execute(text("INSERT INTO orders VALUES ('O-1', 'NEW', 100)"))
        await conn.execute(text("INSERT INTO orders VALUES ('O-2', 'PAID', 200)"))
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return dsn, engine, factory


async def read_status(adapter: SQLAlchemyAdapter, order_id: str = "O-1") -> str | None:
    """Read via an independent session (sees only committed data)."""
    async with adapter.asession() as session:
        result = await session.execute(
            text("SELECT status FROM orders WHERE id = :id"), {"id": order_id}
        )
        return result.scalar_one_or_none()


class TestBasicReads:
    @pytest.mark.asyncio()
    async def test_query_with_filters_and_columns(self, tmp_path):
        _, engine, factory = await make_db(tmp_path)
        adapter = SQLAlchemyAdapter(session_factory=factory)
        await adapter.connect()  # reflect the schema into metadata

        rows = [row async for row in adapter.query("orders", filters={"status": "NEW"})]
        assert len(rows) == 1
        assert rows[0]["id"] == "O-1"

        cols = [row async for row in adapter.query("orders", columns=["id"], limit=1)]
        assert cols == [{"id": "O-1"}]
        await engine.dispose()

    @pytest.mark.asyncio()
    async def test_aexecute_query_and_query_raw(self, tmp_path):
        _, engine, factory = await make_db(tmp_path)
        adapter = SQLAlchemyAdapter(session_factory=factory)

        rows = await adapter.aexecute_query("SELECT id FROM orders ORDER BY id")
        assert [r["id"] for r in rows] == ["O-1", "O-2"]

        raw = [
            r
            async for r in adapter.query_raw("SELECT id FROM orders WHERE amount > :m", {"m": 150})
        ]
        assert raw == [{"id": "O-2"}]
        await engine.dispose()

    @pytest.mark.asyncio()
    async def test_get_table_statistics(self, tmp_path):
        _, engine, factory = await make_db(tmp_path)
        adapter = SQLAlchemyAdapter(session_factory=factory)

        stats = await adapter.get_table_statistics("orders")
        assert stats["row_count"] == 2
        await engine.dispose()


class TestReadYourWritesInRun:
    """The core fix: reads inside a run see the run's uncommitted writes."""

    @pytest.mark.asyncio()
    async def test_all_read_methods_see_staged_write(self, tmp_path):
        _, engine, factory = await make_db(tmp_path)
        adapter = SQLAlchemyAdapter(session_factory=factory)
        await adapter.connect()
        set_run_id("run-1")

        # Stage an uncommitted write on the shared run session.
        async with adapter.get_session() as session:
            await session.execute(text("UPDATE orders SET status = 'SHIPPED' WHERE id = 'O-1'"))
            await session.flush()

        # A fresh independent session must NOT see it yet.
        assert await read_status(adapter) == "NEW"

        # Every transaction-aware read method DOES see it.
        via_exec = await adapter.aexecute_query("SELECT status FROM orders WHERE id = 'O-1'")
        assert via_exec[0]["status"] == "SHIPPED"

        via_raw = [r async for r in adapter.query_raw("SELECT status FROM orders WHERE id = 'O-1'")]
        assert via_raw[0]["status"] == "SHIPPED"

        via_query = [
            r async for r in adapter.query("orders", filters={"id": "O-1"}, columns=["status"])
        ]
        assert via_query[0]["status"] == "SHIPPED"

        # Rolling back the run reverts the staged write.
        await adapter.ateardown(success=False)
        assert await read_status(adapter) == "NEW"
        await engine.dispose()


class TestExecuteRawJoinsRunTransaction:
    @pytest.mark.asyncio()
    async def test_execute_raw_rolls_back_with_failed_run(self, tmp_path):
        _, engine, factory = await make_db(tmp_path)
        adapter = SQLAlchemyAdapter(session_factory=factory)
        set_run_id("run-1")

        await adapter.execute_raw("UPDATE orders SET status = 'SHIPPED' WHERE id = 'O-1'")

        # Nothing committed mid-run.
        assert await read_status(adapter) == "NEW"

        await adapter.ateardown(success=False)
        assert await read_status(adapter) == "NEW"
        await engine.dispose()

    @pytest.mark.asyncio()
    async def test_execute_raw_commits_with_successful_run(self, tmp_path):
        _, engine, factory = await make_db(tmp_path)
        adapter = SQLAlchemyAdapter(session_factory=factory)
        set_run_id("run-1")

        await adapter.execute_raw("UPDATE orders SET status = 'SHIPPED' WHERE id = 'O-1'")
        await adapter.ateardown(success=True)

        assert await read_status(adapter) == "SHIPPED"
        await engine.dispose()

    @pytest.mark.asyncio()
    async def test_execute_raw_autocommits_standalone(self, tmp_path):
        _, engine, factory = await make_db(tmp_path)
        adapter = SQLAlchemyAdapter(session_factory=factory)  # no run_id

        await adapter.execute_raw("UPDATE orders SET status = 'SHIPPED' WHERE id = 'O-1'")
        assert await read_status(adapter) == "SHIPPED"
        await engine.dispose()


class TestFactoryOnlyAdapter:
    """A session_factory-only adapter (no DSN, engine is None) still works."""

    @pytest.mark.asyncio()
    async def test_reads_do_not_require_engine(self, tmp_path):
        _, engine, factory = await make_db(tmp_path)
        adapter = SQLAlchemyAdapter(session_factory=factory)
        assert adapter.engine is None  # never built one

        rows = await adapter.aexecute_query("SELECT COUNT(*) AS n FROM orders")
        assert rows[0]["n"] == 2

        await adapter.connect()  # reflection via the session's connection
        assert "orders" in adapter._metadata.tables
        await engine.dispose()


class TestNoDeadlockWhileIterating:
    @pytest.mark.asyncio()
    async def test_nested_db_call_mid_iteration(self, tmp_path):
        """The run lock is released before yielding, so a nested call
        mid-iteration does not deadlock."""
        _, engine, factory = await make_db(tmp_path)
        adapter = SQLAlchemyAdapter(session_factory=factory)
        set_run_id("run-1")

        seen: list[str] = []
        async for row in adapter.query_raw("SELECT id FROM orders ORDER BY id"):
            seen.append(row["id"])
            # Nested read on the same run session, mid-iteration.
            other = await asyncio.wait_for(
                adapter.aexecute_query("SELECT COUNT(*) AS n FROM orders"), timeout=5
            )
            assert other[0]["n"] == 2

        assert seen == ["O-1", "O-2"]
        await adapter.ateardown(success=True)
        await engine.dispose()
