"""Tests for SQLAlchemyAdapter run-scoped sessions and the state backend."""

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("greenlet")

from hexdag.kernel.context.execution_context import set_run_id
from hexdag.kernel.domain.entity_state import StateMachineConfig
from hexdag.kernel.exceptions import StaleStateError
from hexdag.stdlib.lib.entity_state import EntityState
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from hexdag_plugins.database._ports import SupportsSessionFactory
from hexdag_plugins.database.adapters import (
    SQLAlchemyAdapter,
    SQLAlchemyStateBackend,
)


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
        await conn.execute(text("CREATE TABLE orders (id TEXT PRIMARY KEY, status TEXT)"))
        await conn.execute(text("INSERT INTO orders VALUES ('O-1', 'NEW')"))
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return engine, factory


async def read_status(adapter: SQLAlchemyAdapter, order_id: str = "O-1") -> str:
    """Read via an independent session (sees only committed data)."""
    async with adapter.asession() as session:
        result = await session.execute(
            text("SELECT status FROM orders WHERE id = :id"), {"id": order_id}
        )
        return result.scalar_one()


class TestAdapterConstruction:
    def test_requires_dsn_or_session_factory(self):
        with pytest.raises(ValueError, match="dsn.*session_factory"):
            SQLAlchemyAdapter()

    def test_implements_session_factory_protocol(self, tmp_path):
        adapter = SQLAlchemyAdapter(dsn=f"sqlite+aiosqlite:///{tmp_path}/x.db")
        assert isinstance(adapter, SupportsSessionFactory)


class TestStandaloneMode:
    @pytest.mark.asyncio()
    async def test_autocommit_per_call(self, tmp_path):
        engine, factory = await make_db(tmp_path)
        adapter = SQLAlchemyAdapter(session_factory=factory)

        async with adapter.get_session() as session:
            await session.execute(text("UPDATE orders SET status = 'PAID' WHERE id = 'O-1'"))

        assert await read_status(adapter) == "PAID"
        await engine.dispose()

    @pytest.mark.asyncio()
    async def test_rollback_on_exception(self, tmp_path):
        engine, factory = await make_db(tmp_path)
        adapter = SQLAlchemyAdapter(session_factory=factory)

        with pytest.raises(RuntimeError, match="boom"):
            async with adapter.get_session() as session:
                await session.execute(text("UPDATE orders SET status = 'PAID' WHERE id = 'O-1'"))
                raise RuntimeError("boom")

        assert await read_status(adapter) == "NEW"
        await engine.dispose()


class TestPipelineMode:
    @pytest.mark.asyncio()
    async def test_shared_session_commits_atomically_on_teardown(self, tmp_path):
        engine, factory = await make_db(tmp_path)
        adapter = SQLAlchemyAdapter(session_factory=factory)
        set_run_id("run-1")

        async with adapter.get_session() as session:
            await session.execute(text("UPDATE orders SET status = 'PAID' WHERE id = 'O-1'"))
        async with adapter.get_session() as session:
            await session.execute(text("INSERT INTO orders VALUES ('O-2', 'NEW')"))

        # Nothing committed yet
        assert await read_status(adapter) == "NEW"

        await adapter.ateardown(success=True)

        assert await read_status(adapter) == "PAID"
        assert await read_status(adapter, "O-2") == "NEW"
        await engine.dispose()

    @pytest.mark.asyncio()
    async def test_teardown_failure_rolls_back_everything(self, tmp_path):
        engine, factory = await make_db(tmp_path)
        adapter = SQLAlchemyAdapter(session_factory=factory)
        set_run_id("run-1")

        async with adapter.get_session() as session:
            await session.execute(text("UPDATE orders SET status = 'PAID' WHERE id = 'O-1'"))

        await adapter.ateardown(success=False)

        assert await read_status(adapter) == "NEW"
        await engine.dispose()

    @pytest.mark.asyncio()
    async def test_step_exception_forces_rollback(self, tmp_path):
        engine, factory = await make_db(tmp_path)
        adapter = SQLAlchemyAdapter(session_factory=factory)
        set_run_id("run-1")

        async with adapter.get_session() as session:
            await session.execute(text("UPDATE orders SET status = 'PAID' WHERE id = 'O-1'"))
        with pytest.raises(RuntimeError):
            async with adapter.get_session():
                raise RuntimeError("step failed")

        await adapter.ateardown(success=True)

        assert await read_status(adapter) == "NEW"
        await engine.dispose()

    @pytest.mark.asyncio()
    async def test_aclose_finalizes_leftover_run_sessions(self, tmp_path):
        """Port mounting: aclose() commits the run session (no failure)."""
        engine, factory = await make_db(tmp_path)
        adapter = SQLAlchemyAdapter(session_factory=factory)
        set_run_id("run-1")

        async with adapter.get_session() as session:
            await session.execute(text("UPDATE orders SET status = 'PAID' WHERE id = 'O-1'"))

        await adapter.aclose()

        # Injected engine must not be disposed by the adapter
        assert await read_status(SQLAlchemyAdapter(session_factory=factory)) == "PAID"
        await engine.dispose()


class TestEngineLifecycle:
    @pytest.mark.asyncio()
    async def test_reuse_after_aclose_rebuilds_engine_and_factory(self, tmp_path):
        """aclose() must not leave the session factory bound to a disposed engine."""
        dsn = f"sqlite+aiosqlite:///{tmp_path}/test.db"
        setup = create_async_engine(dsn)
        async with setup.begin() as conn:
            await conn.execute(text("CREATE TABLE orders (id TEXT PRIMARY KEY, status TEXT)"))
            await conn.execute(text("INSERT INTO orders VALUES ('O-1', 'NEW')"))
        await setup.dispose()

        adapter = SQLAlchemyAdapter(dsn=dsn)
        assert await read_status(adapter) == "NEW"

        await adapter.aclose()

        # Sessions after aclose must come from a fresh engine, not the disposed one
        assert await read_status(adapter) == "NEW"
        await adapter.aclose()


class TestStateBackend:
    @pytest.mark.asyncio()
    async def test_read_and_cas_write(self, tmp_path):
        engine, factory = await make_db(tmp_path)
        adapter = SQLAlchemyAdapter(session_factory=factory)
        backend = SQLAlchemyStateBackend(adapter, tables={"order": {"table": "orders"}})

        assert await backend.aread_state("order", "O-1") == "NEW"
        await backend.awrite_state("order", "O-1", "PAID", expected="NEW")
        assert await backend.aread_state("order", "O-1") == "PAID"
        await engine.dispose()

    @pytest.mark.asyncio()
    async def test_cas_mismatch_raises_stale(self, tmp_path):
        engine, factory = await make_db(tmp_path)
        adapter = SQLAlchemyAdapter(session_factory=factory)
        backend = SQLAlchemyStateBackend(adapter, tables={"order": {"table": "orders"}})

        with pytest.raises(StaleStateError):
            await backend.awrite_state("order", "O-1", "SHIPPED", expected="PAID")
        await engine.dispose()

    @pytest.mark.asyncio()
    async def test_missing_row_raises_stale(self, tmp_path):
        engine, factory = await make_db(tmp_path)
        adapter = SQLAlchemyAdapter(session_factory=factory)
        backend = SQLAlchemyStateBackend(adapter, tables={"order": {"table": "orders"}})

        with pytest.raises(StaleStateError):
            await backend.awrite_state("order", "missing", "PAID")
        await engine.dispose()

    @pytest.mark.asyncio()
    async def test_unmapped_entity_type_raises(self, tmp_path):
        engine, factory = await make_db(tmp_path)
        adapter = SQLAlchemyAdapter(session_factory=factory)
        backend = SQLAlchemyStateBackend(adapter, tables={"order": {"table": "orders"}})

        with pytest.raises(KeyError, match="No table mapping"):
            await backend.aread_state("ticket", "T-1")
        await engine.dispose()


class TestEntityStateIntegration:
    """The full stack: transition + status column in one transaction."""

    def _entity_state(self, adapter: SQLAlchemyAdapter) -> EntityState:
        backend = SQLAlchemyStateBackend(adapter, tables={"order": {"table": "orders"}})
        es = EntityState(state_backend=backend)
        es.register_machine(
            StateMachineConfig(
                entity_type="order",
                states={"NEW", "PAID", "SHIPPED"},
                initial_state="NEW",
                transitions={"NEW": {"PAID"}, "PAID": {"SHIPPED"}},
            )
        )
        return es

    @pytest.mark.asyncio()
    async def test_transition_updates_status_column(self, tmp_path):
        engine, factory = await make_db(tmp_path)
        adapter = SQLAlchemyAdapter(session_factory=factory)
        es = self._entity_state(adapter)

        result = await es.atransition("order", "O-1", "PAID")

        assert result["from_state"] == "NEW"
        assert await read_status(adapter) == "PAID"
        await engine.dispose()

    @pytest.mark.asyncio()
    async def test_pipeline_rollback_reverts_transition(self, tmp_path):
        """Failed run: the state transition rolls back with the rest."""
        engine, factory = await make_db(tmp_path)
        adapter = SQLAlchemyAdapter(session_factory=factory)
        es = self._entity_state(adapter)
        set_run_id("run-1")

        await es.atransition("order", "O-1", "PAID")
        await adapter.ateardown(success=False)

        assert await read_status(adapter) == "NEW"
        await engine.dispose()

    @pytest.mark.asyncio()
    async def test_handler_write_joins_run_transaction(self, tmp_path):
        """Handler side effects commit/roll back with the transition."""
        engine, factory = await make_db(tmp_path)
        adapter = SQLAlchemyAdapter(session_factory=factory)
        es = self._entity_state(adapter)
        set_run_id("run-1")

        async def handler(entity_id, to_state, payload=None, **kwargs):
            async with adapter.get_session() as session:
                await session.execute(
                    text("INSERT INTO orders VALUES (:id, :status)"),
                    {"id": f"audit-{entity_id}", "status": to_state},
                )

        es.register_handler("order", handler)
        await es.atransition("order", "O-1", "PAID", payload={"by": "user-1"})
        await adapter.ateardown(success=True)

        assert await read_status(adapter) == "PAID"
        assert await read_status(adapter, "audit-O-1") == "PAID"
        await engine.dispose()
