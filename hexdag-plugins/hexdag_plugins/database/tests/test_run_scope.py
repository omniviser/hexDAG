"""Tests for RunScopedResource — the dual-mode resource primitive."""

import asyncio

import pytest
from hexdag.kernel.context.execution_context import set_run_id

from hexdag_plugins.database.run_scope import RunScopedResource


class FakeSession:
    def __init__(self, n: int) -> None:
        self.n = n
        self.committed = False
        self.rolled_back = False
        self.writes: list[str] = []


def make_scope():
    created: list[FakeSession] = []

    async def factory() -> FakeSession:
        session = FakeSession(len(created))
        created.append(session)
        return session

    async def finalize(session: FakeSession, success: bool) -> None:
        if success:
            session.committed = True
        else:
            session.rolled_back = True

    return RunScopedResource(factory, finalize), created


@pytest.fixture(autouse=True)
def _clear_run_id():
    set_run_id(None)
    yield
    set_run_id(None)


class TestStandaloneMode:
    @pytest.mark.asyncio()
    async def test_fresh_resource_committed_on_exit(self):
        scope, created = make_scope()

        async with scope.aget() as session:
            session.writes.append("a")

        assert len(created) == 1
        assert created[0].committed
        assert not created[0].rolled_back

    @pytest.mark.asyncio()
    async def test_rollback_on_exception(self):
        scope, created = make_scope()

        with pytest.raises(ValueError, match="boom"):
            async with scope.aget():
                raise ValueError("boom")

        assert created[0].rolled_back
        assert not created[0].committed

    @pytest.mark.asyncio()
    async def test_each_call_gets_fresh_resource(self):
        scope, created = make_scope()

        async with scope.aget():
            pass
        async with scope.aget():
            pass

        assert len(created) == 2


class TestRunScopedMode:
    @pytest.mark.asyncio()
    async def test_shared_resource_within_run(self):
        scope, created = make_scope()
        set_run_id("run-1")

        async with scope.aget() as s1:
            s1.writes.append("a")
        async with scope.aget() as s2:
            s2.writes.append("b")

        assert len(created) == 1
        assert created[0].writes == ["a", "b"]
        # Not finalized until afinalize_run
        assert not created[0].committed
        assert not created[0].rolled_back

    @pytest.mark.asyncio()
    async def test_finalize_run_commits_on_success(self):
        scope, created = make_scope()
        set_run_id("run-1")

        async with scope.aget():
            pass
        await scope.afinalize_run(success=True)

        assert created[0].committed

    @pytest.mark.asyncio()
    async def test_finalize_run_rolls_back_on_failure(self):
        scope, created = make_scope()
        set_run_id("run-1")

        async with scope.aget():
            pass
        await scope.afinalize_run(success=False)

        assert created[0].rolled_back

    @pytest.mark.asyncio()
    async def test_step_exception_forces_rollback_despite_success(self):
        """A failed step inside the run wins over ateardown(success=True)."""
        scope, created = make_scope()
        set_run_id("run-1")

        with pytest.raises(RuntimeError):
            async with scope.aget():
                raise RuntimeError("step failed")

        await scope.afinalize_run(success=True)
        assert created[0].rolled_back
        assert not created[0].committed

    @pytest.mark.asyncio()
    async def test_mark_failed_forces_rollback(self):
        scope, created = make_scope()
        set_run_id("run-1")

        async with scope.aget():
            pass
        scope.mark_failed()
        await scope.afinalize_run(success=True)

        assert created[0].rolled_back

    @pytest.mark.asyncio()
    async def test_finalize_without_use_is_noop(self):
        scope, created = make_scope()
        set_run_id("run-1")

        await scope.afinalize_run(success=True)
        assert created == []

    @pytest.mark.asyncio()
    async def test_concurrent_access_serialized(self):
        """Parallel waves share one resource; the lock serializes access."""
        scope, created = make_scope()
        set_run_id("run-1")
        in_block = 0
        max_concurrent = 0

        async def step(label: str) -> None:
            nonlocal in_block, max_concurrent
            async with scope.aget() as session:
                in_block += 1
                max_concurrent = max(max_concurrent, in_block)
                await asyncio.sleep(0.01)
                session.writes.append(label)
                in_block -= 1

        await asyncio.gather(step("a"), step("b"), step("c"))

        assert len(created) == 1
        assert max_concurrent == 1
        assert sorted(created[0].writes) == ["a", "b", "c"]

    @pytest.mark.asyncio()
    async def test_separate_runs_get_separate_resources(self):
        scope, created = make_scope()

        set_run_id("run-1")
        async with scope.aget() as s1:
            s1.writes.append("r1")

        set_run_id("run-2")
        async with scope.aget() as s2:
            s2.writes.append("r2")

        assert len(created) == 2
        await scope.afinalize_run(success=True)  # finalizes run-2
        set_run_id("run-1")
        await scope.afinalize_run(success=False)  # finalizes run-1

        assert created[1].committed
        assert created[0].rolled_back

    @pytest.mark.asyncio()
    async def test_finalize_all_rolls_back_leftovers(self):
        scope, created = make_scope()
        set_run_id("run-1")
        async with scope.aget():
            pass

        set_run_id(None)
        await scope.afinalize_all()

        assert created[0].rolled_back
