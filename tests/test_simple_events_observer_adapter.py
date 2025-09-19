import asyncio

from hexai.core.application.events.events import PipelineStarted
from hexai.core.application.events.observer_manager import ObserverManager

from hexai.simple_events.envelope import SimpleContext
from hexai.simple_events.observer_adapter import SimpleEventEmitterObserver

def test_observer_adapter_emits_envelope():
    got = []
    sink = lambda payload: got.append(payload)

    ctx = SimpleContext(pipeline="doc-index", pipeline_run_id="run#1")
    adapter = SimpleEventEmitterObserver(sink=sink, context=ctx)

    om = ObserverManager()
    om.register(adapter)

    e = PipelineStarted(name="doc-index", total_waves=2, total_nodes=5)

    async def run():
        await om.notify(e)

    asyncio.run(run())

    assert len(got) == 1
    env = got[0]
    assert env["event_type"] == "pipeline:started"
    assert env["pipeline"] == "doc-index"
    assert env["pipeline_run_id"] == "run#1"
    assert env["attrs"] == {"total_waves": 2, "total_nodes": 5}
