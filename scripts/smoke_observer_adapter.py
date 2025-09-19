import asyncio
from typing import Any

from hexai.core.application.events.events import PipelineStarted
from hexai.core.application.events.observer_manager import ObserverManager
from hexai.simple_events.envelope import SimpleContext
from hexai.simple_events.observer_adapter import SimpleEventEmitterObserver


async def main() -> None:
    got: list[dict[str, Any]] = []

    def sink(payload: dict[str, Any]) -> None:
        got.append(payload)

    ctx = SimpleContext(pipeline="doc-index", pipeline_run_id="run#1")
    adapter = SimpleEventEmitterObserver(sink=sink, context=ctx)

    om = ObserverManager()
    om.register(adapter)

    e = PipelineStarted(name="doc-index", total_waves=2, total_nodes=5)
    await om.notify(e)

    if len(got) != 1:
        raise RuntimeError("Smoke adapter should emit exactly one payload")

    print(got[0])


if __name__ == "__main__":
    asyncio.run(main())
