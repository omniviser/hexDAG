import asyncio
from contextlib import suppress

from hexai.core.application.events.events import (
    NodeStarted,
    PipelineStarted,
    WaveCompleted,
    WaveStarted,
)
from hexai.core.application.events.sinks.file_sink import FileSinkObserver
from hexai.core.application.events.sinks.websocket_sink import WebSocketSinkObserver


async def main():
    file_obs = FileSinkObserver("./events.jsonl")
    ws_obs = WebSocketSinkObserver(host="127.0.0.1", port=8081)
    await ws_obs.start()

    # One-time "pipeline started"
    await file_obs.handle(PipelineStarted(name="demo", total_waves=3, total_nodes=2))
    await ws_obs.handle(PipelineStarted(name="demo", total_waves=3, total_nodes=2))

    i = 0
    try:
        # Continuous demo stream (Ctrl+C to stop)
        while True:
            wave_idx = (i % 3) + 1
            await ws_obs.handle(WaveStarted(wave_index=wave_idx, nodes=["A", "B"]))
            await file_obs.handle(WaveStarted(wave_index=wave_idx, nodes=["A", "B"]))

            # Alternate node event types to have variety for filtering
            if i % 2 == 0:
                await ws_obs.handle(NodeStarted(name="A", wave_index=wave_idx, dependencies=["B"]))
                await file_obs.handle(
                    NodeStarted(name="A", wave_index=wave_idx, dependencies=["B"])
                )
            else:
                await ws_obs.handle(WaveCompleted(wave_index=wave_idx, duration_ms=123.0))
                await file_obs.handle(WaveCompleted(wave_index=wave_idx, duration_ms=123.0))

            i += 1
            await asyncio.sleep(1.0)
    finally:
        file_obs.close()
        await ws_obs.stop()


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        asyncio.run(main())
