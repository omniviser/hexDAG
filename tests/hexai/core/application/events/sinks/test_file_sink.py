import asyncio
import contextlib
import json
from pathlib import Path

import pytest
import websockets

from hexai.core.application.events.sinks.file_sink import (
    FileSinkError,
    FileSinkObserver,
)
from hexai.core.application.events.sinks.websocket_sink import (
    WebSocketSinkError,
    WebSocketSinkObserver,
)
from hexdag.core.orchestration.events.events import (
    NodeStarted,
    PipelineStarted,
)


@pytest.mark.asyncio
async def test_file_sink_jsonl_roundtrip(tmp_path: Path):
    """
    Verify that FileSinkObserver:
    - writes exactly one JSON object per line (JSONL),
    - flushes immediately so the file is non-empty after handle(),
    - includes type, envelope and ISO-8601 timestamp.
    """
    path = tmp_path / "events.jsonl"
    obs = FileSinkObserver(str(path))

    # Even though batching is off by default, call start() to be lifecycle-consistent
    await obs.start()
    try:
        ev = PipelineStarted(name="demo", total_waves=2, total_nodes=3)
        await obs.handle(ev)

        content = path.read_text(encoding="utf-8").strip()
        assert content, "JSONL file should not be empty after handle()"
        lines = content.splitlines()
        assert len(lines) == 1, "Expected exactly one JSON line"

        data = json.loads(lines[0])
        assert data["type"] == "pipeline:started"
        assert data["envelope"] == {"pipeline": "demo"}
        assert isinstance(data.get("timestamp"), str) and "T" in data["timestamp"]
    finally:
        await obs.stop()


@pytest.mark.asyncio
async def test_file_sink_raises_in_strict_mode(tmp_path: Path):
    """
    In strict mode (raise_on_error=True), FileSinkObserver should raise FileSinkError
    if a write/flush error occurs. Here we simulate it by closing the file before handle().
    """
    path = tmp_path / "events.jsonl"
    obs = FileSinkObserver(str(path), raise_on_error=True)
    await obs.start()
    try:
        # Force a ValueError on the next write/flush by closing file early
        await obs.stop()

        with pytest.raises(FileSinkError):
            await obs.handle(PipelineStarted(name="x", total_waves=1, total_nodes=1))
    finally:
        # Ensure idempotent stop in case of exceptions
        with contextlib.suppress(Exception):
            await obs.stop()


@pytest.mark.asyncio
async def test_websocket_sink_broadcasts(unused_tcp_port: int):
    """
    Start WebSocketSinkObserver, connect a client, emit an event, and verify
    the client receives a JSON payload with type/envelope/timestamp.
    """
    host = "127.0.0.1"
    port = unused_tcp_port

    ws_obs = WebSocketSinkObserver(host=host, port=port)
    await ws_obs.start()
    try:
        uri = f"ws://{host}:{port}"
        async with websockets.connect(uri) as ws_client:
            # Give the server a brief moment to register the client
            await asyncio.sleep(0.05)

            ev = NodeStarted(name="A", wave_index=1, dependencies=["B"])
            await ws_obs.handle(ev)

            msg = await asyncio.wait_for(ws_client.recv(), timeout=1.0)
            data = json.loads(msg)

            assert data["type"] == "node:started"
            assert data["envelope"] == {"node": "A", "wave": 1}
            assert "timestamp" in data
    finally:
        await ws_obs.stop()


@pytest.mark.asyncio
async def test_websocket_sink_port_in_use(unused_tcp_port: int):
    """
    Starting a second server on the same port should raise WebSocketSinkError.
    """
    host = "127.0.0.1"
    port = unused_tcp_port

    s1 = WebSocketSinkObserver(host=host, port=port)
    await s1.start()

    s2 = WebSocketSinkObserver(host=host, port=port)
    try:
        with pytest.raises(WebSocketSinkError):
            await s2.start()
    finally:
        # Cleanup: stop both if necessary
        await s1.stop()
        # s2 may or may not have started; stop should be safe
        with contextlib.suppress(Exception):
            await s2.stop()
