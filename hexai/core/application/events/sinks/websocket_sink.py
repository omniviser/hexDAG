from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from hexdag.core.orchestration.events.batching import (
    BatchingConfig,
    EventBatchEnvelope,
    EventBatcher,
)
from hexdag.core.ports.observer_manager import Observer

from .file_sink import _event_to_dict


class WebSocketSinkError(Exception):
    """Raised when the WebSocketSinkObserver cannot start/stop or is misconfigured."""


class WebSocketSinkObserver(Observer):
    """
    Observer that runs a WebSocket server and broadcasts events to all clients.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8081,
        *,
        batching: BatchingConfig | None = None,
        keepalive_sec: float | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.keepalive_sec = keepalive_sec
        self._clients: set[Any] = set()
        self._server: Any | None = None
        self._keepalive_task: asyncio.Task[None] | None = None

        self._batching_cfg = batching
        self._batcher: EventBatcher | None = None

    async def start(self) -> None:
        """
        Start the WebSocket server and begin accepting client connections.
        """
        try:
            self._server = await websockets.serve(self._handler, self.host, self.port)
        except OSError as e:
            raise WebSocketSinkError(
                f"Failed to start WebSocket server on {self.host}:{self.port}: {e}"
            ) from e

        if self._batching_cfg is not None:
            self._batcher = EventBatcher(self._flush_envelope, self._batching_cfg)

        if self.keepalive_sec and self.keepalive_sec > 0:
            self._keepalive_task = asyncio.create_task(self._keepalive_loop(self.keepalive_sec))

    async def stop(self) -> None:
        """
        Stop the WebSocket server and close all resources gracefully.
        """
        if self._keepalive_task:
            self._keepalive_task.cancel()
            with suppress(Exception):
                await self._keepalive_task
            self._keepalive_task = None

        if self._batcher is not None:
            await self._batcher.close()
            self._batcher = None

        server = self._server
        if server is None:
            for ws in list(self._clients):
                with suppress(WebSocketException, RuntimeError, asyncio.CancelledError):
                    await ws.close()
            self._clients.clear()
            return

        self._server = None

        server.close()
        try:
            await server.wait_closed()
        except RuntimeError as e:
            raise WebSocketSinkError("Failed to stop WebSocket server cleanly") from e

        for ws in list(self._clients):
            with suppress(WebSocketException, RuntimeError, asyncio.CancelledError):
                await ws.close()
        self._clients.clear()

    async def _handler(self, connection: Any) -> None:
        """
        Per-connection handler.

        websockets >= 11 passes a single connection object to the handler.
        Keep the connection alive and concurrently drain incoming frames to avoid backpressure.
        """
        ws = connection
        self._clients.add(ws)

        async def _drain() -> None:
            try:
                async for _ in ws:
                    pass
            except (ConnectionClosed, WebSocketException, RuntimeError, asyncio.CancelledError):
                pass

        drain_task = asyncio.create_task(_drain())
        try:
            await drain_task
        finally:
            drain_task.cancel()
            with suppress(Exception):
                await drain_task
            self._clients.discard(ws)

    async def handle(self, event: Any) -> None:
        """
        Broadcast the serialized event (or batch) to all connected clients.
        """
        if not self._clients:
            return

        # Batched path: defer to batcher
        if self._batcher is not None:
            await self._batcher.add(event)
            return

        # Non-batched path: send one event per message
        payload = json.dumps(_event_to_dict(event), ensure_ascii=False)

        await asyncio.gather(
            *(self._safe_send(client, payload) for client in tuple(self._clients)),
            return_exceptions=True,
        )

    async def _flush_envelope(self, envelope: EventBatchEnvelope) -> None:
        """
        Send an array of events to all clients in a single message.
        """
        if not self._clients:
            return

        payload = json.dumps([_event_to_dict(ev) for ev in envelope.events], ensure_ascii=False)
        await asyncio.gather(
            *(self._safe_send(client, payload) for client in tuple(self._clients)),
            return_exceptions=True,
        )

    async def _safe_send(self, client: Any, msg: str) -> None:
        """
        Send a message to a single client and remove it on failure.
        """
        try:
            await client.send(msg)
        except (ConnectionClosed, WebSocketException, RuntimeError, asyncio.CancelledError):
            self._clients.discard(client)

    async def _keepalive_loop(self, interval: float) -> None:
        """
        Periodically ping clients to keep connections alive and prune dead ones.
        """
        try:
            while True:
                await asyncio.sleep(interval)
                if not self._clients:
                    continue
                await asyncio.gather(
                    *(self._safe_ping(c) for c in tuple(self._clients)),
                    return_exceptions=True,
                )
        except asyncio.CancelledError:
            return

    async def _safe_ping(self, client: Any) -> None:
        try:
            pong_waiter = await client.ping()
            await asyncio.wait_for(pong_waiter, timeout=5.0)
        except Exception:
            self._clients.discard(client)
