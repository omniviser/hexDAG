"""Tests for HttpClientDriver."""

from __future__ import annotations

import httpx
import pytest

from hexdag.drivers.http_client.http_client import HttpClientDriver, HttpClientError
from hexdag.kernel.ports.api_call import APICall

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_protocol_conformance() -> None:
    """HttpClientDriver satisfies the APICall protocol."""
    driver = HttpClientDriver()
    assert isinstance(driver, APICall)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_response(body: dict, status: int = 200) -> httpx.Response:
    """Build a mock httpx.Response with JSON body."""
    return httpx.Response(
        status_code=status,
        json=body,
        request=httpx.Request("GET", "https://test.example.com"),
    )


def _text_response(text: str, status: int = 200) -> httpx.Response:
    """Build a mock httpx.Response with text body."""
    return httpx.Response(
        status_code=status,
        text=text,
        headers={"content-type": "text/plain"},
        request=httpx.Request("GET", "https://test.example.com"),
    )


class MockTransport(httpx.AsyncBaseTransport):
    """Transport that returns a pre-configured response."""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.last_request = request
        return self._response


class RecordingTransport(httpx.AsyncBaseTransport):
    """Transport that records requests and returns configured responses."""

    def __init__(self, responses: list[httpx.Response] | None = None) -> None:
        self._responses = responses or [_json_response({"ok": True})]
        self._call_count = 0
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]


def _make_driver(
    transport: httpx.AsyncBaseTransport,
    **kwargs: object,
) -> HttpClientDriver:
    """Create a driver with an injected transport (no real HTTP calls)."""
    driver = HttpClientDriver(**kwargs)
    driver._transport = transport
    return driver


# ---------------------------------------------------------------------------
# GET
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_json() -> None:
    transport = MockTransport(_json_response({"users": [1, 2]}))
    driver = _make_driver(transport)

    result = await driver.aget("https://api.example.com/users")

    assert result["status_code"] == 200
    assert result["body"] == {"users": [1, 2]}
    assert "content-type" in result["headers"]
    await driver.aclose()


@pytest.mark.asyncio
async def test_get_with_params() -> None:
    transport = RecordingTransport()
    driver = _make_driver(transport)

    await driver.aget("https://api.example.com/search", params={"q": "test"})

    assert b"q=test" in transport.requests[0].url.raw_path
    await driver.aclose()


@pytest.mark.asyncio
async def test_get_text_response() -> None:
    transport = MockTransport(_text_response("hello world"))
    driver = _make_driver(transport)

    result = await driver.aget("https://api.example.com/text")

    assert result["body"] == "hello world"
    await driver.aclose()


# ---------------------------------------------------------------------------
# POST
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_json() -> None:
    transport = RecordingTransport([_json_response({"id": 42}, status=201)])
    driver = _make_driver(transport, raise_for_status=False)

    result = await driver.apost(
        "https://api.example.com/items",
        json={"name": "widget"},
    )

    assert result["status_code"] == 201
    assert result["body"] == {"id": 42}
    await driver.aclose()


# ---------------------------------------------------------------------------
# PUT
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_put() -> None:
    transport = RecordingTransport([_json_response({"updated": True})])
    driver = _make_driver(transport)

    result = await driver.aput(
        "https://api.example.com/items/1",
        json={"name": "updated"},
    )

    assert result["body"] == {"updated": True}
    assert transport.requests[0].method == "PUT"
    await driver.aclose()


# ---------------------------------------------------------------------------
# DELETE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete() -> None:
    transport = RecordingTransport([_json_response({"deleted": True})])
    driver = _make_driver(transport)

    result = await driver.adelete("https://api.example.com/items/1")

    assert result["body"] == {"deleted": True}
    assert transport.requests[0].method == "DELETE"
    await driver.aclose()


# ---------------------------------------------------------------------------
# Generic request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arequest_patch() -> None:
    transport = RecordingTransport([_json_response({"patched": True})])
    driver = _make_driver(transport)

    result = await driver.arequest(
        "PATCH",
        "https://api.example.com/items/1",
        json={"name": "patched"},
    )

    assert result["body"] == {"patched": True}
    assert transport.requests[0].method == "PATCH"
    await driver.aclose()


# ---------------------------------------------------------------------------
# Base URL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_base_url() -> None:
    transport = RecordingTransport()
    driver = _make_driver(transport, base_url="https://api.example.com")

    await driver.aget("/users")

    assert str(transport.requests[0].url) == "https://api.example.com/users"
    await driver.aclose()


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_headers() -> None:
    transport = RecordingTransport()
    driver = _make_driver(transport, headers={"X-Api-Key": "secret"})

    await driver.aget("https://api.example.com/data")

    req = transport.requests[0]
    assert req.headers.get("x-api-key") == "secret"
    await driver.aclose()


@pytest.mark.asyncio
async def test_per_request_headers_override() -> None:
    transport = RecordingTransport()
    driver = _make_driver(transport, headers={"X-Api-Key": "default"})

    await driver.aget(
        "https://api.example.com/data",
        headers={"X-Api-Key": "override"},
    )

    req = transport.requests[0]
    assert req.headers.get("x-api-key") == "override"
    await driver.aclose()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raise_for_status_4xx() -> None:
    transport = MockTransport(_json_response({"error": "not found"}, status=404))
    driver = _make_driver(transport, raise_for_status=True)

    with pytest.raises(HttpClientError) as exc_info:
        await driver.aget("https://api.example.com/missing")

    assert exc_info.value.status_code == 404
    assert exc_info.value.body == {"error": "not found"}
    await driver.aclose()


@pytest.mark.asyncio
async def test_raise_for_status_5xx() -> None:
    transport = MockTransport(_json_response({"error": "server error"}, status=500))
    driver = _make_driver(transport, raise_for_status=True)

    with pytest.raises(HttpClientError) as exc_info:
        await driver.aget("https://api.example.com/fail")

    assert exc_info.value.status_code == 500
    await driver.aclose()


@pytest.mark.asyncio
async def test_no_raise_for_status() -> None:
    transport = MockTransport(_json_response({"error": "not found"}, status=404))
    driver = _make_driver(transport, raise_for_status=False)

    result = await driver.aget("https://api.example.com/missing")

    assert result["status_code"] == 404
    assert result["body"] == {"error": "not found"}
    await driver.aclose()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lazy_client_creation() -> None:
    driver = HttpClientDriver()
    assert driver._client is None

    # Accessing the client creates it
    client = driver._get_client()
    assert client is not None
    assert driver._client is client
    await driver.aclose()


@pytest.mark.asyncio
async def test_aclose_idempotent() -> None:
    driver = HttpClientDriver()
    # Close without ever creating a client — should be a no-op
    await driver.aclose()
    assert driver._client is None


@pytest.mark.asyncio
async def test_aclose_releases_client() -> None:
    transport = RecordingTransport()
    driver = _make_driver(transport)

    await driver.aget("https://api.example.com/data")
    assert driver._client is not None

    await driver.aclose()
    assert driver._client is None


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check() -> None:
    driver = HttpClientDriver(base_url="https://api.example.com")
    result = await driver.ahealth_check()

    assert result["status"] == "healthy"
    assert result["adapter_name"] == "HttpClientDriver"
    assert result["base_url"] == "https://api.example.com"
    await driver.aclose()
