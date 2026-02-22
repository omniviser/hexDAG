"""Tests for MockHttpClient."""

from __future__ import annotations

import pytest

from hexdag.kernel.ports.api_call import APICall
from hexdag.stdlib.adapters.mock.mock_http import MockHttpClient

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_protocol_conformance() -> None:
    """MockHttpClient satisfies the APICall protocol."""
    mock = MockHttpClient()
    assert isinstance(mock, APICall)


# ---------------------------------------------------------------------------
# Default response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_response() -> None:
    mock = MockHttpClient()
    result = await mock.aget("https://example.com/test")

    assert result["status_code"] == 200
    assert result["headers"] == {}
    assert result["body"] == {}


# ---------------------------------------------------------------------------
# Custom responses
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_response() -> None:
    mock = MockHttpClient(responses={"body": {"users": [1, 2, 3]}})
    result = await mock.aget("https://example.com/users")

    assert result["body"] == {"users": [1, 2, 3]}


@pytest.mark.asyncio
async def test_multiple_responses_cycle() -> None:
    mock = MockHttpClient(
        responses=[
            {"body": {"id": 1}, "status_code": 201},
            {"body": {"id": 2}, "status_code": 201},
        ]
    )

    r1 = await mock.apost("/items", json={"name": "a"})
    r2 = await mock.apost("/items", json={"name": "b"})

    assert r1["body"]["id"] == 1
    assert r2["body"]["id"] == 2


@pytest.mark.asyncio
async def test_repeats_last_response() -> None:
    mock = MockHttpClient(responses=[{"body": {"msg": "only one"}}])

    r1 = await mock.aget("/a")
    r2 = await mock.aget("/b")
    r3 = await mock.aget("/c")

    assert r1["body"]["msg"] == "only one"
    assert r2["body"]["msg"] == "only one"
    assert r3["body"]["msg"] == "only one"


@pytest.mark.asyncio
async def test_custom_status_code() -> None:
    mock = MockHttpClient(status_code=404)
    result = await mock.aget("/missing")

    assert result["status_code"] == 404


@pytest.mark.asyncio
async def test_response_overrides_default_status() -> None:
    mock = MockHttpClient(
        responses=[{"status_code": 500, "body": {"error": "fail"}}],
        status_code=200,
    )
    result = await mock.aget("/fail")

    assert result["status_code"] == 500


# ---------------------------------------------------------------------------
# Request recording
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_records_get() -> None:
    mock = MockHttpClient()
    await mock.aget("https://api.example.com/users", params={"limit": "10"})

    assert len(mock.requests) == 1
    req = mock.requests[0]
    assert req.method == "GET"
    assert req.url == "https://api.example.com/users"
    assert req.params == {"limit": "10"}


@pytest.mark.asyncio
async def test_records_post() -> None:
    mock = MockHttpClient()
    await mock.apost("/items", json={"name": "widget"}, headers={"X-Key": "val"})

    req = mock.requests[0]
    assert req.method == "POST"
    assert req.json == {"name": "widget"}
    assert req.headers == {"X-Key": "val"}


@pytest.mark.asyncio
async def test_records_put() -> None:
    mock = MockHttpClient()
    await mock.aput("/items/1", json={"name": "updated"})

    req = mock.requests[0]
    assert req.method == "PUT"
    assert req.json == {"name": "updated"}


@pytest.mark.asyncio
async def test_records_delete() -> None:
    mock = MockHttpClient()
    await mock.adelete("/items/1", headers={"Authorization": "Bearer token"})

    req = mock.requests[0]
    assert req.method == "DELETE"
    assert req.headers == {"Authorization": "Bearer token"}


@pytest.mark.asyncio
async def test_records_generic_request() -> None:
    mock = MockHttpClient()
    await mock.arequest("PATCH", "/items/1")

    req = mock.requests[0]
    assert req.method == "PATCH"
    assert req.url == "/items/1"


@pytest.mark.asyncio
async def test_call_count() -> None:
    mock = MockHttpClient()
    await mock.aget("/a")
    await mock.apost("/b")
    await mock.adelete("/c")

    assert mock.call_count == 3
    assert len(mock.requests) == 3


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reset() -> None:
    mock = MockHttpClient()
    await mock.aget("/a")
    await mock.apost("/b")

    mock.reset()

    assert mock.call_count == 0
    assert len(mock.requests) == 0


# ---------------------------------------------------------------------------
# Detection integration
# ---------------------------------------------------------------------------


def test_port_detection() -> None:
    """detect_port_type correctly identifies MockHttpClient as api_call."""
    from hexdag.kernel.ports.detection import detect_port_type

    assert detect_port_type(MockHttpClient) == "api_call"
