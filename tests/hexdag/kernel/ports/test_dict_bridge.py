"""Tests for dict_bridge_call utility."""

from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ValidationError

from hexdag.kernel.ports.dict_bridge import dict_bridge_call


class _SampleRequest(BaseModel):
    name: str
    count: int = 1


class _SampleResult(BaseModel):
    message: str
    created_at: datetime


class _FakeAdapter:
    async def process(self, request: _SampleRequest) -> _SampleResult:
        return _SampleResult(
            message=f"Hello {request.name} x{request.count}",
            created_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
        )


@pytest.mark.asyncio
class TestDictBridgeCall:
    async def test_converts_dict_to_model_and_back(self):
        adapter = _FakeAdapter()
        result = await dict_bridge_call(
            adapter.process,
            {"name": "Alice", "count": 3},
            _SampleRequest,
        )
        assert result == {
            "message": "Hello Alice x3",
            "created_at": "2025-01-15T12:00:00Z",
        }

    async def test_uses_model_defaults(self):
        adapter = _FakeAdapter()
        result = await dict_bridge_call(
            adapter.process,
            {"name": "Bob"},
            _SampleRequest,
        )
        assert result["message"] == "Hello Bob x1"

    async def test_validates_input(self):
        adapter = _FakeAdapter()
        with pytest.raises(ValidationError):
            await dict_bridge_call(
                adapter.process,
                {"wrong_field": "value"},
                _SampleRequest,
            )

    async def test_dict_result_passthrough(self):
        async def returns_dict(request: _SampleRequest) -> dict:
            return {"raw": request.name}

        result = await dict_bridge_call(
            returns_dict,
            {"name": "test"},
            _SampleRequest,
        )
        assert result == {"raw": "test"}
