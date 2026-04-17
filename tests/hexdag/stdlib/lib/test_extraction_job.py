"""Tests for ExtractionJob service and ExtractionState domain model."""

import pytest

from hexdag.kernel.domain.extraction_state import ExtractionState, RoundRecord
from hexdag.kernel.service import Service
from hexdag.stdlib.lib.extraction_job import ExtractionJob


class TestExtractionState:
    def test_missing_fields(self):
        state = ExtractionState(
            job_id="j1",
            required_fields=["name", "amount", "date"],
            extracted_data={"name": "Acme"},
        )
        assert state.missing_fields == ["amount", "date"]

    def test_is_complete(self):
        state = ExtractionState(
            job_id="j1",
            required_fields=["name", "amount"],
            extracted_data={"name": "Acme", "amount": 100},
        )
        assert state.is_complete is True

    def test_not_complete(self):
        state = ExtractionState(
            job_id="j1",
            required_fields=["name", "amount"],
            extracted_data={"name": "Acme"},
        )
        assert state.is_complete is False

    def test_none_values_count_as_missing(self):
        state = ExtractionState(
            job_id="j1",
            required_fields=["name"],
            extracted_data={"name": None},
        )
        assert state.missing_fields == ["name"]

    def test_max_rounds_reached(self):
        state = ExtractionState(job_id="j1", max_rounds=3, current_round=3)
        assert state.is_max_rounds_reached is True

    def test_max_rounds_not_reached(self):
        state = ExtractionState(job_id="j1", max_rounds=3, current_round=2)
        assert state.is_max_rounds_reached is False

    def test_serialization_roundtrip(self):
        state = ExtractionState(
            job_id="j1",
            entity_type="claim",
            entity_id="CLM-1",
            required_fields=["name"],
            extracted_data={"name": "Acme"},
            round_history=[
                RoundRecord(round_number=1, extracted_fields={"name": "Acme"}, source="email"),
            ],
        )
        json_str = state.model_dump_json()
        restored = ExtractionState.model_validate_json(json_str)
        assert restored.job_id == "j1"
        assert restored.extracted_data == {"name": "Acme"}
        assert len(restored.round_history) == 1


class TestExtractionJobService:
    def test_is_service(self):
        assert issubclass(ExtractionJob, Service)

    def test_has_tools(self):
        job = ExtractionJob(required_fields=["name"])
        tools = job.get_tools()
        assert "aload_or_create" in tools
        assert "arecord_round" in tools
        assert "aget_missing_fields" in tools
        assert "ais_complete" in tools
        assert "amark_failed" in tools

    def test_has_steps(self):
        job = ExtractionJob(required_fields=["name"])
        steps = job.get_steps()
        assert "aload_or_create" in steps
        assert "arecord_round" in steps
        assert "aevaluate_and_decide" in steps


class TestExtractionJobOperations:
    @pytest.fixture()
    def job(self):
        return ExtractionJob(
            max_rounds=3,
            required_fields=["carrier_name", "policy_number", "claim_amount"],
        )

    @pytest.mark.asyncio()
    async def test_load_or_create_new(self, job):
        state = await job.aload_or_create("j1", "claim", "CLM-1")
        assert state["job_id"] == "j1"
        assert state["entity_type"] == "claim"
        assert state["status"] == "pending"
        assert state["current_round"] == 0

    @pytest.mark.asyncio()
    async def test_load_or_create_existing(self, job):
        await job.aload_or_create("j1", "claim", "CLM-1")
        state = await job.aload_or_create("j1")
        assert state["entity_type"] == "claim"  # Preserved from first call

    @pytest.mark.asyncio()
    async def test_record_round(self, job):
        await job.aload_or_create("j1", "claim", "CLM-1")
        state = await job.arecord_round("j1", {"carrier_name": "Acme"}, "email_1")
        assert state["current_round"] == 1
        assert state["status"] == "extracting"
        assert state["extracted_data"]["carrier_name"] == "Acme"
        assert len(state["round_history"]) == 1

    @pytest.mark.asyncio()
    async def test_multi_round_accumulation(self, job):
        await job.aload_or_create("j1", "claim", "CLM-1")
        await job.arecord_round("j1", {"carrier_name": "Acme"}, "email_1")
        state = await job.arecord_round("j1", {"policy_number": "POL-123"}, "email_2")
        assert state["current_round"] == 2
        assert state["extracted_data"]["carrier_name"] == "Acme"
        assert state["extracted_data"]["policy_number"] == "POL-123"

    @pytest.mark.asyncio()
    async def test_none_values_not_merged(self, job):
        await job.aload_or_create("j1", "claim", "CLM-1")
        await job.arecord_round("j1", {"carrier_name": "Acme"}, "email_1")
        state = await job.arecord_round(
            "j1", {"carrier_name": None, "policy_number": "P1"}, "email_2"
        )
        assert state["extracted_data"]["carrier_name"] == "Acme"  # Not overwritten by None

    @pytest.mark.asyncio()
    async def test_get_missing_fields(self, job):
        await job.aload_or_create("j1", "claim", "CLM-1")
        await job.arecord_round("j1", {"carrier_name": "Acme"}, "email_1")
        missing = await job.aget_missing_fields("j1")
        assert "policy_number" in missing
        assert "claim_amount" in missing
        assert "carrier_name" not in missing

    @pytest.mark.asyncio()
    async def test_is_complete(self, job):
        await job.aload_or_create("j1", "claim", "CLM-1")
        assert await job.ais_complete("j1") is False
        await job.arecord_round(
            "j1",
            {
                "carrier_name": "Acme",
                "policy_number": "P1",
                "claim_amount": 5000,
            },
            "email_1",
        )
        assert await job.ais_complete("j1") is True

    @pytest.mark.asyncio()
    async def test_mark_failed(self, job):
        await job.aload_or_create("j1", "claim", "CLM-1")
        state = await job.amark_failed("j1", "Carrier unresponsive")
        assert state["status"] == "failed"
        assert state["failure_reason"] == "Carrier unresponsive"

    @pytest.mark.asyncio()
    async def test_evaluate_complete(self, job):
        await job.aload_or_create("j1", "claim", "CLM-1")
        await job.arecord_round(
            "j1",
            {
                "carrier_name": "Acme",
                "policy_number": "P1",
                "claim_amount": 5000,
            },
            "email_1",
        )
        decision = await job.aevaluate_and_decide("j1")
        assert decision["action"] == "complete"
        assert decision["round_count"] == 1

    @pytest.mark.asyncio()
    async def test_evaluate_continue(self, job):
        await job.aload_or_create("j1", "claim", "CLM-1")
        await job.arecord_round("j1", {"carrier_name": "Acme"}, "email_1")
        decision = await job.aevaluate_and_decide("j1")
        assert decision["action"] == "continue"
        assert "policy_number" in decision["missing_fields"]
        assert decision["rounds_remaining"] == 2

    @pytest.mark.asyncio()
    async def test_evaluate_fail_max_rounds(self, job):
        await job.aload_or_create("j1", "claim", "CLM-1")
        await job.arecord_round("j1", {"carrier_name": "Acme"}, "r1")
        await job.arecord_round("j1", {}, "r2")
        await job.arecord_round("j1", {}, "r3")
        decision = await job.aevaluate_and_decide("j1")
        assert decision["action"] == "fail"
        assert "Max rounds" in decision["reason"]

    @pytest.mark.asyncio()
    async def test_record_round_not_found(self, job):
        with pytest.raises(ValueError, match="not found"):
            await job.arecord_round("nonexistent", {}, "")
