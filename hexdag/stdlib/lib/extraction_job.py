"""ExtractionJob — multi-round extraction state tracker.

Tracks extraction progress across multiple pipeline invocations. Each
round is a separate pipeline run; the ``ExtractionJob`` service persists
state between runs via ``SupportsKeyValue`` storage.

Usage in YAML::

    spec:
      services:
        extraction:
          class: hexdag.stdlib.lib.ExtractionJob
          config:
            max_rounds: 5
            required_fields: [carrier_name, policy_number, claim_amount]

Programmatic::

    job = ExtractionJob(max_rounds=5, required_fields=["name", "amount"])
    state = await job.aload_or_create("job-1", "claim", "CLM-001")
    state = await job.arecord_round("job-1", {"name": "Acme"}, "email")
    decision = await job.aevaluate_and_decide("job-1")
    # decision = {"action": "continue", "missing_fields": ["amount"], ...}
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hexdag.kernel.service import Service, step, tool

if TYPE_CHECKING:
    from hexdag.kernel.domain.extraction_state import ExtractionState
    from hexdag.kernel.ports.data_store import SupportsKeyValue

_STORAGE_PREFIX = "extraction_job:"


class ExtractionJob(Service):
    """Multi-round extraction state tracker.

    Exposed tools
    -------------
    - ``aload_or_create(job_id, entity_type, entity_id)`` — load or init
    - ``arecord_round(job_id, extracted_data, source)`` — record round
    - ``aget_missing_fields(job_id)`` — fields still needed
    - ``ais_complete(job_id)`` — check completion
    - ``amark_failed(job_id, reason)`` — mark as failed
    - ``aevaluate_and_decide(job_id)`` — decide next action
    """

    def __init__(
        self,
        storage: SupportsKeyValue | None = None,
        max_rounds: int = 5,
        required_fields: list[str] | None = None,
    ) -> None:
        self._storage = storage
        self._max_rounds = max_rounds
        self._required_fields = required_fields or []
        # In-memory cache: job_id -> ExtractionState
        self._jobs: dict[str, ExtractionState] = {}

    async def _load(self, job_id: str) -> ExtractionState | None:
        """Load state from storage or in-memory cache."""
        if job_id in self._jobs:
            return self._jobs[job_id]
        if self._storage is not None:
            from hexdag.kernel.domain.extraction_state import (
                ExtractionState,  # lazy: avoid circular import
            )

            data = await self._storage.aget(f"{_STORAGE_PREFIX}{job_id}")
            if data is not None:
                if isinstance(data, str):
                    state = ExtractionState.model_validate_json(data)
                else:
                    state = ExtractionState.model_validate(data)
                self._jobs[job_id] = state
                return state
        return None

    async def _save(self, state: ExtractionState) -> None:
        """Persist state to storage and update cache."""
        self._jobs[state.job_id] = state
        if self._storage is not None:
            await self._storage.aset(
                f"{_STORAGE_PREFIX}{state.job_id}",
                state.model_dump_json(),
            )

    @tool
    @step
    async def aload_or_create(
        self,
        job_id: str,
        entity_type: str = "",
        entity_id: str = "",
    ) -> dict[str, Any]:
        """Load an existing extraction job or create a new one.

        Args
        ----
            job_id: Unique job identifier.
            entity_type: Entity type (e.g., "claim").
            entity_id: Entity identifier.

        Returns
        -------
            Dict with job state including extracted_data, missing_fields, status.
        """
        from hexdag.kernel.domain.extraction_state import (
            ExtractionState,  # lazy: avoid circular import
        )

        state = await self._load(job_id)
        if state is None:
            state = ExtractionState(
                job_id=job_id,
                entity_type=entity_type,
                entity_id=entity_id,
                max_rounds=self._max_rounds,
                required_fields=list(self._required_fields),
            )
            await self._save(state)

        return state.model_dump()

    @tool
    @step
    async def arecord_round(
        self,
        job_id: str,
        extracted_data: dict[str, Any],
        source: str = "",
    ) -> dict[str, Any]:
        """Record an extraction round — merge new fields into accumulated data.

        Args
        ----
            job_id: Job identifier.
            extracted_data: Fields extracted in this round.
            source: Description of data source (e.g., "carrier_email").

        Returns
        -------
            Updated job state.
        """
        from hexdag.kernel.domain.extraction_state import RoundRecord  # lazy: avoid circular import

        state = await self._load(job_id)
        if state is None:
            msg = f"Extraction job '{job_id}' not found"
            raise ValueError(msg)

        # Merge extracted data (new non-None values overwrite old)
        merged = dict(state.extracted_data)
        merged.update({k: v for k, v in extracted_data.items() if v is not None})

        # Create round record
        round_record = RoundRecord(
            round_number=state.current_round + 1,
            extracted_fields=extracted_data,
            source=source,
        )

        # Update state
        state = state.model_copy(
            update={
                "extracted_data": merged,
                "current_round": state.current_round + 1,
                "status": "extracting",
                "round_history": [*state.round_history, round_record],
            }
        )

        await self._save(state)
        return state.model_dump()

    @tool
    async def aget_missing_fields(self, job_id: str) -> list[str]:
        """Get fields still needed for extraction completion.

        Args
        ----
            job_id: Job identifier.

        Returns
        -------
            List of missing field names.
        """
        state = await self._load(job_id)
        if state is None:
            return list(self._required_fields)
        return state.missing_fields

    @tool
    async def ais_complete(self, job_id: str) -> bool:
        """Check if all required fields have been extracted.

        Args
        ----
            job_id: Job identifier.

        Returns
        -------
            True if all required fields are present.
        """
        state = await self._load(job_id)
        if state is None:
            return False
        return state.is_complete

    @tool
    async def amark_failed(self, job_id: str, reason: str = "") -> dict[str, Any]:
        """Mark an extraction job as failed.

        Args
        ----
            job_id: Job identifier.
            reason: Failure reason.

        Returns
        -------
            Updated job state.
        """
        state = await self._load(job_id)
        if state is None:
            msg = f"Extraction job '{job_id}' not found"
            raise ValueError(msg)

        state = state.model_copy(
            update={
                "status": "failed",
                "failure_reason": reason,
            }
        )
        await self._save(state)
        return state.model_dump()

    @step
    async def aevaluate_and_decide(self, job_id: str) -> dict[str, Any]:
        """Evaluate extraction progress and decide next action.

        Returns
        -------
            Dict with ``action`` ("complete", "continue", or "fail") and
            supporting data (missing_fields, extracted_data, round_count).
        """
        state = await self._load(job_id)
        if state is None:
            return {"action": "fail", "reason": f"Job '{job_id}' not found"}

        if state.is_complete:
            state = state.model_copy(update={"status": "complete"})
            await self._save(state)
            return {
                "action": "complete",
                "extracted_data": state.extracted_data,
                "round_count": state.current_round,
            }

        if state.is_max_rounds_reached:
            state = state.model_copy(
                update={
                    "status": "failed",
                    "failure_reason": f"Max rounds ({state.max_rounds}) reached",
                }
            )
            await self._save(state)
            return {
                "action": "fail",
                "reason": f"Max rounds ({state.max_rounds}) reached",
                "missing_fields": state.missing_fields,
                "extracted_data": state.extracted_data,
            }

        return {
            "action": "continue",
            "missing_fields": state.missing_fields,
            "extracted_data": state.extracted_data,
            "round_count": state.current_round,
            "rounds_remaining": state.max_rounds - state.current_round,
        }
