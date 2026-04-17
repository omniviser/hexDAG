"""Domain models for multi-round extraction state tracking.

Used by :class:`ExtractionJob <hexdag.stdlib.lib.extraction_job.ExtractionJob>`
to track extraction progress across multiple pipeline invocations.

Each extraction round is a separate pipeline run. The ``ExtractionState``
persists between runs via ``SupportsKeyValue`` storage, tracking which
fields have been extracted and which are still needed.

OS metaphor: like a process image saved to disk (SIGSTOP + core dump)
and resumed later (SIGCONT + restore).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, computed_field


class RoundRecord(BaseModel):
    """Snapshot of a single extraction round.

    Attributes
    ----------
    round_number : int
        1-based round index.
    extracted_fields : dict[str, Any]
        Fields successfully extracted in this round.
    source : str
        Description of the data source (e.g., "carrier_email", "api_response").
    timestamp : datetime
        When this round was executed.
    raw_data : dict[str, Any]
        Raw data received (for audit/debugging).
    """

    round_number: int
    extracted_fields: dict[str, Any] = Field(default_factory=dict)
    source: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    raw_data: dict[str, Any] = Field(default_factory=dict)


class ExtractionState(BaseModel):
    """Tracks multi-round extraction progress for a single entity.

    Accumulates extracted data across rounds and computes which required
    fields are still missing.

    Attributes
    ----------
    job_id : str
        Unique extraction job identifier.
    entity_type : str
        Type of entity being extracted (e.g., "claim", "order").
    entity_id : str
        Unique entity identifier.
    status : str
        Current status: "pending", "extracting", "complete", "failed".
    current_round : int
        Number of rounds completed so far.
    max_rounds : int
        Maximum allowed extraction rounds before auto-failure.
    required_fields : list[str]
        Fields that must be extracted for completion.
    extracted_data : dict[str, Any]
        Accumulated extracted data across all rounds.
    round_history : list[RoundRecord]
        Ordered list of round snapshots.
    failure_reason : str | None
        Reason for failure (if status is "failed").
    """

    job_id: str
    entity_type: str = ""
    entity_id: str = ""
    status: str = "pending"
    current_round: int = 0
    max_rounds: int = 5
    required_fields: list[str] = Field(default_factory=list)
    extracted_data: dict[str, Any] = Field(default_factory=dict)
    round_history: list[RoundRecord] = Field(default_factory=list)
    failure_reason: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def missing_fields(self) -> list[str]:
        """Fields still needed for completion."""
        return [
            f
            for f in self.required_fields
            if f not in self.extracted_data or self.extracted_data[f] is None
        ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_complete(self) -> bool:
        """Whether all required fields have been extracted."""
        return len(self.missing_fields) == 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_max_rounds_reached(self) -> bool:
        """Whether the maximum number of rounds has been reached."""
        return self.current_round >= self.max_rounds
