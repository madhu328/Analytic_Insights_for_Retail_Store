"""Event / POS contracts shared by the ingestion API and analytics.

The event schema mirrors the detection pipeline output (``pipeline/emit.py``) and the
provided ``sample_events.jsonl``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EventType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    REENTRY = "REENTRY"
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_DWELL = "ZONE_DWELL"
    BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"


ZONE_EVENTS = {EventType.ZONE_ENTER, EventType.ZONE_EXIT, EventType.ZONE_DWELL}
QUEUE_EVENTS = {EventType.BILLING_QUEUE_JOIN, EventType.BILLING_QUEUE_ABANDON}


def to_naive_utc(ts: datetime) -> datetime:
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts


class EventMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")
    queue_depth: Optional[int] = Field(default=None, ge=0)
    sku_zone: Optional[str] = None
    session_seq: Optional[int] = Field(default=None, ge=0)


class Event(BaseModel):
    model_config = ConfigDict(extra="ignore", use_enum_values=False)

    event_id: UUID
    store_id: str = Field(min_length=1, max_length=64)
    camera_id: str = Field(min_length=1, max_length=64)
    visitor_id: str = Field(min_length=1, max_length=64)
    event_type: EventType
    timestamp: datetime
    zone_id: Optional[str] = Field(default=None, max_length=64)
    dwell_ms: int = Field(default=0, ge=0)
    is_staff: bool = False
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: EventMetadata = Field(default_factory=EventMetadata)

    @field_validator("timestamp")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return to_naive_utc(v)

    @field_validator("zone_id")
    @classmethod
    def _zone_upper(cls, v: Optional[str]) -> Optional[str]:
        return v.strip().upper() if v else None

    @model_validator(mode="after")
    def _zone_required(self) -> "Event":
        if self.event_type in ZONE_EVENTS and not self.zone_id:
            raise ValueError(f"{self.event_type.value} requires zone_id")
        return self


class PosTransaction(BaseModel):
    transaction_id: str = Field(min_length=1, max_length=64)
    store_id: str = Field(min_length=1, max_length=64)
    timestamp: datetime
    basket_value_inr: float = Field(ge=0)

    @field_validator("timestamp")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return to_naive_utc(v)


class IngestError(BaseModel):
    index: int
    id: Optional[str] = None
    error: str


class IngestResult(BaseModel):
    received: int
    inserted: int
    duplicates: int
    rejected: int
    errors: list[IngestError] = []


def validate_batch(model: type[BaseModel], items: list[Any], id_field: str):
    """Validate items one by one so a single bad record doesn't reject the batch."""
    ok, errors = [], []
    for i, raw in enumerate(items):
        try:
            ok.append(model.model_validate(raw))
        except Exception as exc:  # pydantic.ValidationError or TypeError
            rid = raw.get(id_field) if isinstance(raw, dict) else None
            msg = str(exc).splitlines()
            errors.append(IngestError(index=i, id=str(rid) if rid else None, error=" ".join(msg[:3])[:500]))
    return ok, errors
