"""SQLite (or any SQLAlchemy URL) storage: raw events + POS transactions.

Raw events are the source of truth; sessions and KPIs are derived at query time
(event-sourced => auditable and replayable). Ingestion is idempotent on event_id /
transaction_id.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

from sqlalchemy import (Boolean, Column, DateTime, Float, Index, Integer, MetaData, String,
                        Table, Text, create_engine, event, func, select)
from sqlalchemy.engine import Engine

from .config import settings
from .schemas import Event, PosTransaction

metadata = MetaData()

events = Table(
    "events", metadata,
    Column("event_id", String(36), primary_key=True),
    Column("store_id", String(64), nullable=False, index=True),
    Column("camera_id", String(64), nullable=False),
    Column("visitor_id", String(64), nullable=False, index=True),
    Column("event_type", String(32), nullable=False, index=True),
    Column("timestamp", DateTime, nullable=False),
    Column("zone_id", String(64), index=True),
    Column("dwell_ms", Integer, nullable=False, default=0),
    Column("is_staff", Boolean, nullable=False, default=False),
    Column("confidence", Float, nullable=False),
    Column("metadata_json", Text, nullable=False, default="{}"),
    Column("ingested_at", DateTime, nullable=False),
    Index("ix_events_store_timestamp", "store_id", "timestamp"),
    Index("ix_events_store_type_timestamp", "store_id", "event_type", "timestamp"),
)

pos_transactions = Table(
    "pos_transactions", metadata,
    Column("store_id", String(64), primary_key=True),
    Column("transaction_id", String(64), primary_key=True),
    Column("timestamp", DateTime, nullable=False),
    Column("basket_value_inr", Float, nullable=False),
    Column("ingested_at", DateTime, nullable=False),
    Index("ix_pos_store_timestamp", "store_id", "timestamp"),
)


@lru_cache(maxsize=8)
def get_engine(url: Optional[str] = None) -> Engine:
    url = url or settings.database_url
    kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
    eng = create_engine(url, future=True, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(eng, "connect")
        def _pragma(dbapi_conn, _):  # pragma: no cover - trivial
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()
    metadata.create_all(eng)
    return eng


def _chunks(seq: list, n: int = 500):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def insert_events(engine: Engine, items: list[Event]) -> tuple[int, int]:
    """Insert events, skipping ids already stored. Returns (inserted, duplicates)."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    uniq: dict[str, Event] = {}
    for e in items:
        uniq.setdefault(str(e.event_id), e)
    inserted = 0
    with engine.begin() as conn:
        for chunk in _chunks(list(uniq.values())):
            ids = [str(e.event_id) for e in chunk]
            existing = set(conn.execute(select(events.c.event_id).where(events.c.event_id.in_(ids))).scalars())
            rows = [{
                "event_id": str(e.event_id), "store_id": e.store_id, "camera_id": e.camera_id,
                "visitor_id": e.visitor_id, "event_type": e.event_type.value, "timestamp": e.timestamp,
                "zone_id": e.zone_id, "dwell_ms": e.dwell_ms, "is_staff": e.is_staff,
                "confidence": e.confidence,
                "metadata_json": json.dumps(e.metadata.model_dump(exclude_none=True)),
                "ingested_at": now,
            } for e in chunk if str(e.event_id) not in existing]
            if rows:
                conn.execute(events.insert(), rows)
                inserted += len(rows)
    return inserted, len(items) - inserted


def insert_pos(engine: Engine, items: list[PosTransaction]) -> tuple[int, int]:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    uniq: dict[tuple[str, str], PosTransaction] = {}
    for t in items:
        uniq.setdefault((t.store_id, t.transaction_id), t)
    inserted = 0
    with engine.begin() as conn:
        for chunk in _chunks(list(uniq.values())):
            ids = [t.transaction_id for t in chunk]
            existing = set(conn.execute(
                select(pos_transactions.c.store_id, pos_transactions.c.transaction_id)
                .where(pos_transactions.c.transaction_id.in_(ids))).all())
            rows = [{**t.model_dump(), "ingested_at": now} for t in chunk
                    if (t.store_id, t.transaction_id) not in existing]
            if rows:
                conn.execute(pos_transactions.insert(), rows)
                inserted += len(rows)
    return inserted, len(items) - inserted


def fetch_events(engine: Engine, store_id: str, start: Optional[datetime] = None,
                 end: Optional[datetime] = None) -> list[dict]:
    q = select(events).where(events.c.store_id == store_id)
    if start:
        q = q.where(events.c.timestamp >= start)
    if end:
        q = q.where(events.c.timestamp < end)
    q = q.order_by(events.c.timestamp, events.c.event_id)
    with engine.connect() as conn:
        out = []
        for r in conn.execute(q).mappings():
            d = dict(r)
            d["metadata"] = json.loads(d.pop("metadata_json") or "{}")
            out.append(d)
        return out


def fetch_pos(engine: Engine, store_id: str, start: Optional[datetime] = None,
              end: Optional[datetime] = None) -> list[dict]:
    q = select(pos_transactions).where(pos_transactions.c.store_id == store_id)
    if start:
        q = q.where(pos_transactions.c.timestamp >= start)
    if end:
        q = q.where(pos_transactions.c.timestamp < end)
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(q.order_by(pos_transactions.c.timestamp)).mappings()]


def store_summaries(engine: Engine) -> list[dict]:
    with engine.connect() as conn:
        ev = conn.execute(select(events.c.store_id, func.count(), func.min(events.c.timestamp),
                                 func.max(events.c.timestamp)).group_by(events.c.store_id)).all()
        pos = conn.execute(select(pos_transactions.c.store_id, func.count(),
                                  func.min(pos_transactions.c.timestamp),
                                  func.max(pos_transactions.c.timestamp))
                           .group_by(pos_transactions.c.store_id)).all()
        days = conn.execute(select(events.c.store_id, func.date(events.c.timestamp)).distinct()).all()
        pdays = conn.execute(select(pos_transactions.c.store_id,
                                    func.date(pos_transactions.c.timestamp)).distinct()).all()
    out: dict[str, dict] = {}
    for sid, n, lo, hi in ev:
        out[sid] = {"store_id": sid, "events": n, "first_event": lo, "last_event": hi,
                    "transactions": 0, "dates": set()}
    for sid, n, lo, hi in pos:
        out.setdefault(sid, {"store_id": sid, "events": 0, "first_event": None, "last_event": None,
                             "dates": set()})
        out[sid]["transactions"] = n
    for sid, d in list(days) + list(pdays):
        out[sid]["dates"].add(str(d))
    for v in out.values():
        v["dates"] = sorted(v["dates"])
    return sorted(out.values(), key=lambda x: x["store_id"])
