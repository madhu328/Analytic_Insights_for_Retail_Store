"""EventEmitter: builds schema-valid events and writes JSONL.

Events are buffered so that tracks later classified as staff (present for most of the
clip / standing behind the counter) are flagged ``is_staff=true`` before being written.
"""
from __future__ import annotations

import json
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

EVENT_TYPES = {"ENTRY", "EXIT", "REENTRY", "ZONE_ENTER", "ZONE_EXIT", "ZONE_DWELL",
               "BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON"}


class EventEmitter:
    def __init__(self, store_id: str, camera_id: str, out_path: Optional[str | Path] = None):
        self.store_id, self.camera_id = store_id, camera_id
        self.out_path = Path(out_path) if out_path else None
        self.events: list[dict] = []
        self._seq: dict[str, int] = defaultdict(int)
        self.staff: set[str] = set()

    def emit(self, event_type: str, visitor_id: str, ts: datetime, *, zone_id: Optional[str] = None,
             dwell_ms: int = 0, confidence: float = 0.85, queue_depth: Optional[int] = None,
             sku_zone: Optional[str] = None, extra: Optional[dict] = None) -> dict:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event type {event_type}")
        if event_type.startswith("ZONE_") and not zone_id:
            raise ValueError(f"{event_type} needs zone_id")
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        self._seq[visitor_id] += 1
        ev = {
            "event_id": str(uuid.uuid4()), "store_id": self.store_id, "camera_id": self.camera_id,
            "visitor_id": visitor_id, "event_type": event_type,
            "timestamp": ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "zone_id": zone_id, "dwell_ms": int(max(0, dwell_ms)), "is_staff": False,
            "confidence": round(float(min(1.0, max(0.0, confidence))), 3),
            "metadata": {"queue_depth": queue_depth, "sku_zone": sku_zone,
                         "session_seq": self._seq[visitor_id], **(extra or {})},
        }
        self.events.append(ev)
        return ev

    def mark_staff(self, visitor_ids: set[str]) -> None:
        self.staff |= set(visitor_ids)

    def flush(self) -> list[dict]:
        for e in self.events:
            if e["visitor_id"] in self.staff:
                e["is_staff"] = True
        self.events.sort(key=lambda e: e["timestamp"])
        if self.out_path:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            with self.out_path.open("w") as fh:
                for e in self.events:
                    fh.write(json.dumps(e) + "\n")
        return self.events
