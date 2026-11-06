"""Session builder: ENTRY/REENTRY opens a visit, EXIT closes it.

Zone and queue events attach only to the visitor's currently open session; anything else
is counted as an *orphan* (a data-quality signal, e.g. a missed entry-line crossing).
REENTRY opens a new session but the visitor is still counted once (visitor-level funnel).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from ..layout import billing_zone_ids

OPENERS = {"ENTRY", "REENTRY"}


@dataclass
class ZoneVisit:
    zone_id: str
    enter: datetime
    exit: Optional[datetime] = None
    dwell_ms: Optional[int] = None  # from ZONE_DWELL when available

    @property
    def dwell_s(self) -> float:
        if self.dwell_ms is not None:
            return self.dwell_ms / 1000.0
        if self.exit is not None:
            return max(0.0, (self.exit - self.enter).total_seconds())
        return 0.0


@dataclass
class Session:
    session_id: str
    visitor_id: str
    store_id: str
    start: datetime
    is_reentry: bool
    end: Optional[datetime] = None
    closed: bool = False
    close_reason: str = "open"
    last_seen: Optional[datetime] = None
    zone_visits: list[ZoneVisit] = field(default_factory=list)
    queue_joins: list[datetime] = field(default_factory=list)
    queue_abandons: list[datetime] = field(default_factory=list)
    queue_depths: list[int] = field(default_factory=list)
    billing_zone_times: list[datetime] = field(default_factory=list)
    event_count: int = 0
    min_confidence: float = 1.0
    cameras: set = field(default_factory=set)
    # filled by POS matching
    converted: bool = False
    transaction_id: Optional[str] = None
    basket_value_inr: float = 0.0

    @property
    def duration_s(self) -> float:
        end = self.end or self.last_seen or self.start
        return max(0.0, (end - self.start).total_seconds())

    @property
    def product_zones(self) -> list[str]:
        return [v.zone_id for v in self.zone_visits]

    @property
    def unique_zones(self) -> list[str]:
        seen: list[str] = []
        for z in self.product_zones:
            if z not in seen:
                seen.append(z)
        return seen

    @property
    def engaged(self) -> bool:
        return bool(self.zone_visits)

    @property
    def reached_billing(self) -> bool:
        return bool(self.queue_joins or self.billing_zone_times)

    @property
    def abandoned_queue(self) -> bool:
        return bool(self.queue_abandons) and not self.converted

    def dwell_by_zone(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for v in self.zone_visits:
            out[v.zone_id] = out.get(v.zone_id, 0.0) + v.dwell_s
        return out

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id, "visitor_id": self.visitor_id, "store_id": self.store_id,
            "start": self.start, "end": self.end or self.last_seen, "closed": self.closed,
            "close_reason": self.close_reason, "is_reentry": self.is_reentry,
            "duration_s": round(self.duration_s, 1), "zones": self.product_zones,
            "unique_zones": len(self.unique_zones),
            "total_dwell_s": round(sum(v.dwell_s for v in self.zone_visits), 1),
            "queue_joins": len(self.queue_joins), "queue_abandons": len(self.queue_abandons),
            "converted": self.converted, "transaction_id": self.transaction_id,
            "basket_value_inr": round(self.basket_value_inr, 2), "event_count": self.event_count,
        }


@dataclass
class SessionBuild:
    sessions: list[Session]
    orphan_events: int
    staff_events: int
    low_confidence_events: int
    total_events: int
    staff_ids: set

    @property
    def visitors(self) -> set[str]:
        return {s.visitor_id for s in self.sessions}


def build_sessions(events: list[dict], store_id: str, min_confidence: float = 0.0) -> SessionBuild:
    billing_zones = billing_zone_ids(store_id)
    by_visitor: dict[str, list[dict]] = {}
    staff = low = 0
    staff_ids: set[str] = set()
    for e in events:
        if e.get("is_staff"):
            staff += 1
            staff_ids.add(e["visitor_id"])
            continue
        if e.get("confidence", 1.0) < min_confidence:
            low += 1
            continue
        by_visitor.setdefault(e["visitor_id"], []).append(e)

    _order = {"ENTRY": 0, "REENTRY": 0, "ZONE_ENTER": 1, "ZONE_DWELL": 2, "ZONE_EXIT": 3,
              "BILLING_QUEUE_JOIN": 4, "BILLING_QUEUE_ABANDON": 5, "EXIT": 9}
    sessions: list[Session] = []
    orphans = 0
    for vid, evs in by_visitor.items():
        evs.sort(key=lambda e: (e["timestamp"], (e.get("metadata") or {}).get("session_seq") or 0,
                                _order.get(e["event_type"], 5)))
        cur: Optional[Session] = None
        n = 0
        open_zone: dict[str, ZoneVisit] = {}
        for e in evs:
            et, ts, zone = e["event_type"], e["timestamp"], e.get("zone_id")
            if et in OPENERS:
                if cur is not None and not cur.closed:
                    cur.closed, cur.close_reason, cur.end = False, "superseded", cur.last_seen
                n += 1
                cur = Session(session_id=f"{vid}_S{n}", visitor_id=vid, store_id=store_id, start=ts,
                              is_reentry=(et == "REENTRY" or n > 1), last_seen=ts)
                open_zone = {}
                sessions.append(cur)
            elif cur is None or cur.closed:
                orphans += 1
                continue
            cur.event_count += 1
            cur.last_seen = max(cur.last_seen or ts, ts)
            cur.min_confidence = min(cur.min_confidence, float(e.get("confidence", 1.0)))
            cur.cameras.add(e.get("camera_id"))
            if et == "EXIT":
                cur.end, cur.closed, cur.close_reason = ts, True, "exit"
            elif et == "ZONE_ENTER":
                if zone in billing_zones:
                    cur.billing_zone_times.append(ts)
                else:
                    zv = ZoneVisit(zone, ts)
                    cur.zone_visits.append(zv)
                    open_zone[zone] = zv
            elif et == "ZONE_DWELL":
                if zone in billing_zones:
                    cur.billing_zone_times.append(ts)
                    continue
                zv = open_zone.get(zone)
                if zv is None:  # dwell without an enter => synthesise the visit
                    zv = ZoneVisit(zone, ts)
                    cur.zone_visits.append(zv)
                    open_zone[zone] = zv
                zv.dwell_ms = (zv.dwell_ms or 0) + int(e.get("dwell_ms") or 0)
            elif et == "ZONE_EXIT":
                zv = open_zone.pop(zone, None)
                if zv is not None:
                    zv.exit = ts
            elif et == "BILLING_QUEUE_JOIN":
                cur.queue_joins.append(ts)
                depth = (e.get("metadata") or {}).get("queue_depth")
                if depth is not None:
                    cur.queue_depths.append(int(depth))
            elif et == "BILLING_QUEUE_ABANDON":
                cur.queue_abandons.append(ts)
    sessions.sort(key=lambda s: s.start)
    return SessionBuild(sessions, orphans, staff, low, len(events), staff_ids)
