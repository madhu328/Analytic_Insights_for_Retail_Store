"""CAM5 billing: queue polygon + payment polygon.

* BILLING_QUEUE_JOIN once a shopper has stood in the queue polygon for ``join_min_s``
  (``metadata.queue_depth`` = people in the queue at that moment).
* Leaving the queue without reaching the payment polygon after ``abandon_min_s`` in queue
  => BILLING_QUEUE_ABANDON (dwell_ms = time spent queueing).
* Payment polygon visits => ZONE_ENTER / ZONE_DWELL / ZONE_EXIT with zone_id PAYMENT.
Tracks lost *inside* the queue are not called abandons (occlusion is far more likely).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ..geometry import anchor_point, point_in_polygon
from ..tracking import Track
from .base import Processor


@dataclass
class _Q:
    q_since: Optional[datetime] = None
    last_q: Optional[datetime] = None
    joined_at: Optional[datetime] = None
    paid: bool = False
    pay_since: Optional[datetime] = None
    pay_open: bool = False
    last_pay: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    last_in_q: bool = False
    conf: float = 0.85


class QueueProcessor(Processor):
    OPTIONS = Processor.OPTIONS | {"join_min_s", "abandon_min_s", "leave_grace_s", "lost_timeout_s",
                                   "payment_min_s"}

    def __init__(self, cam_cfg: dict, emitter, join_min_s: float = 3.0, abandon_min_s: float = 5.0,
                 leave_grace_s: float = 2.0, lost_timeout_s: float = 3.0, payment_min_s: float = 1.0, **kw):
        super().__init__(cam_cfg, emitter, **kw)
        self.queue_zone = cam_cfg["queue_polygon"]["zone_id"]
        self.queue_poly = cam_cfg["queue_polygon"]["polygon"]
        pay = cam_cfg.get("payment_polygon") or {}
        self.pay_zone, self.pay_poly = pay.get("zone_id", "PAYMENT"), pay.get("polygon")
        self.join_min_s, self.abandon_min_s = join_min_s, abandon_min_s
        self.leave_grace_s, self.lost_timeout_s, self.payment_min_s = leave_grace_s, lost_timeout_s, payment_min_s
        self.q: dict[int, _Q] = {}

    @staticmethod
    def vid(tid: int) -> str:
        return f"VIS_B{tid}"

    def _abandon(self, tid: int, s: _Q) -> None:
        queued = (s.last_q - s.joined_at).total_seconds() if s.last_q and s.joined_at else 0
        if not s.paid and queued >= self.abandon_min_s:
            self.emitter.emit("BILLING_QUEUE_ABANDON", self.vid(tid), s.last_q, zone_id=self.queue_zone,
                              dwell_ms=int(queued * 1000), confidence=s.conf)
        s.joined_at, s.q_since = None, None

    def _close_pay(self, tid: int, s: _Q) -> None:
        if s.pay_open:
            d = (s.last_pay - s.pay_since).total_seconds()
            self.emitter.emit("ZONE_EXIT", self.vid(tid), s.last_pay, zone_id=self.pay_zone, confidence=s.conf)
            self.emitter.emit("ZONE_DWELL", self.vid(tid), s.last_pay, zone_id=self.pay_zone,
                              dwell_ms=int(d * 1000), confidence=s.conf)
        s.pay_open, s.pay_since = False, None

    def _update(self, ts: datetime, tracks: list[Track]) -> None:
        pts = {t.track_id: anchor_point(t.bbox, self.anchor) for t in tracks}
        in_q = {tid for tid, p in pts.items() if point_in_polygon(p, self.queue_poly)}
        in_pay = {tid for tid, p in pts.items() if self.pay_poly and point_in_polygon(p, self.pay_poly)}
        depth = len(in_q)
        for t in tracks:
            tid = t.track_id
            s = self.q.setdefault(tid, _Q())
            s.last_seen, s.conf = ts, 0.5 * s.conf + 0.5 * t.confidence
            self._seen(self.vid(tid), ts, t.bbox)
            s.last_in_q = tid in in_q
            if tid in in_q:
                s.q_since = s.q_since or ts
                s.last_q = ts
                if s.joined_at is None and (ts - s.q_since).total_seconds() >= self.join_min_s:
                    s.joined_at, s.paid = s.q_since, False
                    self.emitter.emit("BILLING_QUEUE_JOIN", self.vid(tid), s.q_since, zone_id=self.queue_zone,
                                      queue_depth=depth, confidence=s.conf)
            elif s.joined_at is None:
                s.q_since = None
            if tid in in_pay:
                s.pay_since = s.pay_since or ts
                s.last_pay = ts
                if s.joined_at is not None:
                    s.paid = True
                if not s.pay_open and (ts - s.pay_since).total_seconds() >= self.payment_min_s:
                    s.pay_open = True
                    self.emitter.emit("ZONE_ENTER", self.vid(tid), s.pay_since, zone_id=self.pay_zone,
                                      confidence=s.conf)
            elif s.pay_since and (ts - (s.last_pay or ts)).total_seconds() > self.leave_grace_s:
                self._close_pay(tid, s)
            if (s.joined_at is not None and tid not in in_q and tid not in in_pay
                    and (ts - s.last_q).total_seconds() > self.leave_grace_s):
                if s.paid:
                    s.joined_at, s.q_since = None, None
                else:
                    self._abandon(tid, s)
        for tid, s in list(self.q.items()):
            if s.last_seen and s.last_seen < ts and (ts - s.last_seen).total_seconds() > self.lost_timeout_s:
                if s.joined_at is not None and not s.last_in_q:
                    self._abandon(tid, s)
                self._close_pay(tid, s)
                del self.q[tid]

    def _finalize(self, ts: datetime) -> None:
        for tid, s in list(self.q.items()):
            self._close_pay(tid, s)
        self.q.clear()
