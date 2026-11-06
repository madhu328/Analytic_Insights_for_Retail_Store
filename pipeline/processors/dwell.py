"""CAM1 / CAM2 shelf cameras: zone polygons -> ZONE_ENTER / ZONE_DWELL / ZONE_EXIT.

Debounced state machine per track: a zone must be occupied for ``min_enter_s`` before
ZONE_ENTER, and vacated for ``exit_grace_s`` before ZONE_EXIT (absorbs detector jitter and
short occlusions). ZONE_DWELL carries the total dwell and is emitted on exit when it is at
least ``min_dwell_s``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from ..geometry import anchor_point, point_in_polygon
from ..tracking import Track
from .base import Processor


@dataclass
class _State:
    zone: Optional[str] = None
    zone_since: Optional[datetime] = None
    last_in_zone: Optional[datetime] = None
    candidate: Optional[str] = None
    candidate_since: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    confs: list = field(default_factory=list)


class ZoneDwellProcessor(Processor):
    OPTIONS = Processor.OPTIONS | {"min_enter_s", "exit_grace_s", "min_dwell_s", "lost_timeout_s",
                                   "visitor_resolver"}

    def __init__(self, cam_cfg: dict, emitter, min_enter_s: float = 1.0, exit_grace_s: float = 1.5,
                 min_dwell_s: float = 3.0, lost_timeout_s: float = 3.0,
                 visitor_resolver: Optional[Callable[[Track], str]] = None, **kw):
        super().__init__(cam_cfg, emitter, **kw)
        self.zones = [(z["zone_id"], z["polygon"]) for z in cam_cfg.get("zones", [])]
        self.min_enter_s, self.exit_grace_s = min_enter_s, exit_grace_s
        self.min_dwell_s, self.lost_timeout_s = min_dwell_s, lost_timeout_s
        self.resolver = visitor_resolver
        self.state: dict[int, _State] = {}
        self.vid: dict[int, str] = {}

    def _visitor(self, t: Track) -> str:
        if t.track_id not in self.vid:
            self.vid[t.track_id] = self.resolver(t) if self.resolver else f"VIS_{t.track_id}"
        return self.vid[t.track_id]

    def zone_of(self, bbox) -> Optional[str]:
        p = anchor_point(bbox, self.anchor)
        for zid, poly in self.zones:
            if point_in_polygon(p, poly):
                return zid
        return None

    def _open(self, tid: int, st: _State, zone: str, ts: datetime) -> None:
        st.zone, st.zone_since, st.last_in_zone, st.candidate = zone, ts, ts, None
        self.emitter.emit("ZONE_ENTER", self.vid[tid], ts, zone_id=zone, sku_zone=zone,
                          confidence=self._conf(st))

    def _close(self, tid: int, st: _State) -> None:
        if st.zone is None:
            return
        exit_ts = st.last_in_zone or st.zone_since
        dwell_s = (exit_ts - st.zone_since).total_seconds()
        conf = self._conf(st)
        self.emitter.emit("ZONE_EXIT", self.vid[tid], exit_ts, zone_id=st.zone, sku_zone=st.zone,
                          confidence=conf)
        if dwell_s >= self.min_dwell_s:
            self.emitter.emit("ZONE_DWELL", self.vid[tid], exit_ts, zone_id=st.zone, sku_zone=st.zone,
                              dwell_ms=int(dwell_s * 1000), confidence=conf)
        st.zone = st.zone_since = st.last_in_zone = None
        st.confs = []

    @staticmethod
    def _conf(st: _State) -> float:
        return sum(st.confs) / len(st.confs) if st.confs else 0.85

    def _update(self, ts: datetime, tracks: list[Track]) -> None:
        seen = set()
        for t in tracks:
            seen.add(t.track_id)
            vid = self._visitor(t)
            self._seen(vid, ts, t.bbox)
            st = self.state.setdefault(t.track_id, _State())
            st.last_seen = ts
            st.confs.append(t.confidence)
            z = self.zone_of(t.bbox)
            if st.zone is not None and z == st.zone:
                st.last_in_zone, st.candidate = ts, None
                continue
            if z is not None:
                if st.candidate != z:
                    st.candidate, st.candidate_since = z, ts
                if (ts - st.candidate_since).total_seconds() >= self.min_enter_s:
                    self._close(t.track_id, st)
                    self._open(t.track_id, st, z, st.candidate_since)
                    continue
            else:
                st.candidate = None
            if st.zone is not None and (ts - st.last_in_zone).total_seconds() > self.exit_grace_s:
                self._close(t.track_id, st)
        for tid, st in list(self.state.items()):
            if tid not in seen and st.last_seen and (ts - st.last_seen).total_seconds() > self.lost_timeout_s:
                self._close(tid, st)
                del self.state[tid]

    def _finalize(self, ts: datetime) -> None:
        for tid, st in list(self.state.items()):
            self._close(tid, st)
        self.state.clear()
