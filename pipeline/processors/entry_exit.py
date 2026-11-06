"""CAM3 doorway: entry line half-plane crossing -> ENTRY / EXIT / REENTRY.

A track's feet point is classified as inside/outside by which side of the entry line it is
on (``inside_point`` defines the store side). A side change confirmed for ``min_frames``
consecutive samples is a crossing. On ENTRY the person's best crop is embedded (OSNet) and
resolved against the Re-ID gallery: a known visitor who previously exited => REENTRY.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ..geometry import anchor_point, side_of_line
from ..reid import Embedder, ReIDManager
from ..tracking import Track
from .base import Processor


@dataclass
class _Door:
    side: Optional[str] = None
    pending: Optional[str] = None
    pending_n: int = 0
    pending_since: Optional[datetime] = None
    best_crop: object = None
    best_area: float = 0.0
    visitor_id: Optional[str] = None
    conf: float = 0.85
    last_seen: Optional[datetime] = None


class EntryExitProcessor(Processor):
    OPTIONS = Processor.OPTIONS | {"reid", "embedder", "min_frames", "lost_timeout_s"}

    def __init__(self, cam_cfg: dict, emitter, reid: Optional[ReIDManager] = None,
                 embedder: Optional[Embedder] = None, min_frames: int = 2,
                 lost_timeout_s: float = 3.0, **kw):
        kw.setdefault("staff_presence_ratio", 0.95)
        super().__init__(cam_cfg, emitter, **kw)
        self.a, self.b = cam_cfg["entry_line"]
        self.inside_sign = 1 if side_of_line(tuple(cam_cfg["inside_point"]), self.a, self.b) > 0 else -1
        self.reid = reid or ReIDManager()
        self.embedder = embedder
        self.min_frames, self.lost_timeout_s = min_frames, lost_timeout_s
        self.tracks: dict[int, _Door] = {}

    def side(self, bbox) -> str:
        s = side_of_line(anchor_point(bbox, self.anchor), self.a, self.b)
        return "in" if s * self.inside_sign > 0 else "out"

    def _embed(self, d: _Door):
        return self.embedder(d.best_crop) if (self.embedder and d.best_crop is not None) else None

    def _resolve(self, tid: int, d: _Door):
        if d.visitor_id:
            return self.reid.store.visitors.get(d.visitor_id), True
        emb = self._embed(d)
        if emb is None:  # no appearance model: identity == track id
            return self.reid.store.get_or_create(f"VIS_T{tid}")
        st, known, sim = self.reid.resolve(emb)
        d.conf = min(d.conf, sim) if known else d.conf
        return st, known

    def _crossed(self, tid: int, d: _Door, new_side: str, ts: datetime) -> None:
        st, known = self._resolve(tid, d)
        d.visitor_id = st.visitor_id
        self._seen(st.visitor_id, ts)
        if new_side == "in":
            if known and st.inside:
                return  # duplicate crossing (jitter / re-detected) -> ignore
            kind = "REENTRY" if (known and st.last_exit is not None) else "ENTRY"
            st.inside, st.last_entry = True, ts
            st.sessions += 1
            self.emitter.emit(kind, st.visitor_id, ts, confidence=d.conf)
        else:
            if known and not st.inside and st.last_exit is not None:
                return
            st.inside, st.last_exit = False, ts
            self.emitter.emit("EXIT", st.visitor_id, ts, confidence=d.conf)

    def _update(self, ts: datetime, tracks: list[Track]) -> None:
        seen = set()
        for t in tracks:
            seen.add(t.track_id)
            d = self.tracks.setdefault(t.track_id, _Door())
            d.last_seen = ts
            d.conf = 0.5 * d.conf + 0.5 * t.confidence
            if d.visitor_id:
                self._seen(d.visitor_id, ts)
            area = (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1])
            if t.crop is not None and area > d.best_area:
                d.best_crop, d.best_area = t.crop, area
            s = self.side(t.bbox)
            if d.side is None:
                d.side = s
                continue
            if s == d.side:
                d.pending, d.pending_n = None, 0
                continue
            if d.pending != s:
                d.pending, d.pending_n, d.pending_since = s, 0, ts
            d.pending_n += 1
            if d.pending_n >= self.min_frames:
                self._crossed(t.track_id, d, s, d.pending_since)
                d.side, d.pending, d.pending_n = s, None, 0
        for tid, d in list(self.tracks.items()):
            if tid not in seen and d.last_seen and (ts - d.last_seen).total_seconds() > self.lost_timeout_s:
                del self.tracks[tid]

    def _finalize(self, ts: datetime) -> None:
        self.tracks.clear()
