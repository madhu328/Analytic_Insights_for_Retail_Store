"""Common processor plumbing: presence bookkeeping and staff heuristics."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from ..tracking import Track


class Processor:
    OPTIONS: set[str] = {"staff_presence_ratio", "staff_min_s"}

    def __init__(self, cam_cfg: dict, emitter, staff_presence_ratio: float = 0.9,
                 staff_min_s: float = 90.0):
        self.cfg, self.emitter = cam_cfg, emitter
        self.anchor = cam_cfg.get("anchor", "bottom_center")
        self.staff_presence_ratio, self.staff_min_s = staff_presence_ratio, staff_min_s
        self._first: dict[str, datetime] = {}
        self._last: dict[str, datetime] = {}
        self.clip_start: Optional[datetime] = None
        self.clip_end: Optional[datetime] = None

    def _seen(self, visitor_id: str, ts: datetime) -> None:
        self._first.setdefault(visitor_id, ts)
        self._last[visitor_id] = ts

    def update(self, ts: datetime, tracks: list[Track]) -> None:
        self.clip_start = self.clip_start or ts
        self.clip_end = ts
        self._update(ts, tracks)

    def _update(self, ts: datetime, tracks: list[Track]) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def finalize(self, ts: Optional[datetime] = None) -> list[dict]:
        ts = ts or self.clip_end
        if ts is not None:
            self._finalize(ts)
        self.emitter.mark_staff(self.staff_ids())
        return self.emitter.flush()

    def _finalize(self, ts: datetime) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def staff_ids(self) -> set[str]:
        """Tracks present for most of a long clip are staff (cashier, floor associate)."""
        if not self.clip_start or not self.clip_end or self.staff_presence_ratio <= 0:
            return set()
        clip = (self.clip_end - self.clip_start).total_seconds()
        if clip < self.staff_min_s:
            return set()
        return {v for v, f in self._first.items()
                if (self._last[v] - f).total_seconds() >= self.staff_presence_ratio * clip}
