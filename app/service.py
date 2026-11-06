"""Composes the analytics for one store + time window (session builder first)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

from sqlalchemy.engine import Engine

from . import db
from .analytics.pos_match import Match, match_transactions
from .analytics.sessions import SessionBuild, build_sessions
from .analytics.timeseries import hourly_series
from .config import settings


@dataclass
class Analysis:
    store_id: str
    start: Optional[datetime]
    end: Optional[datetime]
    events: list[dict]
    transactions: list[dict]
    build: SessionBuild
    matches: list[Match]

    _hourly: Optional[list[dict]] = None

    @property
    def hourly(self) -> list[dict]:
        if self._hourly is None:
            self._hourly = hourly_series(self.build, self.matches, self.transactions)
        return self._hourly


def resolve_window(day: Optional[date], start: Optional[datetime],
                   end: Optional[datetime]) -> tuple[Optional[datetime], Optional[datetime]]:
    if day is not None:
        start = datetime.combine(day, datetime.min.time())
        end = start + timedelta(days=1)
    for v in (start, end):
        if v is not None and v.tzinfo is not None:
            raise ValueError("pass naive UTC timestamps")
    return start, end


def analyse(engine: Engine, store_id: str, start: Optional[datetime] = None,
            end: Optional[datetime] = None, window_before_min: Optional[float] = None,
            window_after_min: Optional[float] = None) -> Analysis:
    events = db.fetch_events(engine, store_id, start, end)
    txs = db.fetch_pos(engine, store_id, start, end)
    build = build_sessions(events, store_id, settings.min_confidence)
    matches = match_transactions(
        build.sessions, txs,
        settings.match_window_before_min if window_before_min is None else window_before_min,
        settings.match_window_after_min if window_after_min is None else window_after_min)
    return Analysis(store_id, start, end, events, txs, build, matches)
