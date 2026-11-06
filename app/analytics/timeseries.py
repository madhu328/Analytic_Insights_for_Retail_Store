"""Hourly time series used by the dashboard, anomaly detection and forecasting."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from statistics import mean

from .pos_match import Match
from .sessions import SessionBuild


def _hour(ts: datetime) -> datetime:
    return ts.replace(minute=0, second=0, microsecond=0)


def hourly_series(build: SessionBuild, matches: list[Match], transactions: list[dict]) -> list[dict]:
    b = defaultdict(lambda: {"entries": 0, "visitors": set(), "queue_joins": 0, "queue_abandons": 0,
                             "depths": [], "transactions": 0, "revenue_inr": 0.0, "conversions": 0,
                             "dwell": []})
    for s in build.sessions:
        h = b[_hour(s.start)]
        h["entries"] += 1
        h["visitors"].add(s.visitor_id)
        h["dwell"].extend(v.dwell_s for v in s.zone_visits)
        if s.converted:
            h["conversions"] += 1
        for t in s.queue_joins:
            b[_hour(t)]["queue_joins"] += 1
        for t in s.queue_abandons:
            b[_hour(t)]["queue_abandons"] += 1
        for t, d in zip(s.queue_joins, s.queue_depths):
            b[_hour(t)]["depths"].append(d)
    for tx in transactions:
        h = b[_hour(tx["timestamp"])]
        h["transactions"] += 1
        h["revenue_inr"] += float(tx["basket_value_inr"])
    if not b:
        return []
    lo, hi = min(b), max(b)
    out = []
    cur = lo
    from datetime import timedelta
    while cur <= hi:
        h = b.get(cur) or b[cur]
        out.append({
            "hour": cur, "entries": h["entries"], "visitors": len(h["visitors"]),
            "queue_joins": h["queue_joins"], "queue_abandons": h["queue_abandons"],
            "avg_queue_depth": round(mean(h["depths"]), 2) if h["depths"] else 0.0,
            "max_queue_depth": max(h["depths"]) if h["depths"] else 0,
            "transactions": h["transactions"], "revenue_inr": round(h["revenue_inr"], 2),
            "conversions": h["conversions"],
            "conversion_rate": round(100 * h["conversions"] / h["entries"], 1) if h["entries"] else 0.0,
            "avg_dwell_s": round(mean(h["dwell"]), 1) if h["dwell"] else 0.0,
        })
        cur += timedelta(hours=1)
    return out
