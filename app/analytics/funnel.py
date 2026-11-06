"""Conversion funnel (GET /funnel): Visitors -> Engaged -> Billing -> Purchased.

``level="visitor"`` (default) counts each person once even if they re-entered;
``level="session"`` counts visits.
"""
from __future__ import annotations

from .sessions import SessionBuild

STAGES = [
    ("visitors", "Visitors", "Entered store"),
    ("engaged", "Engaged", "Visited product zones"),
    ("billing", "Billing", "Reached checkout"),
    ("purchased", "Purchased", "Completed purchase"),
]


def compute_funnel(build: SessionBuild, level: str = "visitor") -> dict:
    sessions = build.sessions
    key = (lambda s: s.visitor_id) if level == "visitor" else (lambda s: s.session_id)
    sets = {
        "visitors": {key(s) for s in sessions},
        "engaged": {key(s) for s in sessions if s.engaged},
        "billing": {key(s) for s in sessions if s.engaged and s.reached_billing},
        "purchased": {key(s) for s in sessions if s.converted},
    }
    # keep the funnel monotonic: a purchaser who skipped camera coverage still counts upstream
    sets["billing"] |= sets["purchased"]
    sets["engaged"] |= sets["billing"]
    top = len(sets["visitors"]) or 1
    out, prev = [], None
    for k, name, desc in STAGES:
        n = len(sets[k])
        out.append({
            "stage": name, "key": k, "description": desc, "count": n,
            "pct_of_visitors": round(100 * n / top, 1),
            "conversion_from_prev": (100.0 if prev is None else round(100 * n / prev, 1) if prev else 0.0),
            "drop_off_from_prev": (prev - n) if prev is not None else 0,
        })
        prev = n
    return {"level": level, "stages": out}
