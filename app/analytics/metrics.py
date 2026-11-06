"""Conversion KPIs (GET /metrics)."""
from __future__ import annotations

from statistics import mean, median

from .pos_match import Match
from .sessions import SessionBuild


def _r(x, n=2):
    return round(float(x), n) if x is not None else None


def compute_metrics(build: SessionBuild, matches: list[Match], transactions: list[dict]) -> dict:
    sessions = build.sessions
    visitors = build.visitors
    converted_visitors = {s.visitor_id for s in sessions if s.converted}
    revenue = sum(float(t["basket_value_inr"]) for t in transactions)
    attributed = sum(m.basket_value_inr for m in matches if m.session_id)
    durations = [s.duration_s for s in sessions if s.closed]
    dwell = [v.dwell_s for s in sessions for v in s.zone_visits]
    joins = sum(len(s.queue_joins) for s in sessions)
    abandons = sum(len(s.queue_abandons) for s in sessions)
    depths = [d for s in sessions for d in s.queue_depths]
    waits = [m.gap_s for m in matches if m.gap_s is not None and m.gap_s >= 0]
    n_vis = len(visitors)
    return {
        "visitors": n_vis,
        "sessions": len(sessions),
        "reentry_sessions": sum(1 for s in sessions if s.is_reentry),
        "engaged_visitors": len({s.visitor_id for s in sessions if s.engaged}),
        "converted_visitors": len(converted_visitors),
        "conversion_rate": _r(100 * len(converted_visitors) / n_vis, 1) if n_vis else 0.0,
        "transactions": len(transactions),
        "matched_transactions": sum(1 for m in matches if m.session_id),
        "revenue_inr": _r(revenue),
        "attributed_revenue_inr": _r(attributed),
        "avg_basket_inr": _r(revenue / len(transactions)) if transactions else 0.0,
        "revenue_per_visitor_inr": _r(revenue / n_vis) if n_vis else 0.0,
        "avg_session_duration_s": _r(mean(durations), 1) if durations else 0.0,
        "median_session_duration_s": _r(median(durations), 1) if durations else 0.0,
        "avg_zone_dwell_s": _r(mean(dwell), 1) if dwell else 0.0,
        "total_dwell_s": _r(sum(dwell), 1),
        "avg_zones_per_session": _r(mean(len(s.unique_zones) for s in sessions), 2) if sessions else 0.0,
        "queue": {
            "joins": joins,
            "abandons": abandons,
            "abandonment_rate": _r(100 * abandons / joins, 1) if joins else 0.0,
            "avg_queue_depth": _r(mean(depths), 2) if depths else 0.0,
            "max_queue_depth": max(depths) if depths else 0,
            "avg_wait_to_pay_s": _r(mean(waits), 1) if waits else 0.0,
        },
        "data_quality": {
            "total_events": build.total_events,
            "staff_events_excluded": build.staff_events,
            "staff_ids": len(build.staff_ids),
            "low_confidence_events_excluded": build.low_confidence_events,
            "orphan_events": build.orphan_events,
            "unclosed_sessions": sum(1 for s in sessions if not s.closed),
            "unmatched_transactions": sum(1 for m in matches if not m.session_id),
        },
    }
