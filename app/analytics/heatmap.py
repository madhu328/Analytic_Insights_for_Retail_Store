"""Zone heatmap (GET /heatmap).

Per zone: visits, unique visitors, dwell, an engagement score (0-100) and a hot / warm /
cold class following Chen et al. (2026): *hot* = high traffic AND dwell above the store
median, *cold* = low traffic or zero visits, *warm* otherwise.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import mean, median

from ..layout import product_zone_ids, store_layout
from .sessions import SessionBuild


def compute_heatmap(build: SessionBuild, store_id: str) -> dict:
    visits = defaultdict(int)
    visitors = defaultdict(set)
    sessions_in = defaultdict(set)
    dwell = defaultdict(list)
    hourly = defaultdict(lambda: defaultdict(int))
    for s in build.sessions:
        for v in s.zone_visits:
            visits[v.zone_id] += 1
            visitors[v.zone_id].add(s.visitor_id)
            sessions_in[v.zone_id].add(s.session_id)
            dwell[v.zone_id].append(v.dwell_s)
            hourly[v.zone_id][v.enter.hour] += 1
    zones = list(dict.fromkeys(product_zone_ids(store_id) + sorted(visits)))
    names = {z["zone_id"]: z.get("name", z["zone_id"]) for z in store_layout(store_id).get("zones", [])}
    max_visits = max(visits.values(), default=0) or 1
    avg_dwell = {z: mean(dwell[z]) if dwell[z] else 0.0 for z in zones}
    max_dwell = max(avg_dwell.values(), default=0) or 1
    total_visits = sum(visits.values()) or 1
    n_sessions = len(build.sessions) or 1
    traffic_sorted = sorted((visits[z] for z in zones), reverse=True)
    hot_cut = traffic_sorted[max(0, len(traffic_sorted) // 3 - 1)] if traffic_sorted else 0
    cold_cut = traffic_sorted[min(len(traffic_sorted) - 1, (2 * len(traffic_sorted)) // 3)] if traffic_sorted else 0
    med_dwell = median([d for d in avg_dwell.values() if d > 0]) if any(avg_dwell.values()) else 0
    rows = []
    for z in zones:
        score = 100 * (0.6 * visits[z] / max_visits + 0.4 * avg_dwell[z] / max_dwell)
        if visits[z] == 0 or visits[z] < cold_cut:
            klass = "cold"
        elif visits[z] >= hot_cut and avg_dwell[z] >= med_dwell:
            klass = "hot"
        else:
            klass = "warm"
        rows.append({
            "zone_id": z, "name": names.get(z, z.replace("_", " ").title()),
            "visits": visits[z], "unique_visitors": len(visitors[z]),
            "session_reach_pct": round(100 * len(sessions_in[z]) / n_sessions, 1),
            "traffic_share_pct": round(100 * visits[z] / total_visits, 1),
            "total_dwell_s": round(sum(dwell[z]), 1), "avg_dwell_s": round(avg_dwell[z], 1),
            "median_dwell_s": round(median(dwell[z]), 1) if dwell[z] else 0.0,
            "engagement_score": round(score, 1), "zone_class": klass,
        })
    rows.sort(key=lambda r: r["engagement_score"], reverse=True)
    hours = sorted({h for z in hourly for h in hourly[z]})
    matrix = {"hours": hours, "zones": [r["zone_id"] for r in rows],
              "values": [[hourly[r["zone_id"]].get(h, 0) for h in hours] for r in rows]}
    hot = [r for r in rows if r["zone_class"] == "hot"]
    return {
        "zones": rows, "hour_zone_matrix": matrix,
        "summary": {
            "zones": len(rows), "hot": len(hot),
            "warm": sum(r["zone_class"] == "warm" for r in rows),
            "cold": sum(r["zone_class"] == "cold" for r in rows),
            "hot_zone_traffic_share_pct": round(sum(r["traffic_share_pct"] for r in hot), 1),
        },
    }
