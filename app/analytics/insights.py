"""Behavioural insights adapted from Chen et al., "Customer movements..." (Procedia CS 2026).

* Path Complexity Index (PCI): how far a shopper's path deviates from the ideal route.
  The paper uses actual length / straight entrance->checkout distance plus a turn penalty.
  Our CCTV events are zone-level, so the path is the polyline entrance -> visited zone
  centroids (floor_xy in store_layout.json) -> checkout/entrance, and the baseline is the
  shortest tour through the same zones (see ``path_complexity``).
  Segments as in the paper: efficient (< 1.5), browsing (1.5 - 2.5), roaming (> 2.5).
* Hesitation / service touch-points: dwell timeout (> 60 s) and loitering (zone revisited
  >= 2 times in a visit); tiered mild / moderate / severe for staff intervention.
* Shelf attention-conversion quadrant: attention = % of sessions visiting the zone,
  conversion = % of those sessions that purchased. Star / Problem / Potential / Obsolete.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean, median, pstdev

from ..config import settings
from ..layout import store_layout, zone_floor_xy
from .sessions import Session, SessionBuild


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _turns(points: list[tuple[float, float]], threshold_deg: float = 45.0) -> int:
    n = 0
    for a, b, c in zip(points, points[1:], points[2:]):
        v1, v2 = (b[0] - a[0], b[1] - a[1]), (c[0] - b[0], c[1] - b[1])
        n1, n2 = math.hypot(*v1), math.hypot(*v2)
        if n1 == 0 or n2 == 0:
            continue
        cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        if math.degrees(math.acos(cos)) > threshold_deg:
            n += 1
    return n


def _shortest_tour(start, stops: list, end) -> list:
    """Shortest open path start -> all stops -> end (exact for <= 7 stops, else nearest-neighbour)."""
    from itertools import permutations
    if len(stops) <= 7:
        best = min(permutations(stops), key=lambda order: sum(
            _dist(a, b) for a, b in zip((start, *order), (*order, end))), default=())
        return [start, *best, end]
    path, rest = [start], list(stops)
    while rest:
        nxt = min(rest, key=lambda q: _dist(path[-1], q))
        rest.remove(nxt)
        path.append(nxt)
    return path + [end]


def path_complexity(session: Session, store_id: str) -> dict | None:
    """PCI = L_actual / L_ideal + 0.1 * extra_turns.

    L_ideal is the shortest tour entrance -> the *same set* of zones -> checkout (or back to
    the entrance for non-billing visits), so PCI ~ 1 means no wasted movement and > 2.5
    means heavy back-tracking / searching. extra_turns = heading changes > 45 deg beyond
    those the ideal tour also needs.
    """
    layout = store_layout(store_id)
    xy = zone_floor_xy(store_id)
    entrance = tuple(layout.get("entrance_xy", (0.0, 0.0)))
    checkout = tuple(layout.get("checkout_xy", entrance))
    visited = [tuple(xy[z]) for z in session.product_zones if z in xy]
    if not visited:
        return None
    end = checkout if session.reached_billing or session.converted else entrance
    pts = [entrance, *visited, end]
    ideal_pts = _shortest_tour(entrance, list(dict.fromkeys(visited)), end)
    actual = sum(_dist(a, b) for a, b in zip(pts, pts[1:]))
    ideal = sum(_dist(a, b) for a, b in zip(ideal_pts, ideal_pts[1:])) or 1.0
    extra_turns = max(0, _turns(pts) - _turns(ideal_pts))
    pci = actual / ideal + 0.1 * extra_turns
    seg = "efficient" if pci < 1.5 else "browsing" if pci <= 2.5 else "roaming"
    return {"session_id": session.session_id, "pci": round(pci, 2), "path_length_m": round(actual, 1),
            "ideal_length_m": round(ideal, 1), "extra_turns": extra_turns, "segment": seg,
            "zones": len(visited), "duration_s": round(session.duration_s, 1),
            "converted": session.converted}


def journey_insights(build: SessionBuild, store_id: str) -> dict:
    rows = [r for s in build.sessions if (r := path_complexity(s, store_id))]
    segs = defaultdict(list)
    for r in rows:
        segs[r["segment"]].append(r)
    seg_out = []
    for name in ("efficient", "browsing", "roaming"):
        g = segs.get(name, [])
        seg_out.append({
            "segment": name, "sessions": len(g),
            "share_pct": round(100 * len(g) / len(rows), 1) if rows else 0.0,
            "avg_duration_s": round(mean(x["duration_s"] for x in g), 1) if g else 0.0,
            "conversion_rate": round(100 * sum(x["converted"] for x in g) / len(g), 1) if g else 0.0,
        })
    # transitions between consecutive zones (top flows)
    flows = Counter()
    for s in build.sessions:
        z = s.product_zones
        flows.update(zip(z, z[1:]))
    pcis = [r["pci"] for r in rows]
    return {
        "pci": {"mean": round(mean(pcis), 2) if pcis else None,
                "std": round(pstdev(pcis), 2) if len(pcis) > 1 else None,
                "median": round(median(pcis), 2) if pcis else None,
                "histogram": Counter(round(p * 2) / 2 for p in pcis).most_common()},
        "segments": seg_out,
        "top_flows": [{"from": a, "to": b, "count": c} for (a, b), c in flows.most_common(15)],
        "sessions": rows,
    }


def hesitation_insights(build: SessionBuild) -> dict:
    t_mild, t_severe, loiter = (settings.hesitation_dwell_s, settings.severe_hesitation_dwell_s,
                                settings.loiter_revisits)
    per_zone = defaultdict(lambda: {"mild": 0, "moderate": 0, "severe": 0, "sessions": set()})
    signals = []
    for s in build.sessions:
        counts = Counter(s.product_zones)
        dwell = s.dwell_by_zone()
        for z, d in dwell.items():
            timeout = d > t_mild
            loitering = counts[z] >= loiter
            if not (timeout or loitering):
                continue
            level = "severe" if (d > t_severe and loitering) or d > 2 * t_severe else \
                    "moderate" if (timeout and loitering) or d > t_severe else "mild"
            per_zone[z][level] += 1
            per_zone[z]["sessions"].add(s.session_id)
            signals.append({"session_id": s.session_id, "zone_id": z, "dwell_s": round(d, 1),
                            "visits": counts[z], "level": level, "converted": s.converted})
    actions = {"mild": "Push promo on shelf screen / e-price tag",
               "moderate": "Alert floor staff to be ready to assist",
               "severe": "Staff to proactively approach and ask about needs"}
    zones = []
    for z, d in per_zone.items():
        n = len(d["sessions"])
        conv = sum(1 for x in signals if x["zone_id"] == z and x["converted"])
        zones.append({"zone_id": z, "hesitating_sessions": n, "mild": d["mild"],
                      "moderate": d["moderate"], "severe": d["severe"],
                      "hesitation_conversion_pct": round(100 * conv / n, 1) if n else 0.0,
                      "recommended_action": actions["severe" if d["severe"] else
                                                    "moderate" if d["moderate"] else "mild"]})
    zones.sort(key=lambda r: (r["severe"], r["moderate"], r["hesitating_sessions"]), reverse=True)
    return {"thresholds": {"dwell_timeout_s": t_mild, "severe_dwell_s": t_severe,
                           "loiter_revisits": loiter},
            "zones": zones, "signals": len(signals), "levels": Counter(x["level"] for x in signals),
            "actions": actions}


def shelf_quadrant(build: SessionBuild, store_id: str) -> dict:
    n = len(build.sessions) or 1
    reach = defaultdict(set)
    buyers = defaultdict(set)
    for s in build.sessions:
        for z in set(s.product_zones):
            reach[z].add(s.session_id)
            if s.converted:
                buyers[z].add(s.session_id)
    rows = []
    for z in reach:
        att = 100 * len(reach[z]) / n
        conv = 100 * len(buyers[z]) / len(reach[z])
        rows.append({"zone_id": z, "attention_pct": round(att, 1), "conversion_pct": round(conv, 1),
                     "viewers": len(reach[z]), "buyers": len(buyers[z])})
    if not rows:
        return {"zones": [], "thresholds": {}}
    a_mid = median(r["attention_pct"] for r in rows)
    c_mid = median(r["conversion_pct"] for r in rows)
    strategy = {"star": "Maintain placement & stock depth",
                "problem": "High attention, low conversion: fix display, pricing or availability",
                "potential": "Converts well but under-seen: increase exposure / move to hot zone",
                "obsolete": "Low attention & conversion: consider reallocating the fixture"}
    for r in rows:
        hi_a, hi_c = r["attention_pct"] >= a_mid, r["conversion_pct"] >= c_mid
        q = "star" if hi_a and hi_c else "problem" if hi_a else "potential" if hi_c else "obsolete"
        r["quadrant"], r["strategy"] = q, strategy[q]
    rows.sort(key=lambda r: r["attention_pct"], reverse=True)
    return {"zones": rows, "thresholds": {"attention_pct": a_mid, "conversion_pct": c_mid}}
