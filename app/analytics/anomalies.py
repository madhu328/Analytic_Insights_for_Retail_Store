"""Rule + robust-statistics anomaly detection, derived at query time.

Each anomaly: type, severity (info / warning / critical), when, where, observed value,
threshold, and a suggested action so the dashboard can be operational, not just descriptive.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, time, timedelta
from statistics import median

from ..config import settings
from ..layout import product_zone_ids, store_layout
from .pos_match import Match
from .sessions import SessionBuild


def _robust_z(values: list[float]) -> list[float]:
    if len(values) < 4:
        return [0.0] * len(values)
    med = median(values)
    mad = median(abs(v - med) for v in values) or 1.0
    return [0.6745 * (v - med) / mad for v in values]


def _a(kind, severity, title, detail, *, at=None, zone=None, value=None, threshold=None, action=""):
    return {"type": kind, "severity": severity, "title": title, "detail": detail,
            "timestamp": at, "zone_id": zone, "value": value, "threshold": threshold,
            "suggested_action": action}


def detect_anomalies(build: SessionBuild, matches: list[Match], transactions: list[dict],
                     hourly: list[dict], events: list[dict], store_id: str) -> list[dict]:
    out: list[dict] = []
    sessions = build.sessions

    # 1. Footfall spikes / drops (robust z-score on hourly entries)
    active = [h for h in hourly if h["entries"] or h["transactions"]]
    z = _robust_z([h["entries"] for h in active])
    for h, s in zip(active, z):
        if s > 3 and h["entries"] >= 5:
            out.append(_a("FOOTFALL_SPIKE", "warning", "Footfall spike",
                          f"{h['entries']} entries in hour (robust z={s:.1f})", at=h["hour"],
                          value=h["entries"], threshold="z>3",
                          action="Add floor staff / open extra billing counter"))
        elif s < -3:
            out.append(_a("FOOTFALL_DROP", "info", "Footfall drop",
                          f"Only {h['entries']} entries in hour (robust z={s:.1f})", at=h["hour"],
                          value=h["entries"], threshold="z<-3",
                          action="Check entrance camera / external factors"))

    # 2. Queue congestion
    for h in hourly:
        if h["max_queue_depth"] >= settings.queue_depth_alert:
            sev = "critical" if h["max_queue_depth"] >= settings.queue_depth_alert * 1.5 else "warning"
            out.append(_a("QUEUE_CONGESTION", sev, "Billing queue congestion",
                          f"Queue depth reached {h['max_queue_depth']} (avg {h['avg_queue_depth']})",
                          at=h["hour"], zone="BILLING", value=h["max_queue_depth"],
                          threshold=settings.queue_depth_alert,
                          action="Open an additional billing counter"))
        if h["queue_joins"] >= 3 and h["queue_abandons"] / h["queue_joins"] > 0.3:
            rate = 100 * h["queue_abandons"] / h["queue_joins"]
            out.append(_a("QUEUE_ABANDONMENT", "warning", "High queue abandonment",
                          f"{h['queue_abandons']}/{h['queue_joins']} shoppers left the queue ({rate:.0f}%)",
                          at=h["hour"], zone="BILLING", value=round(rate, 1), threshold="30%",
                          action="Reduce wait: express lane / mobile POS"))

    # 3. Conversion drop vs. store average
    tot_e = sum(h["entries"] for h in hourly)
    tot_c = sum(h["conversions"] for h in hourly)
    base = tot_c / tot_e if tot_e else 0
    for h in hourly:
        if h["entries"] >= 5 and base > 0 and h["conversions"] / h["entries"] < 0.5 * base:
            out.append(_a("CONVERSION_DROP", "warning", "Conversion drop",
                          f"Hourly conversion {h['conversion_rate']}% vs store avg {100 * base:.1f}%",
                          at=h["hour"], value=h["conversion_rate"], threshold=round(50 * base, 1),
                          action="Check staffing, stock-outs or billing delays in this hour"))

    # 4. POS transactions with no CCTV billing activity
    unmatched = [m for m in matches if not m.session_id]
    if transactions and unmatched:
        share = len(unmatched) / len(transactions)
        sev = "critical" if share > 0.5 else "warning" if share > 0.2 else "info"
        out.append(_a("UNMATCHED_POS", sev, "POS transactions without CCTV billing activity",
                      f"{len(unmatched)}/{len(transactions)} transactions had no billing activity "
                      f"in the match window (e.g. {', '.join(m.transaction_id for m in unmatched[:5])})",
                      value=round(100 * share, 1), threshold="20%",
                      action="Verify billing-camera coverage and DVR/POS clock sync"))

    # 5. Dead (cold) zones: configured product zones never visited
    if len(sessions) >= 10:
        visited = {v.zone_id for s in sessions for v in s.zone_visits}
        for zid in product_zone_ids(store_id):
            if zid not in visited:
                out.append(_a("DEAD_ZONE", "info", "Zone with zero visits",
                              f"No shopper visited {zid} in the selected period", zone=zid, value=0,
                              action="Re-merchandise, add signage or verify camera polygon"))

    # 6. Unusually long sessions (possible unflagged staff / loitering)
    durs = sorted(s.duration_s for s in sessions if s.closed)
    if len(durs) >= 8:
        q1, q3 = durs[len(durs) // 4], durs[(3 * len(durs)) // 4]
        cut = q3 + 3 * (q3 - q1)
        long_s = [s for s in sessions if s.closed and s.duration_s > cut]
        for s in long_s[:10]:
            out.append(_a("LONG_SESSION", "info", "Unusually long visit",
                          f"{s.visitor_id} stayed {s.duration_s / 60:.1f} min", at=s.start,
                          value=round(s.duration_s, 0), threshold=round(cut, 0),
                          action="Check if this is staff/vendor; review for loss prevention"))

    # 7. Data-quality signals
    if build.total_events:
        orphan_share = build.orphan_events / build.total_events
        if orphan_share > 0.05:
            out.append(_a("ORPHAN_EVENTS", "warning", "Events outside any session",
                          f"{build.orphan_events} events ({100 * orphan_share:.0f}%) had no open "
                          "ENTRY session; entrance camera may be missing crossings",
                          value=round(100 * orphan_share, 1), threshold="5%",
                          action="Re-calibrate CAM3 entry line / check Re-ID linking"))
    unclosed = sum(1 for s in sessions if not s.closed)
    if sessions and unclosed / len(sessions) > 0.2:
        out.append(_a("UNCLOSED_SESSIONS", "info", "Sessions without EXIT",
                      f"{unclosed}/{len(sessions)} sessions never recorded an EXIT",
                      value=unclosed, action="Check exit-direction detection on CAM3"))

    # 8. Activity outside opening hours
    oh = store_layout(store_id).get("open_hours")
    if oh:
        o, c = time.fromisoformat(oh["open"]), time.fromisoformat(oh["close"])
        outside = [s for s in sessions if not (o <= s.start.time() <= c)]
        if outside:
            out.append(_a("AFTER_HOURS", "warning", "Activity outside opening hours",
                          f"{len(outside)} visits started outside {oh['open']}-{oh['close']}",
                          at=outside[0].start, value=len(outside),
                          action="Review footage; may be staff or a security event"))

    # 9. Camera silence while store is active (possible camera outage)
    cams = defaultdict(list)
    for e in events:
        cams[e["camera_id"]].append(e["timestamp"])
    if len(cams) > 1:
        lo = min(min(v) for v in cams.values())
        hi = max(max(v) for v in cams.values())
        for cam, ts in cams.items():
            ts = sorted([lo] + ts + [hi])
            gaps = [(b - a, a) for a, b in zip(ts, ts[1:])]
            gap, at = max(gaps)
            if gap > timedelta(minutes=60):
                out.append(_a("CAMERA_SILENCE", "warning", "Camera silent for a long period",
                              f"{cam} produced no events for {gap.total_seconds() / 60:.0f} min "
                              "while other cameras were active", at=at, value=round(gap.total_seconds() / 60),
                              threshold=60, action="Check camera health / stream ingestion"))

    order = {"critical": 0, "warning": 1, "info": 2}
    out.sort(key=lambda a: (order[a["severity"]], a["timestamp"] or datetime.min))
    return out
