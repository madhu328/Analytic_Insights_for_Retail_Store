"""Plain-English executive summary: turns KPIs into findings + recommended actions.

Rules are deterministic and transparent (no LLM), so every sentence can be traced back to
a number on the dashboard. Each insight: level (good / info / warning / critical), title,
detail (what the data says) and action (what the store team should do).
"""
from __future__ import annotations

from .funnel import compute_funnel
from .heatmap import compute_heatmap
from .insights import hesitation_insights, journey_insights, shelf_quadrant
from .metrics import compute_metrics


def _i(level, title, detail, action="", metric=None):
    return {"level": level, "title": title, "detail": detail, "action": action, "metric": metric}


def data_health(m: dict) -> dict:
    """0-100 score: how far the CCTV + POS feeds can be trusted for this period."""
    dq = m["data_quality"]
    total = dq["total_events"] or 1
    sessions = m["sessions"] or 1
    tx = m["transactions"] or 1
    checks = [
        ("Every zone event belongs to a visit", dq["orphan_events"] / total, 0.05,
         f"{dq['orphan_events']} events had no matching store entry"),
        ("Every visit has an exit", dq["unclosed_sessions"] / sessions, 0.20,
         f"{dq['unclosed_sessions']} visits never recorded an exit"),
        ("Sales are linked to CCTV billing activity",
         dq["unmatched_transactions"] / tx if m["transactions"] else 0.0, 0.20,
         f"{dq['unmatched_transactions']} of {m['transactions']} sales had no billing activity nearby"),
        ("Low-confidence detections are rare", dq["low_confidence_events_excluded"] / total, 0.10,
         f"{dq['low_confidence_events_excluded']} low-confidence events excluded"),
    ]
    out, score = [], 100.0
    for name, share, limit, detail in checks:
        ok = share <= limit
        score -= min(25.0, 100 * share / 2) if not ok else min(5.0, 100 * share / 4)
        out.append({"check": name, "ok": ok, "share_pct": round(100 * share, 1),
                    "limit_pct": round(100 * limit, 1), "detail": detail})
    if dq["staff_events_excluded"]:
        out.append({"check": "Staff excluded from customer KPIs", "ok": True, "share_pct": None,
                    "limit_pct": None, "detail": f"{dq['staff_events_excluded']} staff events removed"})
    score = max(0.0, round(score))
    grade = "Reliable" if score >= 85 else "Usable with caution" if score >= 60 else "Unreliable"
    return {"score": score, "grade": grade, "checks": out}


def build_summary(a, anomalies: list[dict]) -> dict:
    m = compute_metrics(a.build, a.matches, a.transactions)
    insights: list[dict] = []
    if not a.build.sessions:
        insights.append(_i("warning", "No shopper visits recorded",
                           "No ENTRY events were captured for this period, so footfall and conversion "
                           "cannot be measured.", "Check that the entrance camera (CAM3) is processed and ingested."))
        return {"headline": "No CCTV visits in this period", "insights": insights,
                "health": data_health(m), "metrics": m}

    # 1. Conversion + biggest funnel leak
    f = compute_funnel(a.build, "visitor")["stages"]
    leaks = [(f[i - 1]["stage"], s["stage"], f[i - 1]["count"] - s["count"], f[i - 1]["count"])
             for i, s in enumerate(f) if i]
    frm, to, lost, base = max(leaks, key=lambda x: x[2])
    cr = m["conversion_rate"]
    insights.append(_i("good" if cr >= 50 else "info" if cr >= 25 else "warning",
                       f"{cr}% of visitors bought something",
                       f"{m['converted_visitors']} of {m['visitors']} unique visitors completed a purchase "
                       f"(₹{m['revenue_inr']:,.0f} in {m['transactions']} bills, average ₹{m['avg_basket_inr']:,.0f}).",
                       metric=cr))
    if lost > 0:
        actions = {"Engaged": "Make the entrance more inviting: promo displays and greeting staff near the door.",
                   "Billing": "Shoppers browse but do not head to billing: add assisted selling, testers and "
                              "price clarity in the busiest zones.",
                   "Purchased": "Shoppers reach billing but leave without paying: cut waiting time and keep "
                                "the queue moving."}
        insights.append(_i("warning" if lost / max(base, 1) > 0.2 else "info",
                           f"Biggest drop-off: {frm} → {to}",
                           f"{lost} shoppers ({100 * lost / max(base, 1):.0f}%) were lost between "
                           f"'{frm}' and '{to}'.", actions.get(to, "")))

    # 2. Peak hour
    hourly = [h for h in a.hourly if h["entries"]]
    if hourly:
        peak = max(hourly, key=lambda h: h["entries"])
        rpk = max(a.hourly, key=lambda h: h["revenue_inr"])
        insights.append(_i("info", f"Busiest hour: {peak['hour']:%H:00}",
                           f"{peak['entries']} shoppers entered between {peak['hour']:%H:00} and "
                           f"{peak['hour'].hour + 1:02d}:00; the highest-revenue hour was {rpk['hour']:%H:00} "
                           f"(₹{rpk['revenue_inr']:,.0f}).",
                           "Schedule breaks outside this window and put the most staff on the floor then."))

    # 3. Zones
    hm = compute_heatmap(a.build, a.store_id)
    visited = [z for z in hm["zones"] if z["visits"]]
    if visited:
        top = visited[0]
        insights.append(_i("good", f"Top zone: {top['name']}",
                           f"{top['name']} drew {top['traffic_share_pct']}% of all zone visits with an average "
                           f"dwell of {top['avg_dwell_s']:.0f}s. The {hm['summary']['hot']} hot zones together take "
                           f"{hm['summary']['hot_zone_traffic_share_pct']}% of traffic.",
                           "Keep it fully stocked; place new launches and promotions next to it."))
    cold = [z["name"] for z in hm["zones"] if z["zone_class"] == "cold"]
    if cold and visited:
        insights.append(_i("info", f"{len(cold)} cold zone(s) get little attention",
                           f"Least visited: {', '.join(cold[:4])}.",
                           "Improve signage or move these brands closer to a hot zone."))
    sq = shelf_quadrant(a.build, a.store_id)["zones"]
    problem = [z for z in sq if z["quadrant"] == "problem"]
    if problem:
        names = ", ".join(f"{z['zone_id']} ({z['conversion_pct']:.0f}% convert)" for z in problem[:3])
        insights.append(_i("warning", "Popular shelves that do not convert",
                           f"Many shoppers look but few buy at: {names}.",
                           "Check prices, stock-outs, testers and staff support at these shelves."))

    # 4. Queue
    q = m["queue"]
    if q["joins"]:
        lvl = "critical" if q["abandonment_rate"] > 25 else "warning" if q["abandonment_rate"] > 10 else "good"
        insights.append(_i(lvl, f"{q['abandonment_rate']}% of shoppers left the billing queue",
                           f"{q['abandons']} of {q['joins']} queue joins were abandoned; average wait to pay "
                           f"{q['avg_wait_to_pay_s'] / 60:.1f} min, longest queue {q['max_queue_depth']} people.",
                           "Open a second counter when the queue passes 5 people." if lvl != "good" else
                           "Billing is flowing well; keep the current counter staffing."))

    # 5. Hesitation + journeys
    hz = hesitation_insights(a.build)["zones"]
    if hz:
        z = hz[0]
        insights.append(_i("info", f"Shoppers hesitate most at {z['zone_id']}",
                           f"{z['hesitating_sessions']} visits lingered or came back to {z['zone_id']}; "
                           f"{z['hesitation_conversion_pct']:.0f}% of them bought.",
                           "Station a beauty advisor here at peak hours."))
    seg = {s["segment"]: s for s in journey_insights(a.build, a.store_id)["segments"]}
    if seg.get("browsing", {}).get("sessions") and seg.get("efficient", {}).get("sessions"):
        b, e = seg["browsing"]["conversion_rate"], seg["efficient"]["conversion_rate"]
        if b > e:
            insights.append(_i("info", "Shoppers who explore buy more",
                               f"Visitors who browse several areas convert at {b:.0f}%, against {e:.0f}% for "
                               "those who go straight to one shelf.",
                               "Use the layout to lead shoppers past more zones (end-cap displays, trails)."))

    # 6. Anomalies
    crit = [x for x in anomalies if x["severity"] == "critical"]
    if crit:
        insights.append(_i("critical", f"{len(crit)} critical alert(s)", crit[0]["detail"],
                           crit[0]["suggested_action"]))

    health = data_health(m)
    if health["score"] < 60:
        bad = "; ".join(c["detail"] for c in health["checks"] if not c["ok"])
        insights.insert(0, _i("critical", "Treat these numbers with caution",
                              f"Data health is {health['score']}/100 ({bad}).",
                              "Fix camera coverage or POS/CCTV clock alignment before acting on this period."))
    order = {"critical": 0, "warning": 1, "info": 2, "good": 3}
    head = next(i["title"] for i in insights if "bought" in i["title"]) if insights else ""
    insights.sort(key=lambda x: order[x["level"]])
    return {"headline": head, "insights": insights, "health": health, "metrics": m}
