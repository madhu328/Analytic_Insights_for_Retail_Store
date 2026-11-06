from datetime import datetime, timedelta

from app.analytics.anomalies import detect_anomalies
from app.analytics.funnel import compute_funnel
from app.analytics.heatmap import compute_heatmap
from app.analytics.insights import hesitation_insights, journey_insights, shelf_quadrant
from app.analytics.metrics import compute_metrics
from app.analytics.pos_match import match_transactions
from app.analytics.sessions import build_sessions
from app.analytics.timeseries import hourly_series
from app.schemas import Event
from tests.conftest import STORE, demo_scenario, ev, tx


def norm(raw):
    out = []
    for r in raw:
        e = Event.model_validate(r).model_dump()
        e["event_type"] = e["event_type"].value
        e["event_id"] = str(e["event_id"])
        out.append(e)
    return out


def norm_tx(raw):
    return [{**t, "timestamp": datetime.fromisoformat(t["timestamp"].rstrip("Z"))} for t in raw]


def test_reentry_opens_new_session_but_same_visitor():
    events = norm([ev("V", "ENTRY", 0), ev("V", "EXIT", 5), ev("V", "REENTRY", 30),
                   ev("V", "ZONE_ENTER", 31, "LAKME"), ev("V", "EXIT", 40)])
    b = build_sessions(events, STORE)
    assert len(b.sessions) == 2 and len(b.visitors) == 1
    assert b.sessions[1].is_reentry and b.sessions[1].engaged
    f = compute_funnel(b, "visitor")["stages"]
    assert f[0]["count"] == 1
    assert compute_funnel(b, "session")["stages"][0]["count"] == 2


def test_orphan_events_do_not_create_sessions():
    events = norm([ev("V", "ZONE_ENTER", 0, "LAKME"), ev("V", "EXIT", 1), ev("W", "ENTRY", 2)])
    b = build_sessions(events, STORE)
    assert len(b.sessions) == 1 and b.orphan_events == 2


def test_dwell_prefers_zone_dwell_then_enter_exit():
    events = norm([ev("V", "ENTRY", 0), ev("V", "ZONE_ENTER", 1, "A"), ev("V", "ZONE_DWELL", 2, "A", 45000),
                   ev("V", "ZONE_EXIT", 3, "A"), ev("V", "ZONE_ENTER", 4, "B"), ev("V", "ZONE_EXIT", 6, "B")])
    s = build_sessions(events, STORE).sessions[0]
    assert s.dwell_by_zone() == {"A": 45.0, "B": 120.0}


def test_pos_matching_window_one_to_one_and_abandon():
    events = norm([
        ev("A", "ENTRY", 0), ev("A", "BILLING_QUEUE_JOIN", 10, "BILLING"), ev("A", "EXIT", 20),
        ev("B", "ENTRY", 0), ev("B", "BILLING_QUEUE_JOIN", 11, "BILLING"), ev("B", "EXIT", 20),
        ev("C", "ENTRY", 0), ev("C", "BILLING_QUEUE_JOIN", 12, "BILLING"),
        ev("C", "BILLING_QUEUE_ABANDON", 13, "BILLING"), ev("C", "EXIT", 20),
        ev("D", "ENTRY", 0), ev("D", "BILLING_QUEUE_JOIN", 1, "BILLING"), ev("D", "EXIT", 20),
    ])
    b = build_sessions(events, STORE)
    m = match_transactions(b.sessions, norm_tx([tx("T1", 14, 100), tx("T2", 15, 50)]), 5, 1)
    got = {x.transaction_id: x.visitor_id for x in m}
    assert got == {"T1": "A", "T2": "B"}  # FIFO; C abandoned, D outside the window
    assert all(0 < x.confidence <= 1 for x in m)


def test_unmatched_transaction_is_reported():
    events = norm([ev("A", "ENTRY", 0), ev("A", "EXIT", 3)])
    b = build_sessions(events, STORE)
    txs = norm_tx([tx("T1", 30, 100)])
    m = match_transactions(b.sessions, txs)
    met = compute_metrics(b, m, txs)
    assert met["converted_visitors"] == 0 and met["data_quality"]["unmatched_transactions"] == 1
    kinds = {a["type"] for a in detect_anomalies(b, m, txs, hourly_series(b, m, txs), events, STORE)}
    assert "UNMATCHED_POS" in kinds


def test_queue_congestion_anomaly():
    raw = []
    for i in range(10):
        raw += [ev(f"V{i}", "ENTRY", i), ev(f"V{i}", "BILLING_QUEUE_JOIN", i + 1, "BILLING", queue_depth=i + 3),
                ev(f"V{i}", "EXIT", i + 5)]
    events = norm(raw)
    b = build_sessions(events, STORE)
    m = match_transactions(b.sessions, [])
    a = detect_anomalies(b, m, [], hourly_series(b, m, []), events, STORE)
    q = [x for x in a if x["type"] == "QUEUE_CONGESTION"]
    assert q and q[0]["value"] == 12 and q[0]["severity"] == "critical"


def test_heatmap_classes_and_shelf_quadrant():
    raw_events, txs = demo_scenario()
    events, txs = norm(raw_events), norm_tx(txs)
    b = build_sessions(events, STORE)
    match_transactions(b.sessions, txs)
    h = compute_heatmap(b, STORE)
    classes = {z["zone_id"]: z["zone_class"] for z in h["zones"]}
    assert classes["ALPS"] == "cold"  # configured but never visited
    q = {z["zone_id"]: z["quadrant"] for z in shelf_quadrant(b, STORE)["zones"]}
    assert q["MARS"] in {"obsolete", "problem"} and q["LAKME"] in {"star", "potential"}


def test_pci_efficient_vs_roaming():
    straight = norm([ev("A", "ENTRY", 0), ev("A", "ZONE_ENTER", 1, "PILGRIM"), ev("A", "ZONE_ENTER", 2, "GOODVIBES"),
                     ev("A", "EXIT", 5)])
    zigzag = norm([ev("B", "ENTRY", 0)] + [ev("B", "ZONE_ENTER", i + 1, z) for i, z in
                                          enumerate(["PILGRIM", "ALPS", "PILGRIM", "ALPS", "PILGRIM"])]
                  + [ev("B", "EXIT", 9)])
    j = journey_insights(build_sessions(straight + zigzag, STORE), STORE)
    pci = {r["session_id"]: r for r in j["sessions"]}
    assert pci["A_S1"]["segment"] == "efficient"
    assert pci["B_S1"]["pci"] > 2 * pci["A_S1"]["pci"] and pci["B_S1"]["segment"] != "efficient"


def test_hesitation_levels():
    events = norm([ev("A", "ENTRY", 0), ev("A", "ZONE_ENTER", 1, "LAKME"), ev("A", "ZONE_DWELL", 2, "LAKME", 70000),
                   ev("A", "ZONE_ENTER", 3, "MARS"), ev("A", "ZONE_ENTER", 4, "LAKME"),
                   ev("A", "ZONE_DWELL", 5, "LAKME", 200000), ev("A", "EXIT", 9)])
    h = hesitation_insights(build_sessions(events, STORE))
    z = {r["zone_id"]: r for r in h["zones"]}
    assert z["LAKME"]["severe"] == 1 and "MARS" not in z


def test_forecast_shapes():
    from app.analytics.forecast import forecast, holt_damped
    fc, fitted = holt_damped([1, 2, 3, 4, 5], 3)
    assert len(fc) == 3 and fc[0] > 4 and len(fitted) == 5
    hourly = [{"hour": datetime(2026, 4, 10, 10) + timedelta(hours=i), "entries": e, "visitors": e,
               "transactions": e // 2, "revenue_inr": 500.0 * e, "avg_dwell_s": 60.0 + i, "queue_joins": e // 2}
              for i, e in enumerate([3, 5, 8, 6, 9, 12, 10])]
    out = forecast(hourly, 4)
    assert len(out["footfall"]) == 4 and out["fusion_model"]["r2"] > 0.9
