from tests.conftest import STORE, demo_scenario, ev, tx

P = {"store_id": STORE, "date": "2026-04-10"}


def seed(client):
    events, txs = demo_scenario()
    assert client.post("/events/ingest", json=events).json()["inserted"] == len(events)
    assert client.post("/pos/ingest", json={"transactions": txs}).json()["inserted"] == 2
    return events, txs


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_ingest_is_idempotent_and_validates(client):
    events, _ = seed(client)
    r = client.post("/events/ingest", json={"events": events}).json()
    assert r["inserted"] == 0 and r["duplicates"] == len(events)
    bad = [ev("VIS_9", "TELEPORT", 1), {**ev("VIS_9", "ENTRY", 1), "confidence": 1.7},
           {**ev("VIS_9", "ZONE_ENTER", 1), "zone_id": None}, ev("VIS_9", "ENTRY", 2)]
    r = client.post("/events/ingest", json=bad).json()
    assert r == {**r, "received": 4, "inserted": 1, "rejected": 3}
    assert {e["index"] for e in r["errors"]} == {0, 1, 2}


def test_demo_validation_metrics(client):
    seed(client)
    m = client.get("/metrics", params=P).json()
    assert m["visitors"] == 3 and m["sessions"] == 3
    assert m["converted_visitors"] == 2
    assert m["conversion_rate"] == 66.7
    assert m["revenue_inr"] == 2148.50
    assert m["data_quality"]["staff_events_excluded"] == 1


def test_funnel_heatmap_anomalies_endpoints(client):
    seed(client)
    f = client.get("/funnel", params=P).json()["stages"]
    assert [s["count"] for s in f] == [3, 3, 2, 2]
    h = client.get("/heatmap", params=P).json()
    assert all(0 <= z["engagement_score"] <= 100 for z in h["zones"])
    assert {"LAKME", "PILGRIM", "MARS"} <= {z["zone_id"] for z in h["zones"] if z["visits"]}
    assert "anomalies" in client.get("/anomalies", params=P).json()
    conv = client.get("/conversions", params=P).json()["matches"]
    assert {m["transaction_id"]: m["visitor_id"] for m in conv} == {"TXN_A": "VIS_1", "TXN_B": "VIS_2"}
    s = client.get("/sessions", params={**P, "converted": True}).json()
    assert s["total"] == 2
    d = client.get(f"/sessions/{s['sessions'][0]['session_id']}", params=P).json()
    assert d["events"] and d["zone_visits"]
    for route in ("/timeseries", "/insights/journeys", "/insights/hesitation", "/insights/shelf", "/forecast"):
        assert client.get(route, params=P).status_code == 200, route
    stores = client.get("/stores").json()
    assert stores[0]["store_id"] == STORE and stores[0]["transactions"] == 2


def test_same_transaction_id_in_two_stores(client):
    t = tx("TXN_0001", 1, 10.0)
    r = client.post("/pos/ingest", json=[t, {**t, "store_id": "STORE_BLR_003"}]).json()
    assert r["inserted"] == 2
