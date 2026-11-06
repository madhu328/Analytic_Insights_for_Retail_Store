import uuid
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app, get_engine

STORE = "STORE_BLR_002"
T0 = datetime(2026, 4, 10, 12, 0, 0)


def ev(visitor, etype, minute, zone=None, dwell_ms=0, camera="CAM_ENTRY_01", **meta):
    return {
        "event_id": str(uuid.uuid4()), "store_id": STORE, "camera_id": camera, "visitor_id": visitor,
        "event_type": etype, "timestamp": (T0 + timedelta(minutes=minute)).isoformat() + "Z",
        "zone_id": zone, "dwell_ms": dwell_ms, "is_staff": meta.pop("is_staff", False),
        "confidence": meta.pop("confidence", 0.9), "metadata": meta,
    }


def tx(tid, minute, value):
    return {"store_id": STORE, "transaction_id": tid,
            "timestamp": (T0 + timedelta(minutes=minute)).isoformat() + "Z", "basket_value_inr": value}


def demo_scenario():
    """The 'Demo Validation Results' scenario: 3 visitors, 3 sessions, 2 converted, INR 2,148.50."""
    events = [
        ev("VIS_1", "ENTRY", 0),
        ev("VIS_1", "ZONE_ENTER", 1, "LAKME", camera="CAM_SHELF_02"),
        ev("VIS_1", "ZONE_DWELL", 3, "LAKME", 120000, camera="CAM_SHELF_02"),
        ev("VIS_1", "ZONE_EXIT", 3, "LAKME", camera="CAM_SHELF_02"),
        ev("VIS_1", "BILLING_QUEUE_JOIN", 4, "BILLING", camera="CAM_BILLING_01", queue_depth=2),
        ev("VIS_1", "EXIT", 9),
        ev("VIS_2", "ENTRY", 2),
        ev("VIS_2", "ZONE_ENTER", 3, "PILGRIM", camera="CAM_SHELF_01"),
        ev("VIS_2", "ZONE_EXIT", 5, "PILGRIM", camera="CAM_SHELF_01"),
        ev("VIS_2", "BILLING_QUEUE_JOIN", 6, "BILLING", camera="CAM_BILLING_01", queue_depth=3),
        ev("VIS_2", "EXIT", 12),
        ev("VIS_3", "ENTRY", 5),
        ev("VIS_3", "ZONE_ENTER", 6, "MARS", camera="CAM_SHELF_02"),
        ev("VIS_3", "ZONE_EXIT", 7, "MARS", camera="CAM_SHELF_02"),
        ev("VIS_3", "EXIT", 8),
        ev("STAFF_1", "ENTRY", 0, is_staff=True),
    ]
    txs = [tx("TXN_A", 7, 1248.50), tx("TXN_B", 9, 900.00)]
    return events, txs


@pytest.fixture()
def engine(tmp_path):
    get_engine_cached = db.get_engine
    eng = get_engine_cached(f"sqlite:///{tmp_path / 'test.db'}")
    yield eng
    eng.dispose()


@pytest.fixture()
def client(engine):
    app.dependency_overrides[get_engine] = lambda: engine
    yield TestClient(app)
    app.dependency_overrides.clear()
