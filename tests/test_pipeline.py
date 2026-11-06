import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from app.schemas import Event
from pipeline.emit import EventEmitter
from pipeline.geometry import anchor_point, point_in_polygon, side_of_line
from pipeline.processors import EntryExitProcessor, QueueProcessor, ZoneDwellProcessor, build_processor
from pipeline.reid import ReIDManager, SessionStore
from pipeline.tracking import Track

T0 = datetime(2026, 4, 10, 20, 10, tzinfo=timezone.utc)
SQUARE = [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]


def box_at(x, y, w=0.06, h=0.2):
    """bbox whose centre is (x, y)."""
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def feet_at(x, y, w=0.06, h=0.2):
    return (x - w / 2, y - h, x + w / 2, y)


def run(proc, frames, fps=5):
    for i, tracks in enumerate(frames):
        proc.update(T0 + timedelta(seconds=i / fps), tracks)
    return proc.finalize()


def test_geometry():
    assert point_in_polygon((0.25, 0.25), SQUARE) and not point_in_polygon((0.75, 0.25), SQUARE)
    assert side_of_line((0, 1), (0, 0), (1, 0)) * side_of_line((0, -1), (0, 0), (1, 0)) < 0
    assert anchor_point((0, 0, 1, 1)) == (0.5, 1) and anchor_point((0, 0, 1, 1), "center") == (0.5, 0.5)


def test_zone_dwell_enter_dwell_exit_with_jitter():
    cam = {"camera_id": "CAM_SHELF_01", "anchor": "center",
           "zones": [{"zone_id": "LAKME", "polygon": SQUARE}, {"zone_id": "MARS", "polygon": [[0.5, 0], [1, 0], [1, 0.5], [0.5, 0.5]]}]}
    em = EventEmitter("S", "CAM_SHELF_01")
    p = ZoneDwellProcessor(cam, em, staff_presence_ratio=0)
    frames = [[Track(7, box_at(0.25, 0.25), 0.9)]] * 50  # 10 s in LAKME
    frames += [[Track(7, box_at(0.25, 0.9), 0.9)]] * 3     # 0.6 s jitter out -> no exit
    frames += [[Track(7, box_at(0.25, 0.25), 0.9)]] * 25   # 5 s more
    frames += [[Track(7, box_at(0.75, 0.25), 0.9)]] * 30   # move to MARS
    frames += [[]] * 30                                    # leaves view
    evs = run(p, frames)
    kinds = [(e["event_type"], e["zone_id"]) for e in evs]
    assert kinds.count(("ZONE_ENTER", "LAKME")) == 1 and kinds.count(("ZONE_EXIT", "LAKME")) == 1
    dwell = [e for e in evs if e["event_type"] == "ZONE_DWELL" and e["zone_id"] == "LAKME"][0]
    assert 15000 <= dwell["dwell_ms"] <= 16000
    assert ("ZONE_ENTER", "MARS") in kinds and ("ZONE_EXIT", "MARS") in kinds
    for e in evs:  # every emitted event satisfies the API contract
        Event.model_validate(e)


def test_entry_exit_and_reentry_via_reid_gallery():
    cam = {"camera_id": "CAM_ENTRY_01", "anchor": "bottom_center",
           "entry_line": [[0, 0.5], [1, 0.5]], "inside_point": [0.5, 0.1]}
    em = EventEmitter("S", "CAM_ENTRY_01")
    reid = ReIDManager(SessionStore())

    class FakeEmbedder:  # same appearance for the same person
        def __call__(self, crop):
            import numpy as np
            v = np.array(crop, dtype=float)
            return v / np.linalg.norm(v)

    p = EntryExitProcessor(cam, em, reid=reid, embedder=FakeEmbedder(), staff_presence_ratio=0)
    person = [1.0, 0.0, 0.2]
    walk_in = [[Track(1, feet_at(0.5, y), 0.9, crop=person)] for y in (0.9, 0.8, 0.6, 0.4, 0.3, 0.2)]
    walk_out = [[Track(1, feet_at(0.5, y), 0.9, crop=person)] for y in (0.2, 0.4, 0.6, 0.8, 0.9)]
    gap = [[]] * 30
    back_in = [[Track(5, feet_at(0.5, y), 0.9, crop=person)] for y in (0.9, 0.7, 0.4, 0.3)]
    other = [[Track(9, feet_at(0.5, y), 0.9, crop=[0.0, 1.0, 0.0])] for y in (0.9, 0.7, 0.4, 0.3)]
    evs = run(p, walk_in + walk_out + gap + back_in + other)
    seq = [(e["event_type"], e["visitor_id"]) for e in evs]
    vid = seq[0][1]
    assert seq[:3] == [("ENTRY", vid), ("EXIT", vid), ("REENTRY", vid)]
    assert seq[3][0] == "ENTRY" and seq[3][1] != vid


def test_queue_join_abandon_and_payment():
    cam = {"camera_id": "CAM_BILLING_01", "anchor": "bottom_center",
           "queue_polygon": {"zone_id": "BILLING", "polygon": SQUARE},
           "payment_polygon": {"zone_id": "PAYMENT", "polygon": [[0.5, 0], [1, 0], [1, 0.5], [0.5, 0.5]]}}
    em = EventEmitter("S", "CAM_BILLING_01")
    p = QueueProcessor(cam, em, staff_presence_ratio=0)
    payer_q = [Track(1, feet_at(0.2, 0.3), 0.9)]
    leaver_q = [Track(2, feet_at(0.3, 0.4), 0.9)]
    frames = [payer_q + leaver_q] * 40                                         # both queue 8 s
    frames += [[Track(1, feet_at(0.75, 0.3), 0.9), Track(2, feet_at(0.3, 0.9), 0.9)]] * 30  # pay / walk away
    frames += [[]] * 25
    evs = run(p, frames)
    joins = [e for e in evs if e["event_type"] == "BILLING_QUEUE_JOIN"]
    assert len(joins) == 2 and joins[0]["metadata"]["queue_depth"] == 2
    ab = [e for e in evs if e["event_type"] == "BILLING_QUEUE_ABANDON"]
    assert [e["visitor_id"] for e in ab] == ["VIS_B2"] and ab[0]["dwell_ms"] >= 5000
    assert any(e["event_type"] == "ZONE_ENTER" and e["zone_id"] == "PAYMENT" and e["visitor_id"] == "VIS_B1"
               for e in evs)


def test_staff_flagging_and_jsonl(tmp_path):
    cam = {"camera_id": "CAM_SHELF_02", "anchor": "center", "zones": [{"zone_id": "LAKME", "polygon": SQUARE}]}
    out = tmp_path / "cam2.jsonl"
    p = build_processor({**cam, "processor": "cam2"}, EventEmitter("S", "CAM_SHELF_02", out))
    frames = [[Track(1, box_at(0.25, 0.25), 0.9)] + ([Track(2, box_at(0.25, 0.3), 0.8)] if 100 < i < 160 else [])
              for i in range(600)]  # 120 s clip; track 1 present throughout
    run(p, frames)
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert {e["visitor_id"] for e in lines if e["is_staff"]} == {"VIS_1"}
    assert any(e["visitor_id"] == "VIS_2" and not e["is_staff"] for e in lines)


def test_pos_loader_reproduces_purplle_csv():
    from pipeline.pos_loader import aggregate, load_items, to_pos_contract
    root = Path(__file__).resolve().parent.parent
    agg = aggregate(load_items(root / "data/pos/brigade_bangalore_10_april_26_anonymised.csv"))
    pos = to_pos_contract(agg, {"ST1008": "STORE_BLR_002"})
    expected = pd.read_csv(root / "data/pos/purpple_pos_transactions.csv")
    assert sorted(pos["transaction_id"]) == sorted(expected["transaction_id"])
    assert round(pos["basket_value_inr"].sum(), 2) == round(expected["basket_value_inr"].sum(), 2)
    assert "customer_number" not in agg.columns and "customer_name" not in agg.columns


def test_bundled_events_match_contract():
    root = Path(__file__).resolve().parent.parent
    files = list((root / "data").rglob("*.jsonl"))
    assert files
    for f in files:
        for line in f.read_text().splitlines():
            Event.model_validate_json(line)
