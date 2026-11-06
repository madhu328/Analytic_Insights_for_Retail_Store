from pathlib import Path

from training.build_dataset import build, split_groups
from training.evaluate_events import evaluate


def e(t, ts, cam="CAM_ENTRY_01", zone=None, dwell=0):
    return {"event_type": t, "timestamp": f"2026-04-10T20:10:{ts:04.1f}Z", "camera_id": cam,
            "zone_id": zone, "dwell_ms": dwell}


def test_event_evaluation_perfect_and_tolerance():
    truth = [e("ENTRY", 1), e("ENTRY", 20), e("ZONE_ENTER", 30, "CAM_SHELF_01", "LAKME"),
             e("ZONE_DWELL", 40, "CAM_SHELF_01", "LAKME", 10000)]
    pred = [e("ENTRY", 2.5), e("ENTRY", 20.4), e("ZONE_ENTER", 31, "CAM_SHELF_01", "lakme"),
            e("ZONE_DWELL", 41, "CAM_SHELF_01", "LAKME", 11000)]
    r = evaluate(pred, truth)
    assert r["overall"]["f1"] == 1.0 and r["passed"]["event_f1"]
    assert r["dwell"]["within_20pct"] == 1.0 and r["counting_accuracy"]["entries"] == 1.0


def test_event_evaluation_counts_errors():
    truth = [e("ENTRY", 1), e("ENTRY", 20), e("EXIT", 40)]
    pred = [e("ENTRY", 1), e("ENTRY", 26),            # 6 s late -> outside 2 s tolerance
            e("ZONE_ENTER", 30, "CAM_SHELF_01", "MARS")]  # wrong type -> FP
    r = evaluate(pred, truth)
    o = r["overall"]
    assert (o["tp"], o["fp"], o["fn"]) == (1, 2, 2)
    assert r["per_event_type"]["EXIT"]["recall"] == 0.0
    assert not r["passed"]["event_f1"]


def test_dataset_split_is_by_video(tmp_path: Path):
    imgs, labs = tmp_path / "img", tmp_path / "lab"
    imgs.mkdir()
    labs.mkdir()
    for v in range(10):
        for f in range(5):
            (imgs / f"store1_cam{v}__{f:06d}.jpg").write_bytes(b"x")
            if f % 2 == 0:
                (labs / f"store1_cam{v}__{f:06d}.txt").write_text("0 0.5 0.5 0.1 0.3\n")
    stats = build(imgs, labs, tmp_path / "ds", val=0.2, test=0.2)
    assert sum(s["images"] for s in stats.values()) == 50
    assert stats["test"]["videos"] == 2 and stats["val"]["videos"] == 2
    seen = {}
    for split in ("train", "val", "test"):
        for p in (tmp_path / "ds" / "images" / split).iterdir():
            seen.setdefault(p.stem.split("__")[0], set()).add(split)
    assert all(len(s) == 1 for s in seen.values())  # no video leaks across splits
    assert "names:\n  0: person" in (tmp_path / "ds" / "data.yaml").read_text()
    assert split_groups(["a", "b", "c"], 0.15, 0.15) == split_groups(["c", "b", "a"], 0.15, 0.15)
