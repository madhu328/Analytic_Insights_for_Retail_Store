"""Ground-truth event labelling tool (Streamlit).

  streamlit run training/label_events_app.py -- --videos videos/

Scrub through a clip frame by frame (zones and lines from store_layout.json are drawn on
top), and click the event that happens at that moment. Rows are appended to
``training/ground_truth/<store>.csv`` in the format ``training/evaluate_events.py`` reads.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
EVENTS = ["ENTRY", "EXIT", "REENTRY", "ZONE_ENTER", "ZONE_EXIT", "ZONE_DWELL", "BILLING_QUEUE_JOIN",
          "BILLING_QUEUE_ABANDON"]

ap = argparse.ArgumentParser()
ap.add_argument("--videos", default="videos")
args, _ = ap.parse_known_args()

st.set_page_config(page_title="Event labelling", layout="wide")
st.title("Ground-truth event labelling")
layout = json.loads((ROOT / "config" / "store_layout.json").read_text())
stores = {s["store_id"]: s for s in layout["stores"]}
vids = sorted(str(p) for p in Path(args.videos).rglob("*") if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"})
if not vids:
    st.warning(f"No videos found under {args.videos}. Pass `-- --videos <dir>`.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
video = c1.selectbox("Video", vids)
store_id = c2.selectbox("Store", list(stores))
cam_key = c3.selectbox("Camera", list(stores[store_id]["cameras"]))
start = c4.text_input("Clip start time (UTC ISO)", "2026-04-10T20:10:00")
cam = stores[store_id]["cameras"][cam_key]


@st.cache_resource
def open_video(path):
    import cv2
    cap = cv2.VideoCapture(path)
    return cap, cap.get(cv2.CAP_PROP_FPS) or 25.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


cap, fps, n = open_video(video)
sec = st.slider("Time in clip (s)", 0.0, max(0.1, n / fps - 0.1), 0.0, step=0.2)
import cv2  # noqa: E402
import numpy as np  # noqa: E402

cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
ok, frame = cap.read()
if ok:
    h, w = frame.shape[:2]
    for z in cam.get("zones", []) + [cam[k] for k in ("queue_polygon", "payment_polygon") if k in cam]:
        pts = np.array([[int(x * w), int(y * h)] for x, y in z["polygon"]], np.int32)
        cv2.polylines(frame, [pts], True, (0, 200, 255), 2)
        cv2.putText(frame, z["zone_id"], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    if "entry_line" in cam:
        (ax, ay), (bx, by) = cam["entry_line"]
        cv2.line(frame, (int(ax * w), int(ay * h)), (int(bx * w), int(by * h)), (0, 0, 255), 3)
    st.image(frame[:, :, ::-1], width="stretch")

ts = datetime.fromisoformat(start) + timedelta(seconds=sec)
zones = [z["zone_id"] for z in cam.get("zones", [])] + [cam[k]["zone_id"] for k in ("queue_polygon", "payment_polygon") if k in cam]
c1, c2, c3 = st.columns(3)
etype = c1.selectbox("Event", EVENTS)
zone = c2.selectbox("Zone", [""] + zones)
dwell = c3.number_input("Dwell ms (ZONE_DWELL only)", 0, step=1000)
out = ROOT / "training" / "ground_truth" / f"{store_id}.csv"
if st.button(f"➕ Add {etype} at {ts:%H:%M:%S.%f}"[:-3]):
    new = not out.exists()
    with out.open("a", newline="") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["store_id", "camera_id", "event_type", "timestamp", "zone_id", "dwell_ms"])
        wr.writerow([store_id, cam["camera_id"], etype, ts.isoformat() + "Z", zone, dwell])
    st.success("saved")
if out.exists():
    st.dataframe(list(csv.DictReader(out.open())), width="stretch")
