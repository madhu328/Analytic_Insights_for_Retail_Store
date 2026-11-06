"""Run the detection pipeline on one camera clip.

  python -m pipeline.run_pipeline --store-id STORE_BLR_002 --camera CAM1 \
      --video videos/store1_cam1.mp4 --start-time 2026-04-10T20:10:00Z \
      --out data/pipeline_outputs/store_1/cam1_events.jsonl [--post http://localhost:8000]

Per processed frame: YOLO11m person detections -> ByteTrack IDs -> camera logic
(CAM1/2 zone polygons + dwell, CAM3 entry line + OSNet Re-ID, CAM5 queue + payment)
-> EventEmitter -> JSONL (optionally POSTed to /events/ingest through the bridge).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .emit import EventEmitter
from .processors import build_processor
from .reid import Embedder, ReIDManager, SessionStore

ROOT = Path(__file__).resolve().parent.parent


def load_camera(layout_path: str | Path, store_id: str, camera: str) -> dict:
    layout = json.loads(Path(layout_path).read_text())
    for s in layout["stores"]:
        if s["store_id"] == store_id:
            cams = s["cameras"]
            if camera in cams:
                return cams[camera]
            for cfg in cams.values():  # allow CAM_SHELF_01 style ids too
                if cfg["camera_id"] == camera:
                    return cfg
            raise SystemExit(f"camera {camera} not in layout for {store_id}: {list(cams)}")
    raise SystemExit(f"store {store_id} not in {layout_path}")


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def draw_debug(frame, cam_cfg: dict, tracks, proc):  # pragma: no cover - visual aid
    import cv2
    import numpy as np

    h, w = frame.shape[:2]

    def px(poly):
        return np.array([[int(x * w), int(y * h)] for x, y in poly], np.int32)

    for z in cam_cfg.get("zones", []):
        cv2.polylines(frame, [px(z["polygon"])], True, (0, 200, 255), 2)
        x, y = px(z["polygon"])[0]
        cv2.putText(frame, z["zone_id"], (x + 3, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    for key in ("queue_polygon", "payment_polygon"):
        if key in cam_cfg:
            cv2.polylines(frame, [px(cam_cfg[key]["polygon"])], True, (255, 200, 0), 2)
    if "entry_line" in cam_cfg:
        a, b = px(cam_cfg["entry_line"])
        cv2.line(frame, tuple(a), tuple(b), (0, 0, 255), 3)
    for t in tracks:
        x1, y1, x2, y2 = (int(t.bbox[0] * w), int(t.bbox[1] * h), int(t.bbox[2] * w), int(t.bbox[3] * h))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"#{t.track_id} {t.confidence:.2f}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)
    return frame


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store-id", required=True)
    ap.add_argument("--camera", required=True, help="CAM1 | CAM2 | CAM3 | CAM5 (or camera_id)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--start-time", required=True, help="wall-clock time of the first frame (ISO-8601)")
    ap.add_argument("--layout", default=str(ROOT / "config" / "store_layout.json"))
    ap.add_argument("--out", help="JSONL output path")
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--fps", type=float, default=5.0, help="processing frame rate")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--reid-gallery", help="JSON gallery shared across cameras (CAM3 writes it)")
    ap.add_argument("--reid-threshold", type=float, default=0.80)
    ap.add_argument("--debug-video", help="write an annotated MP4 (zones, lines, tracks) for calibration")
    ap.add_argument("--post", help="Ingest API base URL to POST events to")
    args = ap.parse_args(argv)

    from .tracking import PersonTracker, iter_frames

    cam = load_camera(args.layout, args.store_id, args.camera)
    out = args.out or f"data/pipeline_outputs/{args.store_id}/{args.camera.lower()}_events.jsonl"
    emitter = EventEmitter(args.store_id, cam["camera_id"], out)
    kw = {}
    if cam.get("role") == "entry":
        store = SessionStore.load(args.reid_gallery) if args.reid_gallery else SessionStore()
        kw = {"reid": ReIDManager(store, threshold=args.reid_threshold), "embedder": Embedder()}
        print(f"Re-ID embedder: {kw['embedder'].kind}", file=sys.stderr)
    proc = build_processor(cam, emitter, **kw)
    tracker = PersonTracker(args.model, conf=args.conf, device=args.device)
    t0 = parse_ts(args.start_time)
    writer = None
    n = 0
    for idx, sec, frame in iter_frames(args.video, args.fps):
        tracks = tracker.update(frame, keep_crops=cam.get("role") == "entry")
        proc.update(t0 + timedelta(seconds=sec), tracks)
        n += 1
        if args.debug_video:  # pragma: no cover
            import cv2
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.debug_video, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            writer.write(draw_debug(frame, cam, tracks, proc))
    events = proc.finalize()
    if writer is not None:  # pragma: no cover
        writer.release()
    if cam.get("role") == "entry" and args.reid_gallery:
        kw["reid"].store.save(args.reid_gallery)
    print(f"{n} frames processed -> {len(events)} events -> {out}", file=sys.stderr)
    if args.post:
        from .bridge import post_batches
        print(json.dumps(post_batches(args.post, "/events/ingest", events)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
