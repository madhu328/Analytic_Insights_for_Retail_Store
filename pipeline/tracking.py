"""Per-frame person detection (YOLO11m) + multi-object tracking (ByteTrack).

ByteTrack's two-stage association (high- then low-confidence boxes) keeps IDs stable
through the frequent partial occlusions around shelves and the billing counter.
Ultralytics ships both; this module only adapts its output to a small dataclass so the
processors stay framework-free and unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

PERSON_CLASS = 0


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]  # normalised x1, y1, x2, y2
    confidence: float
    crop: Optional[object] = None  # BGR numpy crop (for Re-ID), optional


class PersonTracker:
    def __init__(self, model: str = "yolo11m.pt", conf: float = 0.35, iou: float = 0.5,
                 device: Optional[str] = None, tracker_cfg: str = "bytetrack.yaml",
                 imgsz: int = 960):
        from ultralytics import YOLO  # lazy: only needed when processing video

        self.model = YOLO(model)
        self.conf, self.iou, self.device, self.tracker_cfg, self.imgsz = conf, iou, device, tracker_cfg, imgsz

    def update(self, frame, keep_crops: bool = False) -> list[Track]:
        h, w = frame.shape[:2]
        res = self.model.track(frame, persist=True, classes=[PERSON_CLASS], conf=self.conf, iou=self.iou,
                               tracker=self.tracker_cfg, device=self.device, imgsz=self.imgsz,
                               verbose=False)[0]
        boxes = res.boxes
        if boxes is None or boxes.id is None:
            return []
        out = []
        for (x1, y1, x2, y2), tid, c in zip(boxes.xyxy.tolist(), boxes.id.int().tolist(), boxes.conf.tolist()):
            crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)].copy() if keep_crops else None
            out.append(Track(int(tid), (x1 / w, y1 / h, x2 / w, y2 / h), float(c), crop))
        return out


def iter_frames(video_path: str, target_fps: float = 5.0) -> Iterator[tuple[int, float, object]]:
    """Yields (frame_index, seconds_from_start, frame) sampled at ~target_fps."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps / target_fps))
    idx = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    yield idx, idx / fps, frame
            idx += 1
    finally:
        cap.release()
