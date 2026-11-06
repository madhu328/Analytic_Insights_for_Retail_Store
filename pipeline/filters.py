"""Detection hygiene before camera logic — the cheapest accuracy gains on store CCTV.

* ``exclude_polygons`` (per camera, in store_layout.json): posters, mirrors, glass doors,
  screens and mannequins that YOLO repeatedly mistakes for people.
* ``min_box_area`` / ``max_box_area``: drop far-away fragments and frame-filling glitches.
* ``min_conf``: per-camera confidence floor (ByteTrack still uses low boxes internally).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import anchor_point, point_in_polygon
from .tracking import Track


@dataclass
class DetectionFilter:
    min_box_area: float = 0.002     # fraction of the frame
    max_box_area: float = 0.6
    min_conf: float = 0.0
    exclude_polygons: list = field(default_factory=list)

    @classmethod
    def from_camera(cls, cam_cfg: dict) -> "DetectionFilter":
        f = cam_cfg.get("filters", {})
        return cls(f.get("min_box_area", 0.002), f.get("max_box_area", 0.6), f.get("min_conf", 0.0),
                   [p["polygon"] if isinstance(p, dict) else p for p in cam_cfg.get("exclude_polygons", [])])

    def keep(self, t: Track) -> bool:
        x1, y1, x2, y2 = t.bbox
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if not (self.min_box_area <= area <= self.max_box_area) or t.confidence < self.min_conf:
            return False
        c = anchor_point(t.bbox, "center")
        return not any(point_in_polygon(c, poly) for poly in self.exclude_polygons)

    def __call__(self, tracks: list[Track]) -> list[Track]:
        return [t for t in tracks if self.keep(t)]
