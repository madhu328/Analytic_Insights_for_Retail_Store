"""Pure-python geometry helpers (normalised [0..1] image coordinates)."""
from __future__ import annotations

from typing import Sequence

Point = tuple[float, float]


def point_in_polygon(p: Point, poly: Sequence[Sequence[float]]) -> bool:
    """Ray casting; points on the boundary may go either way."""
    x, y = p
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def side_of_line(p: Point, a: Sequence[float], b: Sequence[float]) -> float:
    """>0 on one side of line a->b, <0 on the other (2D cross product)."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def anchor_point(bbox: Sequence[float], mode: str = "bottom_center") -> Point:
    """bbox = (x1, y1, x2, y2) normalised. Feet point for floor zones, body centre for shelves."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    if mode == "center":
        return cx, (y1 + y2) / 2
    if mode == "torso":
        return cx, y1 + 0.6 * (y2 - y1)
    return cx, y2
