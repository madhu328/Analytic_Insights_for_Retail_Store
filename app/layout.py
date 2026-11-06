"""Loads store_layout.json (zones, cameras, open hours, floor coordinates)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .config import settings

BILLING_ZONE_IDS = {"BILLING", "PAYMENT", "CHECKOUT", "QUEUE"}


@lru_cache(maxsize=4)
def load_layout(path: Optional[str] = None) -> dict:
    p = Path(path or settings.layout_path)
    if not p.exists():
        return {"stores": []}
    return json.loads(p.read_text())


def store_layout(store_id: str) -> dict:
    for s in load_layout().get("stores", []):
        if s["store_id"] == store_id:
            return s
    return {"store_id": store_id, "zones": [], "cameras": {}}


def product_zone_ids(store_id: str) -> list[str]:
    return [z["zone_id"] for z in store_layout(store_id).get("zones", []) if z.get("type") == "product"]


def billing_zone_ids(store_id: str) -> set[str]:
    ids = {z["zone_id"] for z in store_layout(store_id).get("zones", []) if z.get("type") == "billing"}
    return ids | BILLING_ZONE_IDS


def zone_floor_xy(store_id: str) -> dict[str, tuple[float, float]]:
    return {z["zone_id"]: tuple(z["floor_xy"]) for z in store_layout(store_id).get("zones", [])
            if z.get("floor_xy")}
