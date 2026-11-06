"""Re-identification for CAM3 (doorway): OSNet appearance embeddings -> stable VIS_ ids.

* ``Embedder`` uses OSNet (torchreid, ``osnet_x0_25`` by default) when installed and falls
  back to an HSV colour histogram so the pipeline still runs on CPU-only boxes.
* ``ReIDManager`` keeps a gallery of visitor embeddings (EMA-updated) and matches by cosine
  similarity; a match on a visitor who already exited => REENTRY.
* ``SessionStore`` tracks inside/outside state per visitor and can be persisted to JSON so
  other cameras (CAM1/2/5) can resolve their local track ids against the same gallery.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


class Embedder:
    def __init__(self, model_name: str = "osnet_x0_25", device: str = "cpu"):
        self.kind = "hist"
        try:  # pragma: no cover - heavy optional dependency
            from torchreid.utils import FeatureExtractor

            self._fx = FeatureExtractor(model_name=model_name, device=device)
            self.kind = "osnet"
        except Exception:
            self._fx = None

    def __call__(self, crop) -> Optional[np.ndarray]:
        if crop is None or getattr(crop, "size", 0) == 0:
            return None
        if self._fx is not None:  # pragma: no cover
            v = self._fx([crop[:, :, ::-1]]).cpu().numpy()[0]
        else:
            v = hsv_histogram(crop)
        n = np.linalg.norm(v)
        return v / n if n else None


def hsv_histogram(crop, bins=(8, 4, 4)) -> np.ndarray:
    """Upper/lower-body HSV histograms (clothing colour) as a cheap Re-ID descriptor."""
    import cv2

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv.shape[0]
    parts = [hsv[: h // 2], hsv[h // 2:]]
    feats = [cv2.calcHist([p], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256]).flatten() for p in parts]
    return np.concatenate(feats).astype(np.float32)


@dataclass
class VisitorState:
    visitor_id: str
    embedding: np.ndarray
    inside: bool = False
    last_entry: Optional[datetime] = None
    last_exit: Optional[datetime] = None
    sessions: int = 0


@dataclass
class SessionStore:
    visitors: dict[str, VisitorState] = field(default_factory=dict)
    next_id: int = 1

    def new_visitor(self, emb: np.ndarray) -> VisitorState:
        vid = f"VIS_{self.next_id:05d}"
        self.next_id += 1
        st = VisitorState(vid, emb)
        self.visitors[vid] = st
        return st

    def get_or_create(self, visitor_id: str) -> tuple[VisitorState, bool]:
        """Identity without appearance features (track-id based). Returns (state, existed)."""
        if visitor_id in self.visitors:
            return self.visitors[visitor_id], True
        st = VisitorState(visitor_id, np.zeros(1, np.float32))
        self.visitors[visitor_id] = st
        return st, False

    def save(self, path: str | Path) -> None:
        data = {"next_id": self.next_id, "visitors": [
            {"visitor_id": v.visitor_id, "embedding": v.embedding.tolist(), "inside": v.inside,
             "last_entry": v.last_entry.isoformat() if v.last_entry else None,
             "last_exit": v.last_exit.isoformat() if v.last_exit else None, "sessions": v.sessions}
            for v in self.visitors.values()]}
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "SessionStore":
        p = Path(path)
        if not p.exists():
            return cls()
        d = json.loads(p.read_text())
        st = cls(next_id=d.get("next_id", 1))
        for v in d.get("visitors", []):
            st.visitors[v["visitor_id"]] = VisitorState(
                v["visitor_id"], np.asarray(v["embedding"], dtype=np.float32), v["inside"],
                datetime.fromisoformat(v["last_entry"]) if v["last_entry"] else None,
                datetime.fromisoformat(v["last_exit"]) if v["last_exit"] else None, v["sessions"])
        return st


class ReIDManager:
    def __init__(self, store: Optional[SessionStore] = None, threshold: float = 0.80, ema: float = 0.7):
        self.store = store or SessionStore()
        self.threshold, self.ema = threshold, ema

    def match(self, emb: Optional[np.ndarray]) -> tuple[Optional[VisitorState], float]:
        if emb is None or not self.store.visitors:
            return None, 0.0
        best, best_sim = None, -1.0
        for v in self.store.visitors.values():
            sim = float(np.dot(emb, v.embedding))
            if sim > best_sim:
                best, best_sim = v, sim
        return (best, best_sim) if best_sim >= self.threshold else (None, best_sim)

    def resolve(self, emb: Optional[np.ndarray]) -> tuple[VisitorState, bool, float]:
        """Returns (visitor, is_known, similarity). Unknown embeddings create a new visitor."""
        v, sim = self.match(emb)
        if v is None:
            v = self.store.new_visitor(emb if emb is not None else np.zeros(1, np.float32))
            return v, False, 1.0
        if emb is not None and v.embedding.shape == emb.shape:
            upd = self.ema * v.embedding + (1 - self.ema) * emb
            v.embedding = upd / (np.linalg.norm(upd) or 1.0)
        return v, True, sim
