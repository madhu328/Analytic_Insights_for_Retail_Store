"""Runtime settings, overridable through environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'store_intelligence.db'}"
        )
    )
    layout_path: Path = field(
        default_factory=lambda: Path(os.getenv("STORE_LAYOUT", ROOT / "config" / "store_layout.json"))
    )
    reports_dir: Path = field(default_factory=lambda: Path(os.getenv("REPORTS_DIR", ROOT / "reports")))
    # POS correlation: billing activity must happen this many minutes before the POS timestamp.
    match_window_before_min: float = field(default_factory=lambda: _f("MATCH_WINDOW_BEFORE_MIN", 5))
    # Grace after the POS timestamp (clock skew between CCTV DVR and POS terminal).
    match_window_after_min: float = field(default_factory=lambda: _f("MATCH_WINDOW_AFTER_MIN", 1))
    min_confidence: float = field(default_factory=lambda: _f("MIN_EVENT_CONFIDENCE", 0.0))
    # Behaviour thresholds (from "Customer movements" paper: hesitation = dwell timeout > 60 s, loitering > 2 visits)
    hesitation_dwell_s: float = field(default_factory=lambda: _f("HESITATION_DWELL_S", 60))
    severe_hesitation_dwell_s: float = field(default_factory=lambda: _f("SEVERE_HESITATION_DWELL_S", 180))
    loiter_revisits: int = field(default_factory=lambda: int(_f("LOITER_REVISITS", 2)))
    queue_depth_alert: int = field(default_factory=lambda: int(_f("QUEUE_DEPTH_ALERT", 8)))


settings = Settings()
