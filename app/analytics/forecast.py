"""Footfall / revenue forecasting and video+POS fusion (after Deepa G et al., ICDSAAI 2026).

The dataset spans a single day per store, so heavy sequence models (LSTM) would overfit.
We use damped Holt exponential smoothing for the hourly footfall series (with an
hour-of-day profile when several days are available) and a small multimodal regression
revenue_h ~ visitors_h + avg_dwell_h + queue_joins_h that quantifies how CCTV-derived
signals explain sales. The interface is model-agnostic, so an LSTM can be dropped in
once weeks of data are collected.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

import numpy as np


def holt_damped(y: list[float], horizon: int, alpha: float = 0.5, beta: float = 0.2,
                phi: float = 0.85) -> tuple[list[float], list[float]]:
    """Returns (forecast, one-step-ahead fitted values)."""
    if not y:
        return [0.0] * horizon, []
    level, trend = y[0], (y[1] - y[0]) if len(y) > 1 else 0.0
    fitted = [y[0]]
    for t in range(1, len(y)):
        pred = level + phi * trend
        fitted.append(pred)
        new_level = alpha * y[t] + (1 - alpha) * pred
        trend = beta * (new_level - level) + (1 - beta) * phi * trend
        level = new_level
    fc, damp = [], 0.0
    for h in range(1, horizon + 1):
        damp += phi ** h
        fc.append(max(0.0, level + damp * trend))
    return fc, fitted


def forecast(hourly: list[dict], horizon: int = 6, open_hours: dict | None = None) -> dict:
    if not hourly:
        return {"horizon_hours": horizon, "footfall": [], "fusion_model": None}
    y = [float(h["entries"]) for h in hourly]
    fc, fitted = holt_damped(y, horizon)
    # hour-of-day profile (multiplicative) when > 1 day is available
    days = {h["hour"].date() for h in hourly}
    profile = None
    if len(days) > 1:
        by_h = defaultdict(list)
        for h in hourly:
            by_h[h["hour"].hour].append(h["entries"])
        overall = np.mean(y) or 1.0
        profile = {k: float(np.mean(v)) / overall for k, v in by_h.items()}
    last = hourly[-1]["hour"]
    o = int(open_hours["open"][:2]) if open_hours else 0
    c = int(open_hours["close"][:2]) if open_hours else 24

    def next_open(ts):  # forecast only hours when the store is open
        ts += timedelta(hours=1)
        while not (o <= ts.hour < c):
            ts += timedelta(hours=1)
        return ts
    resid = np.array(y[1:]) - np.array(fitted[1:]) if len(y) > 1 else np.array([0.0])
    sigma = float(np.std(resid)) if len(resid) else 0.0
    points = []
    ts = last
    for i, v in enumerate(fc, start=1):
        ts = next_open(ts)
        if profile:
            v = v * profile.get(ts.hour, 1.0)
        points.append({"hour": ts, "entries": round(v, 1),
                       "lower": round(max(0.0, v - 1.96 * sigma * np.sqrt(i)), 1),
                       "upper": round(v + 1.96 * sigma * np.sqrt(i), 1)})
    mae = float(np.mean(np.abs(resid))) if len(resid) else 0.0
    return {
        "horizon_hours": horizon, "method": "damped Holt" + (" x hour-of-day profile" if profile else ""),
        "history": [{"hour": h["hour"], "entries": h["entries"], "fitted": round(f, 2)}
                    for h, f in zip(hourly, fitted)],
        "footfall": points, "in_sample_mae": round(mae, 2),
        "fusion_model": fusion_regression(hourly),
    }


def fusion_regression(hourly: list[dict]) -> dict | None:
    rows = [h for h in hourly if h["entries"] or h["transactions"]]
    if len(rows) < 5:
        return None
    feats = ["visitors", "avg_dwell_s", "queue_joins"]
    X = np.array([[h[f] for f in feats] for h in rows], dtype=float)
    yv = np.array([h["revenue_inr"] for h in rows], dtype=float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = np.column_stack([np.ones(len(rows)), (X - mu) / sd])
    coef, *_ = np.linalg.lstsq(Xs, yv, rcond=None)
    pred = Xs @ coef
    ss_res = float(((yv - pred) ** 2).sum())
    ss_tot = float(((yv - yv.mean()) ** 2).sum()) or 1.0
    return {"target": "hourly revenue (INR)", "features": feats, "n_hours": len(rows),
            "intercept": round(float(coef[0]), 2),
            "standardised_coefficients": {f: round(float(c), 2) for f, c in zip(feats, coef[1:])},
            "r2": round(1 - ss_res / ss_tot, 3),
            "interpretation": "INR change in hourly revenue per 1 std-dev increase of each CCTV signal"}
