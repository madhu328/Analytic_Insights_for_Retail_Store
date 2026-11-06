"""FastAPI: REST contract for ingestion and analytics.

Run:  uvicorn app.main:app --port 8000
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from fastapi import Body, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.engine import Engine

from . import __version__, db
from .analytics.anomalies import detect_anomalies
from .analytics.forecast import forecast as run_forecast
from .analytics.funnel import compute_funnel
from .analytics.heatmap import compute_heatmap
from .analytics.insights import hesitation_insights, journey_insights, shelf_quadrant
from .analytics.metrics import compute_metrics
import json

from .config import settings
from .layout import store_layout
from .schemas import Event, IngestResult, PosTransaction, validate_batch
from .service import Analysis, analyse, resolve_window

app = FastAPI(
    title="Store Intelligence API",
    version=__version__,
    description="CCTV events + POS -> sessions, conversion KPIs, funnel, heatmap, anomalies.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def get_engine() -> Engine:
    return db.get_engine()


def _items(payload: Any, key: str) -> list:
    if isinstance(payload, dict) and isinstance(payload.get(key), list):
        return payload[key]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise HTTPException(422, f"expected a list or {{'{key}': [...]}}")


def analysis_dep(
    store_id: str = Query(..., description="e.g. STORE_BLR_002"),
    date_: Optional[date] = Query(None, alias="date", description="YYYY-MM-DD (UTC day)"),
    start: Optional[datetime] = Query(None, description="ISO start (inclusive, naive UTC)"),
    end: Optional[datetime] = Query(None, description="ISO end (exclusive, naive UTC)"),
    window_before_min: Optional[float] = Query(None, ge=0, le=60),
    window_after_min: Optional[float] = Query(None, ge=0, le=60),
    engine: Engine = Depends(get_engine),
) -> Analysis:
    try:
        s, e = resolve_window(date_, start, end)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return analyse(engine, store_id, s, e, window_before_min, window_after_min)


def _envelope(a: Analysis, **body) -> dict:
    return {"store_id": a.store_id, "start": a.start, "end": a.end, **body}


@app.get("/health")
def health(engine: Engine = Depends(get_engine)) -> dict:
    with engine.connect() as c:
        c.exec_driver_sql("SELECT 1")
    return {"status": "ok", "version": __version__}


@app.post("/events/ingest", response_model=IngestResult)
def ingest_events(payload: Any = Body(...), engine: Engine = Depends(get_engine)) -> IngestResult:
    items = _items(payload, "events")
    ok, errors = validate_batch(Event, items, "event_id")
    inserted, dups = db.insert_events(engine, ok) if ok else (0, 0)
    return IngestResult(received=len(items), inserted=inserted, duplicates=dups,
                        rejected=len(errors), errors=errors[:50])


@app.post("/pos/ingest", response_model=IngestResult)
def ingest_pos(payload: Any = Body(...), engine: Engine = Depends(get_engine)) -> IngestResult:
    items = _items(payload, "transactions")
    ok, errors = validate_batch(PosTransaction, items, "transaction_id")
    inserted, dups = db.insert_pos(engine, ok) if ok else (0, 0)
    return IngestResult(received=len(items), inserted=inserted, duplicates=dups,
                        rejected=len(errors), errors=errors[:50])


@app.get("/stores")
def stores(engine: Engine = Depends(get_engine)) -> list[dict]:
    out = db.store_summaries(engine)
    for s in out:
        lay = store_layout(s["store_id"])
        s["name"] = lay.get("name", s["store_id"])
        s["open_hours"] = lay.get("open_hours")
    return out


@app.get("/stores/{store_id}/layout")
def layout(store_id: str) -> dict:
    return store_layout(store_id)


@app.get("/metrics")
def metrics(a: Analysis = Depends(analysis_dep)) -> dict:
    return _envelope(a, **compute_metrics(a.build, a.matches, a.transactions))


@app.get("/funnel")
def funnel(a: Analysis = Depends(analysis_dep),
           level: str = Query("visitor", pattern="^(visitor|session)$")) -> dict:
    return _envelope(a, **compute_funnel(a.build, level))


@app.get("/heatmap")
def heatmap(a: Analysis = Depends(analysis_dep)) -> dict:
    return _envelope(a, **compute_heatmap(a.build, a.store_id))


@app.get("/anomalies")
def anomalies(a: Analysis = Depends(analysis_dep)) -> dict:
    items = detect_anomalies(a.build, a.matches, a.transactions, a.hourly, a.events, a.store_id)
    counts = {k: sum(1 for x in items if x["severity"] == k) for k in ("critical", "warning", "info")}
    return _envelope(a, counts=counts, anomalies=items)


@app.get("/timeseries")
def timeseries(a: Analysis = Depends(analysis_dep)) -> dict:
    return _envelope(a, hourly=a.hourly)


@app.get("/sessions")
def sessions(a: Analysis = Depends(analysis_dep), visitor_id: Optional[str] = None,
             converted: Optional[bool] = None, limit: int = Query(200, ge=1, le=5000),
             offset: int = Query(0, ge=0)) -> dict:
    rows = [s for s in a.build.sessions
            if (visitor_id is None or s.visitor_id == visitor_id)
            and (converted is None or s.converted == converted)]
    return _envelope(a, total=len(rows), sessions=[s.to_dict() for s in rows[offset:offset + limit]])


@app.get("/sessions/{session_id}")
def session_detail(session_id: str, a: Analysis = Depends(analysis_dep)) -> dict:
    for s in a.build.sessions:
        if s.session_id == session_id:
            evs = [e for e in a.events if e["visitor_id"] == s.visitor_id
                   and s.start <= e["timestamp"] <= (s.end or s.last_seen or s.start)]
            d = s.to_dict()
            d["zone_visits"] = [{"zone_id": v.zone_id, "enter": v.enter, "exit": v.exit,
                                 "dwell_s": round(v.dwell_s, 1)} for v in s.zone_visits]
            d["events"] = evs
            return d
    raise HTTPException(404, "session not found")


@app.get("/conversions")
def conversions(a: Analysis = Depends(analysis_dep)) -> dict:
    return _envelope(a, matches=[m.to_dict() for m in a.matches])


@app.get("/insights/journeys")
def journeys(a: Analysis = Depends(analysis_dep)) -> dict:
    return _envelope(a, **journey_insights(a.build, a.store_id))


@app.get("/insights/hesitation")
def hesitation(a: Analysis = Depends(analysis_dep)) -> dict:
    return _envelope(a, **hesitation_insights(a.build))


@app.get("/insights/shelf")
def shelf(a: Analysis = Depends(analysis_dep)) -> dict:
    return _envelope(a, **shelf_quadrant(a.build, a.store_id))


@app.get("/forecast")
def forecast(a: Analysis = Depends(analysis_dep), horizon: int = Query(6, ge=1, le=48)) -> dict:
    return _envelope(a, **run_forecast(a.hourly, horizon, store_layout(a.store_id).get("open_hours")))


@app.get("/evaluation")
def evaluation() -> dict:
    """Latest model / pipeline accuracy reports written by the training toolkit."""
    out = {"target": 0.95}
    for key, name in (("detector", "detector_metrics.json"), ("events", "event_accuracy.json")):
        f = settings.reports_dir / name
        out[key] = json.loads(f.read_text()) if f.exists() else None
    return out


@app.get("/compare")
def compare(engine: Engine = Depends(get_engine)) -> list[dict]:
    """Headline KPIs for every store on its most recent day with CCTV activity."""
    rows = []
    for s in db.store_summaries(engine):
        days = s["dates"]
        if not days:
            continue
        ev_days = [d for d in days if s["first_event"] and s["last_event"]
                   and str(s["first_event"].date()) <= d <= str(s["last_event"].date())] or days
        for day in ev_days:
            st, en = resolve_window(date.fromisoformat(day), None, None)
            a = analyse(engine, s["store_id"], st, en)
            m = compute_metrics(a.build, a.matches, a.transactions)
            rows.append({"store_id": s["store_id"], "name": store_layout(s["store_id"]).get("name", s["store_id"]),
                         "date": day, **{k: m[k] for k in ("visitors", "sessions", "converted_visitors",
                                                           "conversion_rate", "revenue_inr", "attributed_revenue_inr",
                                                           "avg_basket_inr", "avg_session_duration_s",
                                                           "avg_zone_dwell_s")},
                         "queue_abandonment_rate": m["queue"]["abandonment_rate"],
                         "max_queue_depth": m["queue"]["max_queue_depth"],
                         "matched_transactions": m["matched_transactions"], "transactions": m["transactions"]})
    return rows
